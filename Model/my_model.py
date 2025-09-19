import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import geohash2
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, coalesce
from transformers import GPT2Model, GPT2Config
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType


# --- 단기 선호도 ---
class UserShortPrefMemory(nn.Module):
    """
    Inference 전용 전역 단기메모리 버퍼.
    - 학습 시에는 build_functional_short_pref()로 (B,L,d) 단기 임베딩을 '계산'만 해서 쓰고,
      momentum_update()는 호출하지 않는 것을 권장합니다.
    """
    def __init__(self, num_users: int, dim: int, default_beta: float = 0.5):
        super().__init__()
        self.register_buffer('memory', torch.zeros(num_users, dim))  # (U, d)
        self.default_beta = float(default_beta)

    def forward(self, user_ids) -> torch.Tensor:
        # int, 0-d 텐서, (B,) 모두 지원
        if isinstance(user_ids, int):
            user_ids = torch.tensor([user_ids], device=self.memory.device, dtype=torch.long)
            return self.memory[user_ids][0]  # (d,)
        elif isinstance(user_ids, torch.Tensor) and user_ids.ndim == 0:
            return self.memory[user_ids.view(1)][0]  # (d,)
        else:
            user_ids = user_ids.to(device=self.memory.device, dtype=torch.long)
            return self.memory[user_ids]  # (B, d)

    @torch.no_grad()
    def momentum_update(self,
                        user_ids,
                        new_short_prefs: torch.Tensor,
                        beta: float = None):
        """
        Inference 전용: 전역 버퍼를 모멘텀으로 갱신.
        user_ids: int | 0-d tensor | (B,) LongTensor
        new_short_prefs: (d,) or (B, d)
        """
        b = float(self.default_beta if beta is None else beta)
        b = max(0.0, min(1.0, b))  # clamp

        # 표준화
        if isinstance(user_ids, int):
            user_ids = torch.tensor([user_ids], device=self.memory.device, dtype=torch.long)
        elif isinstance(user_ids, torch.Tensor) and user_ids.ndim == 0:
            user_ids = user_ids.view(1).to(self.memory.device, torch.long)
        else:
            user_ids = user_ids.to(self.memory.device, torch.long)

        v = new_short_prefs.to(self.memory.device, self.memory.dtype)
        if v.ndim == 1:
            v = v.unsqueeze(0)  # (1, d)

        # (B,)와 (B,d) 길이 일치 확인
        if user_ids.numel() != v.size(0):
            # 단일 벡터로 여러 사용자에 동일 업데이트를 의도한 경우 허용
            if v.size(0) == 1:
                v = v.expand(user_ids.numel(), -1)
            else:
                raise ValueError(f"Shape mismatch: user_ids={user_ids.shape}, new_short_prefs={new_short_prefs.shape}")

        # 동일 user_id가 B 내에 중복될 수 있으면 마지막 항으로 덮어쓰기(간단/안전)
        self.memory[user_ids] = b * self.memory[user_ids] + (1.0 - b) * v

    @torch.no_grad()
    def reset_users(self, user_ids):
        if isinstance(user_ids, int):
            user_ids = torch.tensor([user_ids], device=self.memory.device, dtype=torch.long)
        elif isinstance(user_ids, torch.Tensor) and user_ids.ndim == 0:
            user_ids = user_ids.view(1).to(self.memory.device, torch.long)
        else:
            user_ids = user_ids.to(self.memory.device, torch.long)
        self.memory.index_fill_(0, user_ids, 0.0)

    # ===== 학습용(배치) 함수형 단기 메모리 생성기 =====
    @staticmethod
    def build_functional_short_pref(poi_seq_emb: torch.Tensor,
                                    K: int = 2,
                                    beta: float = 0.5) -> torch.Tensor:
        """
        학습 시 사용: 전역 버퍼를 수정하지 않고, 과거 K 스텝을 EMA로 요약한 단기 임베딩 시퀀스를 생성.
        poi_seq_emb: (B, L, d)  # 각 시점의 POI 임베딩(예: E_poi[poi_ids])
        return:      (B, L, d)  # t 시점에서 과거 ≤K 스텝의 EMA (자기 자신 제외; t=0은 0 벡터)
        """
        B, L, d = poi_seq_emb.shape
        out = poi_seq_emb.new_zeros(B, L, d)
        if L <= 1:
            return out
        K = int(max(1, K))
        beta = float(max(0.0, min(1.0, beta)))

        for t in range(1, L):
            # window: (B, w, d), where w = min(K, t)
            w = min(K, t)
            window = poi_seq_emb[:, t - w:t, :]  # 과거 구간만
            # 순차 EMA
            v = window[:, 0, :]
            for k in range(1, w):
                v = beta * v + (1.0 - beta) * window[:, k, :]
            out[:, t, :] = v
        return out


# --- 장기 선호도(학습) ---
class UserLongPrefMemory(nn.Module):
    def __init__(self, num_users: int, dim: int):
        super().__init__()
        self.long_pref_emb = nn.Embedding(num_users, dim)    # 학습될 장기 선호도 

    def forward(self, user_ids) -> torch.Tensor:
        if isinstance(user_ids, int):
            user_ids = torch.tensor([user_ids], device=self.long_pref_emb.weight.device, dtype=torch.long)
            return self.long_pref_emb(user_ids)[0]      # (d,)
        elif isinstance(user_ids, torch.Tensor) and user_ids.ndim == 0:
            return self.long_pref_emb(user_ids.view(1))[0]  # (d,)
        return self.long_pref_emb(user_ids)             # (B, d)    -> 배치단위의 경우
    

# --- 결합 User Embedding ---
class UserEmb_Merge(nn.Module):
    """
    사용자 임베딩 병합기.
    - merge_type: 'gate' (추천), 'concat', 'sum'
    - short_dim, long_dim이 달라도 내부에서 out_dim으로 projection 후 병합
    - context(optional): 시간/세션길이/Δt 등 추가 피처로 게이트를 더 똑똑하게 만들 수 있음
    """
    def __init__(self,
                 short_dim: int,
                 long_dim: int,
                 out_dim: int = 768,
                 merge_type: str = "gate",
                 context_dim: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.merge_type = merge_type.lower()
        self.proj_s = nn.Linear(short_dim, out_dim)
        self.proj_l = nn.Linear(long_dim, out_dim)
        self.ln_s = nn.LayerNorm(out_dim)
        self.ln_l = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

        if self.merge_type == "concat":
            self.proj_out = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            )
        elif self.merge_type == "gate":
            # 게이트: α ∈ (0,1), 입력은 [s,l,(ctx)]
            in_g = out_dim * 2 + context_dim
            self.gate = nn.Sequential(
                nn.Linear(in_g, max(128, in_g // 2)),
                nn.ReLU(),
                nn.Linear(max(128, in_g // 2), 1),
                nn.Sigmoid()
            )
        elif self.merge_type == "sum":
            # sum은 단순합. 차원만 맞추면 별도 모듈 불필요
            pass
        else:
            raise ValueError("merge_type must be one of {'gate','concat','sum'}")

    def forward(self,
                short_emb: torch.Tensor,  # (B, short_dim)
                long_emb: torch.Tensor,   # (B, long_dim)
                context: torch.Tensor = None  # (B, context_dim) or None
                ) -> torch.Tensor:        # (B, out_dim)
        s = self.drop(self.ln_s(self.proj_s(short_emb)))  # (B, out_dim)
        l = self.drop(self.ln_l(self.proj_l(long_emb)))   # (B, out_dim)

        if self.merge_type == "concat":
            x = torch.cat([s, l], dim=-1)                # (B, 2*out_dim)
            u = self.proj_out(x)                         # (B, out_dim)
            return u

        elif self.merge_type == "gate":
            feats = [s, l]
            if context is not None:
                feats.append(context)
            g = self.gate(torch.cat(feats, dim=-1))      # (B, 1), α
            u = g * s + (1.0 - g) * l                    # (B, out_dim)
            return self.drop(u)

        elif self.merge_type == "sum":
            # 단순 가중합을 원하면 외부에서 λ로 조합하거나 여기서 스칼라 파라미터를 학습해도 됨
            u = s + l
            return self.drop(u)
        

# POI 임베딩
class POIEmbedding(nn.Module):
    def __init__(self, num_pois: int, dim: int):
        super().__init__()
        self.poi_emb = nn.Embedding(num_pois, dim)    # POI 임베딩

    def forward(self, poi_ids: torch.Tensor):
        poi_emb = self.poi_emb(poi_ids)
        return poi_emb

# 각 POI 그래프를 PyG Data 객체로 변환
def df_to_pyg_data(
    df: pd.DataFrame,
    poi_id2idx: dict,
    src_col: str = "src",
    dst_col: str = "dst",
    weight_col: str = "weight",
    device: torch.device | None = None,
) -> Data:
    # ID → index 매핑
    s = df[src_col].map(poi_id2idx)
    t = df[dst_col].map(poi_id2idx)
    w = df[weight_col] if weight_col in df.columns else pd.Series([1.0]*len(df))

    # 매핑이 안 된 행 제거
    mask = (~s.isna()) & (~t.isna())
    s, t, w = s[mask].astype(int).values, t[mask].astype(int).values, w[mask].astype(float).values

    edge_index = torch.tensor([s, t], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)

    # 중복 엣지 합치기(coalesce)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, m=len(poi_id2idx), n=len(poi_id2idx), reduce='mean')

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=len(poi_id2idx))
    if device is not None:
        data = data.to(device)
        
    return data

# GCN 학습    
class POIEncoderGCN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hid_dim)

    def forward(self, x: torch.Tensor, data) -> torch.Tensor:
        """
        x: 초기 POI 임베딩 (num_nodes, in_dim)
        data: PyG Data 객체 (edge_index, edge_weight 포함)
        return: 최종 POI 임베딩 (num_nodes, out_dim)
        """
        x = self.conv1(x, data.edge_index, data.edge_weight)
        x = torch.relu(x)
        x = self.dropout(self.ln1(x))   # 첫 번째 레이어 후 Dropout
        x = self.conv2(x, data.edge_index, data.edge_weight)
        return x
    

# POI 임베딩 병합
class POIMergeGate(nn.Module):
    """
    3-view POI 임베딩 병합기.
    - 입력: z_space, z_time, z_traj (모두 [N, d])
    - 선택: x0 (초기 POI 임베딩), ctx (메타/카테고리 등) -> 게이트에 컨텍스트로 사용 가능
    - 출력: z_merged ([N, d])
    """
    def __init__(self, dim: int, ctx_dim: int = 0, dropout: float = 0.1,
                 use_residual: bool = True, residual_alpha: float = 0.2):
        super().__init__()
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha

        in_g = dim * 3 + ctx_dim   # [z_s, z_t, z_r, (ctx)]로 게이트 산출
        h = max(128, in_g // 2)

        self.gate = nn.Sequential(
            nn.Linear(in_g, h), nn.ReLU(),
            nn.Linear(h, 3)  # 3개 뷰 점수
        )
        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        # 출력 안정화용 (선택)
        self.out_ln = nn.LayerNorm(dim)

        # 초기 바이어스 살짝 주고 싶으면 아래 주석 해제 (초기엔 traj 쪽을 약간 더 보게)
        # with torch.no_grad():
        #     self.gate[-1].bias[:] = torch.tensor([0.0, 0.0, 0.2])

    def forward(self, z_space: torch.Tensor, z_time: torch.Tensor, z_traj: torch.Tensor,
                x0: torch.Tensor | None = None, ctx: torch.Tensor | None = None):
        # 1) 정규화로 스케일 맞춤
        s = self.drop(self.norm_s(z_space))
        t = self.drop(self.norm_t(z_time))
        r = self.drop(self.norm_r(z_traj))

        # 2) 게이트 점수 → softmax 가중치
        feats = [s, t, r]
        if ctx is not None:
            feats.append(ctx)
        g_in = torch.cat(feats, dim=-1)                 # [N, 3d(+ctx)]
        scores = self.gate(g_in)                        # [N, 3]
        alphas = F.softmax(scores, dim=-1)              # [N, 3]
        a_s, a_t, a_r = alphas[:, 0:1], alphas[:, 1:2], alphas[:, 2:3]

        # 3) 가중합
        z = a_s * s + a_t * t + a_r * r                 # [N, d]

        # 4) (옵션) 초기 임베딩 잔차 연결
        if self.use_residual and x0 is not None:
            z = (1.0 - self.residual_alpha) * z + self.residual_alpha * x0

        return self.out_ln(z)


# Category 임베딩
class CatEmbedding(nn.Module):
    def __init__(self, num_cats: int, dim: int):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cats, dim)

    def forward(self, cat_ids: torch.Tensor):
        cat_emb = self.cat_emb(cat_ids)
        return cat_emb


# 시간 임베딩
def _cyc_emb(x: torch.Tensor, period: int, K: int = 2) -> torch.Tensor:
    """
    주기(period)를 가진 정수/실수 시계열 x를 사인/코사인으로 임베딩.
    x: (...,)     // 예: weekday(0~6), hour(0~23), minute(0~59)
    return: 2K 차원 (..., 2K)
    """
    x = x.float()
    outs = []
    for k in range(1, K + 1):
        ang = 2 * math.pi * k * x / period
        outs.append(torch.sin(ang))
        outs.append(torch.cos(ang))
    return torch.stack(outs, dim=-1)  # (..., 2K)


class TimeEmbedding(nn.Module):
    """
    weekday, hour, holiday 정보를 시간 임베딩으로 변환
    """
    def __init__(self, dim: int, k_weekday: int=2, k_hour: int=2):
        super().__init__()
        self.kw = k_weekday
        self.kh = k_hour
        self.tdim = dim
        
        self.bdim = 2 * (k_weekday + k_hour) + 1  # holiday 포함 (기본 시간 특징 차원)
        
        # 나머지 임베딩과 차원 일치
        self.proj = nn.Sequential(
            nn.Linear(self.bdim, self.tdim),
            nn.ReLU(),
            nn.LayerNorm(self.tdim)
        )

    def forward(self, weekday: torch.Tensor, hour: torch.Tensor, holiday: torch.Tensor) -> torch.Tensor:
        """
        return: (..., self.tdim)
        (배치/시퀀스/단일 모두 OK: 입력 모양을 그대로 유지한 채 마지막 차원만 확장됩니다)
        """
        e_w = _cyc_emb(weekday, period=7,  K=self.kw)   # 체크인 요일 임베딩
        e_h = _cyc_emb(hour,    period=24, K=self.kh)   # 체크인 시간 임베딩

        # Holiday는 이진 스칼라 그대로 사용
        e_ho = holiday.float().unsqueeze(-1)            # (..., 1)
        
        x = torch.cat([e_w, e_h, e_ho], dim=-1)         # (..., bdim)

        # 최종 차원 조절
        x = self.proj(x)                                # (..., tdim)

        return x
    
# 최종 체크인 임베딩 생성
class CheckInFusion(nn.Module):
    """
    (poi, time, space, cat) 임베딩을 병합해 check-in 토큰 임베딩을 만듭니다.
    - 기본: concat → FC → 768
    - 옵션(gate='softmax'): 각 파트의 스칼라 가중치를 학습해 동적으로 강조/완화
    입력 텐서는 (B, L, d) 또는 (B, d)를 지원합니다.
    """
    def __init__(self,
                 d_poi: int,
                 d_time: int,
                 d_space: int,
                 d_cat: int,
                 out_dim: int = 768,
                 dropout: float = 0.1,
                 gate: str | None = None   # None | 'softmax' (스칼라 게이트)
                 ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.ln_p = nn.LayerNorm(d_poi)
        self.ln_t = nn.LayerNorm(d_time)
        self.ln_s = nn.LayerNorm(d_space)
        self.ln_c = nn.LayerNorm(d_cat)

        fused_in = d_poi + d_time + d_space + d_cat

        # 게이트(선택): 파트별 스칼라 가중치
        self.gate_type = gate
        if gate == 'softmax':
            # 파트별 통계를 얕게 요약해서 4개 점수 산출
            g_hid = max(64, fused_in // 4)
            self.gate_mlp = nn.Sequential(
                nn.Linear(fused_in, g_hid),
                nn.ReLU(),
                nn.Linear(g_hid, 4)  # poi, time, space, cat
            )

        # 최종 체크인 임베딩 생성 -> concat → FC → out_dim
        self.proj = nn.Sequential(
            nn.Linear(fused_in, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim)
        )

    def _ensure_3d(self, x):
        # (B, d) -> (B, 1, d)로 승격해 시퀀스/배치 코드 통일
        if x.dim() == 2:
            return x.unsqueeze(1), True
        return x, False

    def forward(self,
                e_poi: torch.Tensor,   # (B, L, d_poi) or (B, d_poi)
                e_time: torch.Tensor,  # (B, L, d_time)
                e_space: torch.Tensor, # (B, L, d_space)
                e_cat: torch.Tensor    # (B, L, d_cat)
                ) -> torch.Tensor:     # (B, L, out_dim) or (B, out_dim)
        # 입력 차원 통일
        e_poi, squeeze_p = self._ensure_3d(e_poi)
        e_time, _ = self._ensure_3d(e_time)
        e_space, _ = self._ensure_3d(e_space)
        e_cat, _ = self._ensure_3d(e_cat)

        # 파트별 정규화 + 드롭아웃
        p = self.dropout(self.ln_p(e_poi))
        t = self.dropout(self.ln_t(e_time))
        s = self.dropout(self.ln_s(e_space))
        c = self.dropout(self.ln_c(e_cat))

        # [B, L, dp+dt+ds+dc]
        fused = torch.cat([p, t, s, c], dim=-1)

        # (선택) 파트 가중치 게이팅
        if self.gate_type == 'softmax':
            # 간단히 파트 concat 자체를 요약해서 4개 점수
            scores = self.gate_mlp(fused)                  # (B, L, 4)
            alphas = F.softmax(scores, dim=-1)             # (B, L, 4)
            a_p, a_t, a_s, a_c = [alphas[..., i:i+1] for i in range(4)]
            # 파트별 가중 적용 후 다시 concat
            fused = torch.cat([a_p * p, a_t * t, a_s * s, a_c * c], dim=-1)

        # concat → FC → out_dim(=768)
        out = self.proj(fused)  # (B, L, out_dim)

        # 입력이 (B, d)였으면 (B, out_dim)으로 되돌림
        if squeeze_p:
            out = out.squeeze(1)
        return out
    
    
# PFA 구현
class PFA(nn.Module):
    """
    GPT-2를 PFA 스타일로 학습:
    - 앞쪽 (gpt_layers - U) 블록: 거의 동결 (LN만 학습 옵션)
    - 마지막 U 블록: 어텐션 위주 학습(기본). LoRA 가능.
    - 필요 시 MLP 동결 유지(권장) 또는 해제도 가능.
    - 다음 토큰을 예측
    """
    def __init__(self,
                 gpt_name: str = "gpt2",
                 gpt_layers: int = 12,     # 사용할 블록 수 (<= 사전학습 모델 깊이) -> (원래 12층)
                 U: int = 3,               # 마지막 U개 블록을 더 적극적으로 학습
                 train_early_ln: bool = True,        # 앞쪽 블록에서 ln_1/ln_2 학습 허용
                 train_pos_emb_early: bool = False,  # 포지셔널 임베딩 학습 여부 앞단에서 학습할지 여부
                 train_final_ln: bool = True,        # 최상단 ln_f(출력 레이어노름)를 학습 여부
                 train_last_mlp: bool = False,       # 마지막 U 블록의 MLP를 학습할지 (기본 False 권장)
                 use_lora: bool = False,    # LoRA 사용 여부 (권장)
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 lora_targets=("c_attn", "c_proj"),  # GPT-2에서 어텐션 관련 모듈
                 output_attentions: bool = False,
                 output_hidden_states: bool = False):
        super().__init__()

        # GPT-2 모델 불러오기
        self.gpt = GPT2Model.from_pretrained(
            gpt_name,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        assert gpt_layers <= self.gpt.config.n_layer, "gpt_layers가 모델 깊이보다 클 수 없습니다."
        self.gpt.h = self.gpt.h[:gpt_layers]  # Transformer 블록 수 맞춤
        self.gpt.config.n_layer = gpt_layers  # config도 맞춰줌
        self.last_start = gpt_layers - U      # 마지막 U개 블록의 시작 인덱스

        # 1) 기본적으로 전체 동결
        for p in self.gpt.parameters():
            p.requires_grad = False

        # 2) 앞쪽 블록들: layer norm만 학습 허용(옵션)
        if train_early_ln:
            for li in range(0, self.last_start):
                blk = self.gpt.h[li]
                for name, p in blk.named_parameters():
                    if name.startswith("ln_"):   # ln_1, ln_2
                        p.requires_grad = True

        # 3) positional embedding(wpe)와 token embedding(wte)
        #    - 보통 동결이 안정적. (필요 시만 켜세요)
        if train_pos_emb_early:
            for p in self.gpt.wpe.parameters():
                p.requires_grad = True
        # wte는 inputs_embeds를 직접 주므로 사용하지 않지만, 혹시 쓸 경우를 대비해 기본은 동결 유지

        # 4) 마지막 U개 블록: 어텐션 위주 학습
        for li in range(self.last_start, gpt_layers):
            blk = self.gpt.h[li]
            for name, p in blk.named_parameters():
                if ("attn" in name) or (train_last_mlp and "mlp" in name):
                    p.requires_grad = True
                # layer norm은 기본 동결 or 위에서 켰으면 이미 True일 수 있음

        # 5) 최종 레이어노름(탑레벨 ln_f)
        if train_final_ln:
            for p in self.gpt.ln_f.parameters():
                p.requires_grad = True

        # 6) LoRA 장착 (권장). 전 블록에 장착 후, 마지막 U개 블록의 LoRA만 학습 허용
        self.use_lora = use_lora
        if use_lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_targets),  # 예: c_attn, c_proj
            )
            self.gpt = get_peft_model(self.gpt, peft_cfg)
            # LoRA 모듈은 target_modules에 모두 삽입됨 → 마지막 U 블록만 학습하도록 필터링
            for name, p in self.gpt.named_parameters():
                if "lora_" in name:
                    # 이름에 h.{li}. 이 포함되면 해당 블록의 LoRA
                    keep = any(f"h.{li}." in name for li in range(self.last_start, gpt_layers))
                    p.requires_grad = keep  # 마지막 U 블록의 LoRA만 학습

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        inputs_embeds: (B, L, H)  # 예: H=768
        """
        out = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, L, H)

    def trainable_parameters(self):
        """ Optimizer에 물릴 파라미터만 반환 """
        return [p for p in self.gpt.parameters() if p.requires_grad]

# 최종 Next POI 예측기
class NextPOIWithPFA(nn.Module):
    def __init__(self, pfa: PFA, num_pois: int):
        super().__init__()
        self.pfa = pfa   # PFA 모듈
        self.readout = nn.Linear(self.pfa.gpt.config.hidden_size, num_pois)   # 최종 POI 예측

    # 다음 토큰 예측용 시프트드 크로스엔트로피 손실
    @staticmethod
    def shifted_ce_loss(logits, labels, mask):
        B, L, V = logits.shape
        pred = logits[:, :-1, :].contiguous()
        gold = labels[:, 1:].contiguous()
        m    = mask[:, 1:].contiguous().float()
        loss = nn.functional.cross_entropy(pred.reshape(-1, V), gold.reshape(-1),
                                           reduction="none").reshape(B, L-1)
        return (loss * m).sum() / m.sum().clamp_min(1.0)

    def forward(self, inputs_embeds, attention_mask, labels=None):
        h = self.pfa(inputs_embeds, attention_mask)     # (B, L, H)
        logits = self.readout(h)                        # (B, L, V)
        if labels is None:
            return logits, None
        loss = self.shifted_ce_loss(logits, labels, attention_mask)
        return logits, loss

    def trainable_parameters(self):
        return self.pfa.trainable_parameters() + list(self.readout.parameters())
