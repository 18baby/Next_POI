"""Parsing the parameters."""
import argparse

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run My Model")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')
    # Data
    parser.add_argument('--data-train',
                        type=str,
                        default='../data/nyc/raw/NYC_train.csv',
                        help='Training data path')
    parser.add_argument('--data-val',
                        type=str,
                        default='../data/nyc/raw/NYC_val.csv',
                        help='Validation data path')
    
    # Graph data
    parser.add_argument('--poi_space_graph',
                        type=str,
                        default='../data/nyc/graph/nyc_space_graph_01km.csv',
                        help='POI space graph path')
    parser.add_argument('--poi_time_graph',
                        type=str,
                        default='../data/nyc/graph/nyc_time_graph_09.csv',
                        help='POI time graph path')
    parser.add_argument('--poi_traj_graph',
                        type=str,
                        default='../data/nyc/graph/nyc_traj_graph.csv',
                        help='POI trajectory graph path')
    
    # Space embedding
    parser.add_argument('--geohash_embedding',
                        type=str,
                        default='../data/nyc/graph/nyc_geohash_gat_space_embedding_320.csv',    # 여기서 첫번째 열은 geohash -> 두번째 열부터 임베딩 벡터
                        help='Geohash embedding path')
    
    # Other data
    parser.add_argument('--short-traj-thres',                 # 너무 짧은 시퀀스 제거
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')
    parser.add_argument('--time-units',                       # 시간 단위 (0.5시간 단위면 48, 1시간 단위면 24)
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')

    # Model hyper-parameters   
    """
    * GPT2(small) -> 각 입력 토큰 크기 768차원  -> (POI, space, time, cat) -> (384, 256, 64, 64)
    * LLAMA2-7B   -> 각 입력 토큰 크기 4096차원 -> (POI, space, time, cat) -> (2048, 1365, 512, 128)
    """
    parser.add_argument('--LLM_model',
                        type=str,
                        default='gpt2',
                        help='LLM model name')
    parser.add_argument('--poi_dim',
                        type=int,
                        default=384,
                        help='POI embedding dimensions')            # POI 임베딩 차원 -> (traj, temporal, spatial, cat) 모두 동일한 차원
    parser.add_argument('--time_dim',
                        type=int,
                        default=64,
                        help='Check-in time embedding dimensions')  # time은 상대적으로 많은 정보가 없으니까 수정
    parser.add_argument('--space_dim',
                        type=int,
                        default=256,
                        help='POI space dimensions')
    parser.add_argument('--cat_dim',
                        type=int,
                        default=64,
                        help='Category embedding dimensions')
    parser.add_argument('--user_short_dim',                                # 사용자 단기 선호도 임베딩 크기
                        type=int,
                        default=256,
                        help='User short Preference embedding dimensions')
    parser.add_argument('--user_long_dim',                                 # 사용자 장기 선호도 임베딩 크기       
                        type=int,
                        default=768,
                        help='User long Preference embedding dimensions')
    parser.add_argument('--input_tok_dim',
                        type=int,
                        default=768,
                        help='Input token embedding dimensions')     # LLM 모델 입력 차원 (user, check-in time) 모두 동일 차원

    # GCN hyper-parameters
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=int,
                        default=128,
                        help='List of hidden dims for gcn layers')
    
    # Fusion hidden size
    parser.add_argument('--fuse-hid-dim',
                        type=int,
                        default=1024,
                        help='Hidden dimensions for fusing embeddings')

    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')

    return parser.parse_args()
