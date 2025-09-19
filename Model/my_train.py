import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss
    
    
def train(args):
    args.save_dir = increment_path(Path(args.save_dir) / args.name, exist_ok=args.exist_ok)    # 저장 디렉토리 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    
    # %% ======================================= Load data =======================================
    logging.info("Loading data...")
    # 여기서는 training만 할거니까 training data만 불러옴
    train_df = pd.read_csv(args.train_data_path)
    val_df = pd.read_csv(args.val_data_path)
    
    # POI 그래프 로드
    logging.info("Loading POI graphs...")
    poi_space_G = pd.read_csv(args.poi_space_graph)  # POI 인접성 공간 그래프
    poi_time_G = pd.read_csv(args.poi_time_graph)    # POI 유사 체크인 시간 그래프
    poi_traj_G = pd.read_csv(args.poi_traj_graph)    # POI 전이 그래프
    
    # Geohash 임베딩 로드
    logging.info("Loading geohash embeddings...")
    space_emb = pd.read_csv(args.geohash_embedding)
    space_emb = space_emb.rename(columns={space_emb.columns[0]: "geohash"})   # 첫 열 이름을 'geohash'로 변경 
    
    # POI, 사용자, 카테고리 종류 확인
    poi_ids = pd.concat([train_df['PoiId'], val_df['PoiId']]).unique()
    cat_ids = pd.concat([train_df['PoiCategoryId'], val_df['PoiCategoryId']]).unique()
    user_list = list(train_df['UserId'].unique())     # 일단은 training 데이터의 사용자들만 확인
    user_set = set(user_list)
    
    poi_id2idx = {pid: idx for idx, pid in enumerate(poi_ids)}    # POI ID -> index
    cat_id2idx = {cid: idx for idx, cid in enumerate(cat_ids)}    # 카테고리 ID -> index

    num_pois = len(poi_ids)
    num_users = len(train_df['UserId'].unique())     # 일단은 training 데이터의 사용자들만 확인
    num_cats = len(cat_ids)
    logging.info(f"Number of POIs: {num_pois}, Number of users: {num_users}, Number of categories: {num_cats}")

    # Train + Val에 등장한 POI 정보 (POI id, cat, lat, lon) 저장
    poi_info = pd.concat([
                            train_df[['PoiId', 'PoiCategoryId', 'Latitude', 'Longitude']],
                            val_df[['PoiId', 'PoiCategoryId', 'Latitude', 'Longitude']]
                        ]).drop_duplicates().reset_index(drop=True)
    
    # 사용자 단기 선호도 저장 dict 생성
    user_short_pre_dict = {uid: None for uid in user_set}

    logging.info(f"poi_info shape: {poi_info.shape}")

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        """
        trajectory 데이터를 모델이 학습할 수 있는 시퀀스 형태로 변환
        """
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []   # traj id 저장
            self.input_seqs = []  # 각 traj id 별 input 시퀀스 저장
            self.label_seqs = []  # 각 traj id 별 label 시퀀스 저장

            for traj_id in tqdm(set(train_df['TrajectoryId'].tolist())):
                traj_df = train_df[train_df['TrajectoryId'] == traj_id]    # traj_id 별로 그룹핑
                poi_ids = traj_df['PoiId'].to_list()                       # 해당 traj_id 내 check-in POI 리스트     
                poi_idxs = [poi_id2idx[each] for each in poi_ids]          # POI ID -> index
                
                # 시간 정보 저장
                weekday = traj_df['Weekday'].to_list()
                hours = traj_df['Hour'].to_list()
                holiday = traj_df['Holiday'].to_list()
                
                input_seq = []
                label_seq = []
                # shift 하면서 input, label 시퀀스 생성 -> (p1, p2, p3, p4)일때 input_Seq = [p1, p2, p3], label_seq = [p2, p3, p4] (뒤에 mask로 입력 시퀀스 생성)
                # (poi_idx, (weekday, hours, holiday)) 형태로 저장
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], (weekday[i], hours[i], holiday[i])))
                    label_seq.append((poi_idxs[i + 1], (weekday[i + 1], hours[i + 1], holiday[i + 1])))

                if len(input_seq) < args.short_traj_thres:   # 너무 짧은 시퀀스는 무시 (2개 이하 체크인)
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        """
        Test용 데이터 셋 생성
        """
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['TrajectoryId'].tolist())):
                user_id = traj_id.split('_')[0]      # 사용자 정보 먼저 추출

                # 일단 신규 사용자는 무시
                if user_id not in user_set:
                    continue

                # Get POIs idx in this trajectory
                traj_df = df[df['TrajectoryId'] == traj_id]
                poi_ids = traj_df['PoiId'].to_list()
                
                # 시간 정보 저장
                weekdays = traj_df['Weekday'].to_list()
                hours = traj_df['Hour'].to_list()
                holidays = traj_df['Holiday'].to_list()
                
                poi_idxs = []

                # Train 데이터에 있는 POI인지 확인 -> 여기서는 val poi 들도 같이 만들었으니까 val에만 있는 poi도 들어감
                for each in poi_ids:
                    if each in poi_id2idx.keys():
                        poi_idxs.append(poi_id2idx[each])
                    else:
                        continue

                # Construct input seq and label seq (Val에서는 모든 시점의 예측 결과를 확인)
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], (weekdays[i], hours[i], holidays[i])))
                    label_seq.append((poi_idxs[i + 1], (weekdays[i + 1], hours[i + 1], holidays[i + 1])))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])
    
    # %% ====================== Define dataloader ======================
    # Traj 단위로 배치 생성 -> (batch당 Traj 개수는 args.batch) -> padding은 배치 내에서 진행하면 됨
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)    # Trajectory 구성
    val_dataset = TrajectoryDatasetVal(val_df)          # Trajectory 구성

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)
    
    # %% ====================== Build Models ======================
    print('Building models...')
    
    
    
    
    
    
    
    
    
    
    
    
    
    