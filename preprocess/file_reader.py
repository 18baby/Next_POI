import os.path as osp
import pickle
import pandas as pd
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import logging
import ast


class FileReaderBase:
    root_path = '/disk1/jupyter/hongju/Next_POI/Next_POI_data_Preprocess/data'

    @classmethod
    def read_dataset(cls, file_name, dataset_name):
        raise NotImplementedError

# raw 데이터 불러오기
class FileReader(FileReaderBase):
    @classmethod
    def read_dataset(cls, file_name, dataset_name):
        """
        input: 
            cls: FileReader 클래스
            file_name: 파일 이름
            dataset_name: 데이터셋 이름
        output:
            df: 데이터프레임
        """
        file_path = osp.join(cls.root_path, dataset_name, 'raw', file_name)
        if dataset_name == 'ca':
            df = pd.read_csv(file_path, sep=',', iterator=False)
            df['LocalTime'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
            # 내가 수정
            # 1단계: 문자열 → 리스트로 변환
            df['PoiCategoryId'] = df['PoiCategoryId'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[{") else x)
            df['PoiCategoryName'] = df['PoiCategoryId'].apply(lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 and 'name' in x[0] else None)
            df['PoiCategoryId'] = df['PoiCategoryId'].apply(lambda x: int(x[0]['url'].split('/')[-1]) if isinstance(x, list) and len(x) > 0 and 'url' in x[0] else None)
            df['PoiCategoryId'] = pd.to_numeric(df['PoiCategoryId'], errors='coerce').astype('Int64')
        else:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1', names=[
                'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryName', 'Latitude', 'Longitude', 'TimezoneOffset',
                'UTCTime'
            ], iterator=False)
            df['UTCTime'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y"))
            df['LocalTime'] = df['UTCTime'] + df['TimezoneOffset'].apply(lambda x: timedelta(hours=x/60))
        df['Weekday'] = df['LocalTime'].apply(lambda x: x.weekday())
        df['Hour'] = df['LocalTime'].apply(lambda x: x.hour)
        df['Day'] = df['LocalTime'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df['UserRank'] = df.groupby('UserId')['LocalTime'].rank(method='first')

        logging.info(
            f'[Preprocess - Load Raw Data] min LocalTime: {min(df["LocalTime"])}, '
            f'max LocalTime: {max(df["LocalTime"])}, #User: {df["UserId"].nunique()}, '
            f'#POI: {df["PoiId"].nunique()}, #check-in: {df.shape[0]}'
        )
        return df

    @classmethod
    def do_filter(cls, df, poi_min_freq, user_min_freq):
        """
        input: 
            df: 데이터프레임
            poi_min_freq: POI 최소 횟수
            user_min_freq: User 최소 횟수
        output:
            df: 데이터프레임
        """
        poi_count = df.groupby('PoiId')['UserId'].count().reset_index()                          # 각 POI들이 check-in 된 횟수
        df = df[df['PoiId'].isin(poi_count[poi_count['UserId'] > poi_min_freq]['PoiId'])]        # 최소 횟수 이상 check-in 된 POI만 남기기
        user_count = df.groupby('UserId')['PoiId'].count().reset_index()                         # 각 User들이 check-in 된 횟수
        df = df[df['UserId'].isin(user_count[user_count['PoiId'] > user_min_freq]['UserId'])]    # 최소 횟수 이상 check-in 된 User만 남기기

        logging.info(
            f"[Preprocess - Filter Low Frequency User] User count: {len(user_count)}, "
            f"Low frequency user count: {len(user_count[user_count['PoiId'] <= user_min_freq])}, "
            f"ratio: {len(user_count[user_count['PoiId'] <= user_min_freq]) / len(user_count):.5f}"
        )
        logging.info(
            f"[Preprocess - Filter Low Frequency POI] POI count: {len(poi_count)}, "
            f"Low frequency POI count: {len(poi_count[poi_count['UserId'] <= poi_min_freq])}, "
            f"ratio: {len(poi_count[poi_count['UserId'] <= poi_min_freq]) / len(poi_count):.5f}"
        )
        return df

    @classmethod
    def split_train_test(cls, df, is_sorted=False):
        """
        input: 
            df: 데이터프레임
            is_sorted: 정렬 여부
        output:
            df: 데이터프레임 SplitTag 추가
        """
        if not is_sorted:
            df = df.sort_values(by=['UserId', 'LocalTime'], ascending=True)      # UserId와 LocalTime을 기준으로 정렬

        df['SplitTag'] = 'train'
        total_len = df.shape[0]
        validation_index = int(total_len * 0.8)
        test_index = int(total_len * 0.9)
        df = df.sort_values(by='LocalTime', ascending=True)          # LocalTime을 기준으로 정렬 -> 전체 check-in을 시간 순으로 정렬해 데이터 분할
        df.loc[df.index[validation_index:test_index], 'SplitTag'] = 'validation'
        df.loc[df.index[test_index:], 'SplitTag'] = 'test'
        
        df['UserRank'] = df.groupby('UserId')['LocalTime'].rank(method='first')  # UserId별로 check-in 순서 부여

        # Filter out check-in records when their gaps with thier previous check-in and later check-in are both larger than 24 hours
        df = df.sort_values(by=['UserId', 'LocalTime'], ascending=True)
        isolated_index = []
        for idx, diff1, diff2, user, user1, user2 in zip(
            df.index,
            df['LocalTime'].diff(1),    # 이전 check-in과의 시간 차이
            df['LocalTime'].diff(-1),   # 이후 check-in과의 시간 차이
            df['UserId'],
            df['UserId'].shift(1),          # 이전 check-in의 UserId
            df['UserId'].shift(-1)          # 이후 check-in의 UserId
        ):
            if pd.isna(diff1) and abs(diff2.total_seconds()) > 86400 and user == user2:    # 이전 check-in이 없고 이후 check-in과의 시간 차이가 24시간 이상인 경우
                isolated_index.append(idx)
            elif pd.isna(diff2) and abs(diff1.total_seconds()) > 86400 and user == user1:  # 이후 check-in이 없고 이전 check-in과의 시간 차이가 24시간 이상인 경우
                isolated_index.append(idx)
            if abs(diff1.total_seconds()) > 86400 and abs(diff2.total_seconds()) > 86400 and user == user1 and user == user2:  # 이전 check-in과 이후 check-in의 시간 차이가 24시간 이상인 경우
                isolated_index.append(idx)
            elif abs(diff2.total_seconds()) > 86400 and user == user2 and user != user1:  # 이후 check-in과의 시간 차이가 24시간 이상이고 이전 check-in과 다른 User인 경우  
                isolated_index.append(idx)
            elif abs(diff1.total_seconds()) > 86400 and user == user1 and user != user2:  # 이전 check-in과의 시간 차이가 24시간 이상이고 이후 check-in과 다른 User인 경우
                isolated_index.append(idx)
        df = df[~df.index.isin(set(isolated_index))]

        logging.info('[Preprocess - Train/Validate/Test Split] Done.')
        return df

    @classmethod
    def generate_traj_id(cls, df, session_time_interval=24, rv_thresh=2):
        df = df.sort_values(['SplitTag','UserId','LocalTime']).copy()
        tw = pd.Timedelta(hours=session_time_interval)

        def _make(part):
            if part.empty: return part
            traj_idxs = []
            for user_id, user_df in part.groupby('UserId', sort=False):
                user_df = user_df.sort_values('LocalTime')
                start = user_df['LocalTime'].iloc[0]
                end = start + tw
                idx = 1
                for _, row in user_df.iterrows():
                    if row['LocalTime'] <= end:
                        traj_idxs.append(f"{user_id}_{idx}")
                    else:
                        idx += 1
                        start = row['LocalTime']; end = start + tw
                        traj_idxs.append(f"{user_id}_{idx}")
            part = part.copy()
            part['TrajectoryId'] = traj_idxs

            # 짧은 세션 제거
            keep = part['TrajectoryId'].map(part['TrajectoryId'].value_counts()) >= rv_thresh
            dropped = (~keep).sum()
            part = part[keep]
            logging.info(f'[Preprocess] dropped trajectories len<{rv_thresh}: {dropped}')
            return part

        out = []
        for tag in ['train','validation','test']:
            out.append(_make(df[df['SplitTag']==tag]))
            
        return pd.concat(out).sort_values(['UserId','LocalTime'])
