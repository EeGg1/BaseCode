import os
from pathlib import Path
from typing import Union, Tuple

import torch
import datetime
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from yacs.config import CfgNode as CN

from utils.timefeatures import time_features
from utils.misc import return_type

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from tqdm import tqdm


class ForecastingDataset(Dataset):
    def __init__(
        self, 
        data_dir : Union[str, Path],
        n_var : int,
        seq_len : int,
        label_len : int,
        pred_len : int,
        features : str,
        timeenc : int,
        freq : str,
        date_idx : int,
        target_start_idx: int,
        scale = "standard",
        train_ratio = 0.7,
        test_ratio = 0.2,
        data_type: str = np.float32
        ):
        
        self.data_dir = data_dir
        
        self.data_type = data_type
        
        self.n_var = n_var
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        self.date_idx = date_idx
        self.target_start_idx = target_start_idx
        
        self.scale = scale
        self.split = None
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        
        self.all_train, self.all_val, self.all_test, self.all_train_stamp, self.all_val_stamp, self.all_test_stamp, \
            self.all_train_window, self.all_val_window, self.all_test_window = self._load_data()
        assert self.all_train.shape[1] == n_var
        
        print("Training size: {0}, Validation size: {1}, Test size: {2}".format(len(self.all_train),len(self.all_val),len(self.all_test)))
        
        #TODO 원래 파일이랑 nonstationary_trasnformer 확인해보기 
        # self._normalize_data()
        # # self.print_data_stats()

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        #! 체크
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        # files = files[0:10]
        # files = ['BTCUSDT.csv', 'ETHUSDT.csv', 'DOGEUSDT.csv']
        # files = [f for f in os.listdir(self.data_dir) if (f.endswith('USDT.csv') and os.path.getsize(os.path.join(self.data_dir, f)) > 15*1024*1024)]
        
        total_files = len(files)
        results = [None] * total_files
        batch_size = len(files) # or you can set a designated batch size.
        with ProcessPoolExecutor(max_workers=4) as executor:
            with tqdm(total=total_files, desc='Loading Data') as pbar:
                for i in range(0, total_files, batch_size):
                    batch = files[i:i+batch_size]
                    futures = {executor.submit(_load_single_data, file, self): idx for idx, file in enumerate(batch, start=i)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
                        pbar.update(1)
        
        # with Pool(processes=4) as pool:
        #     results = list(tqdm(pool.imap(self._load_single_data, files), total=len(files), desc = 'Loading Data')) # len(results) = files 개수
        
        all_train, all_val, all_test, all_train_stamp, all_val_stamp, all_test_stamp, all_train_window, all_val_window, all_test_window = map(list, zip(*results))
        # all_train: list. len(all_train) = len(files). all_train[0]: ndarray. shape = (train data 길이, n_var)
        
        total_len = [0,0,0]
        for i in range(len(files)-1):
            total_len[0] += len(all_train[i])
            total_len[1] += len(all_val[i])
            total_len[2] += len(all_test[i])
            
            all_train_window[i+1] += total_len[0]
            all_val_window[i+1]   += total_len[1]
            all_test_window[i+1]  += total_len[2]
            
        # Convert lists to ndarrays and concatenate along the first axis
        all_train        = np.concatenate(all_train, axis=0)
        all_val          = np.concatenate(all_val, axis=0)
        all_test         = np.concatenate(all_test, axis=0)
        all_train_stamp  = np.concatenate(all_train_stamp, axis=0)
        all_val_stamp    = np.concatenate(all_val_stamp, axis=0)
        all_test_stamp   = np.concatenate(all_test_stamp, axis=0)
        all_train_window = np.concatenate(all_train_window, axis=0)
        all_val_window   = np.concatenate(all_val_window, axis=0)
        all_test_window  = np.concatenate(all_test_window, axis=0)
        
        print("Data loading is completed.")
            
        return all_train, all_val, all_test, all_train_stamp, all_val_stamp, all_test_stamp, all_train_window, all_val_window, all_test_window
    
    #TODO 필요하다면 implementation 하기!
    # def print_data_stats(self): 
    #     if self.split == 'train':
    #         print(f"Train data shape: {self.train.shape}, mean: {np.mean(self.train, axis=0)}, std: {np.std(self.train, axis=0)}")
    #     elif self.split == 'val':
    #         print(f"Validation data shape: {self.val.shape}, mean: {np.mean(self.val, axis=0)}, std: {np.std(self.val, axis=0)}")
    #     elif self.split == 'test':
    #         print(f"Test data shape: {self.test.shape}, mean: {np.mean(self.test, axis=0)}, std: {np.std(self.test, axis=0)}")

    def __len__(self): #! 이 함수 맞나..?
        if self.split == "train":
            return len(self.all_train_window)
        elif self.split == "val":
            return len(self.all_val_window)
        elif self.split == "test":
            return len(self.all_test_window)

    # 순서대로 되어 있는 데이터 셋에서 index 부터 시작되는 window nparray 반환하는 함수이다.
    def __getitem__(self, index):
        if self.split == "train":
            enc_start_idx = self.all_train_window[index][0]
            enc_end_idx = self.all_train_window[index][1]
            dec_start_idx = self.all_train_window[index][2]
            dec_end_idx = self.all_train_window[index][3]
            
            enc_window = self.all_train[enc_start_idx:enc_end_idx]
            enc_window_stamp = self.all_train_stamp[enc_start_idx:enc_end_idx]
            dec_window = self.all_train[dec_start_idx:dec_end_idx]
            dec_window_stamp = self.all_train_stamp[dec_start_idx:dec_end_idx] 
            
        elif self.split == 'val':
            enc_start_idx = self.all_val_window[index][0]
            enc_end_idx = self.all_val_window[index][1]
            dec_start_idx = self.all_val_window[index][2]
            dec_end_idx = self.all_val_window[index][3]
            
            enc_window = self.all_val[enc_start_idx:enc_end_idx]
            enc_window_stamp = self.all_val_stamp[enc_start_idx:enc_end_idx]
            dec_window = self.all_val[dec_start_idx:dec_end_idx]
            dec_window_stamp = self.all_val_stamp[dec_start_idx:dec_end_idx]
            
        elif self.split == 'test':
            enc_start_idx = self.all_test_window[index][0]
            enc_end_idx = self.all_test_window[index][1]
            dec_start_idx = self.all_test_window[index][2]
            dec_end_idx = self.all_test_window[index][3]
            
            enc_window = self.all_test[enc_start_idx:enc_end_idx]
            enc_window_stamp = self.all_test_stamp[enc_start_idx:enc_end_idx]
            dec_window = self.all_test[dec_start_idx:dec_end_idx]
            dec_window_stamp = self.all_test_stamp[dec_start_idx:dec_end_idx]
            
        else:
            raise ValueError("Invalid split: " + self.split)
        
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp
   
    
def _load_single_data(file, context):
    df_raw = pd.read_csv(os.path.join(context.data_dir, file)) # float64 & int64 로 구성
    
    # crypto일 경우 시간 역순하고 symbol 제거해줘야됨
    # r_idx = [i for i in range(df_raw.shape[0]-1,-1,-1)]
    # df_raw = pd.DataFrame(df_raw, index=r_idx)
    # del df_raw['Symbol']
    # df_raw.rename(columns={'Date':'date'},inplace=True)
    
    assert df_raw.columns[context.date_idx] == 'date'

    data = _split_data(df_raw, context) # type(data) = tuple | train, val, test, train_stamp, val_stamp, test_stamp (ndarry x 6)
    # float64 & int32 로 구성됨
    
    # data = self._normalize_data(list(data))
    train, val, test, train_stamp, val_stamp, test_stamp  = _normalize_data(list(data), context) # float64 & int32 로 구성됨
    
    # ndarray로 나옴 (train data길이 (csv마다 다름), 4)
    train_window, val_window, test_window = _create_windows([len(train), len(val), len(test)], context)
    
    return train, val, test, train_stamp, val_stamp, test_stamp, train_window, val_window, test_window

def _split_data(df_raw: pd.DataFrame, context) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]: #TODO context dtype
    assert 0.0 < context.train_ratio < 1.0 and 0.0 < context.test_ratio < 1.0 and context.train_ratio + context.test_ratio <= 1.0
    
    data      = df_raw[df_raw.columns[1:]].values
    data = data.astype(context.data_type)
    train_len = int(len(data) * context.train_ratio)
    test_len  = int(len(data) * context.test_ratio)
    val_len   = len(data) - train_len - test_len
    
    train_start = 0
    train_end   = train_len

    val_start = train_len - context.seq_len 
    val_end   = train_len + val_len
    
    test_start = train_len + val_len - context.seq_len 
    test_end   = len(data)
    
    train = data[train_start:train_end]
    val   = data[val_start:val_end]
    test  = data[test_start:test_end]
    
    df_stamp = df_raw[['date']].copy()
    df_stamp['date'] = pd.to_datetime(df_stamp.date,format='mixed')
    
    if context.timeenc == 0:
        df_stamp.loc[:, 'month'] = df_stamp.date.dt.month
        df_stamp.loc[:, 'day'] = df_stamp.date.dt.day
        df_stamp.loc[:, 'weekday'] = df_stamp.date.dt.weekday
        df_stamp.loc[:, 'hour'] = df_stamp.date.dt.hour
        data_stamp = df_stamp.drop(['date'], axis=1).values
        data_stamp = data_stamp.astype(context.data_type)
    elif context.timeenc == 1: #! 이거 뭔지 모름
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=context.freq)
        data_stamp = data_stamp.transpose(1, 0)
    
    train_stamp = data_stamp[train_start:train_end]
    val_stamp = data_stamp[val_start:val_end]
    test_stamp = data_stamp[test_start:test_end]
    
    return train, val, test, train_stamp, val_stamp, test_stamp

def _normalize_data(data: list, context) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]: 
    if context.scale == "standard":
        scaler = StandardScaler()
    elif context.scale == "min-max":
        scaler = MinMaxScaler()
    elif context.scale == "none":
        return data
    else:
        raise ValueError("Invalid scale: " + context.scale)

    data[0] = scaler.fit_transform(data[0]) # train data
    data[1] = scaler.transform(data[1]) # val data
    data[2] = scaler.transform(data[2]) # test data
    return tuple(data)

def _create_windows(current: list, context) -> tuple[ndarray, ndarray, ndarray]: 
    window = {} 
    for i in range(3):
        temp1 = np.empty((0,4), dtype=int)
        for j in range(current[i] - context.seq_len - context.pred_len + 1): # 원래 __len__에 이렇게 사용   
            # 원래 __getitem__ 참고해서 만듦
            enc_start_idx = j 
            enc_end_idx = enc_start_idx + context.seq_len
            dec_start_idx = enc_end_idx - context.label_len # all[i] + j  + self.seq_len - self.label_len
            dec_end_idx = dec_start_idx + context.label_len + context.pred_len # all[i] + j + self.seq_len + self.pred_len 
            
            temp2 = np.array([enc_start_idx, enc_end_idx, dec_start_idx, dec_end_idx])
            temp2 = temp2[np.newaxis, :]
            temp1 = np.concatenate((temp1, temp2), axis=0)
        window[i] = temp1
        
    return window[0], window[1], window[2]


def build_dataset(cfg):
    data_name = cfg.DATA.NAME
    dataset_config = dict(
        data_dir=os.path.join(cfg.DATA.BASE_DIR, data_name),
        n_var=cfg.DATA.N_VAR,
        seq_len=cfg.DATA.SEQ_LEN,
        label_len=cfg.DATA.LABEL_LEN,
        pred_len=cfg.DATA.PRED_LEN,
        features=cfg.DATA.FEATURES,
        timeenc=cfg.DATA.TIMEENC,
        freq=cfg.DATA.FREQ,
        date_idx=cfg.DATA.DATE_IDX,
        target_start_idx=cfg.DATA.TARGET_START_IDX,
        scale=cfg.DATA.SCALE,
        train_ratio=cfg.DATA.TRAIN_RATIO,
        test_ratio=cfg.DATA.TEST_RATIO,
        data_type=return_type(cfg, 'numpy')
    )
    
    dataset = ForecastingDataset(**dataset_config)

    return dataset

def update_cfg_from_dataset(cfg: CN, dataset_name: str):
    cfg.DATA.NAME = dataset_name
    if dataset_name == 'weather':
        n_var = 21
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2 # train, val, test 비율, 데이터는 train, val, test 순서대로 자름
        cfg.DATA.DATE_IDX = 0 # raw data에서 date가 있지만 날려야 되니까 날리는 column index 
    elif dataset_name == 'illness':
        n_var = 7
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0
    elif dataset_name == 'electricity':
        n_var = 321
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0
    elif dataset_name == 'traffic':
        n_var = 862
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0
    elif dataset_name == 'exchange_rate':
        n_var = 8
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var    
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0   
    elif dataset_name == 'ETTh1' or dataset_name == 'ETTh2' or dataset_name == 'ETTm1' or dataset_name == 'ETTm2':
        n_var = 7
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.6, 0.2
        cfg.DATA.DATE_IDX = 0
    elif dataset_name == 'solar':
        n_var = 137
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0
    elif dataset_name == 'PEMS03':
        n_var = 358
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0
    elif dataset_name == 'energydata_complete':
        n_var = 28
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0
    elif dataset_name == 'crypto_binance_spot_1h':
        n_var = 7
        cfg.DATA.N_VAR, cfg.MODEL.enc_in, cfg.MODEL.dec_in, cfg.MODEL.c_out = n_var, n_var, n_var, n_var
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
        cfg.DATA.DATE_IDX = 0
    else:
        raise ValueError("Invalid dataset_name: " + dataset_name)
