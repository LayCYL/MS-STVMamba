import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag="train", size=None, scale=True):        
        assert size is not None, "size = [seq_len, label_len, pred_len] 必须指定"
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.scale = scale
        self.root_path = root_path

        self.__read_data__()
    

    def _load_flow(self):        
        data_file = os.path.join(self.root_path)
        npz = np.load(data_file, allow_pickle=True)
        flow = npz["data_array"][:, :, 0].astype(np.float32)
        return flow  # (T, N)

    def _load_poi(self, T, N):
        
        base_dir = os.path.dirname(os.path.abspath(self.root_path))
        poi_file = os.path.join(base_dir, "poi.csv")
        poi_features = np.zeros((T, N, 9), dtype=np.float32)
        if not os.path.exists(poi_file):
            print(f"[Dataset_PEMS] 未找到 poi.csv，POI 特征通道将为 0。期望路径: {poi_file}")
            return poi_features
        poi_df = pd.read_csv(poi_file)
        poi_cols = [
            "hospital",
            "education",
            "retail",
            "residence",
            "recreation",
            "industrial",
            "office_facilities",
            "public_institution",
            "transportation",
        ]
        for col in poi_cols:
            if col not in poi_df.columns:
                raise ValueError(f"[Dataset_PEMS] poi.csv 缺少列 '{col}'，请检查列名。")        
        poi_df_sorted = poi_df.sort_values("device")
        poi_mat = poi_df_sorted[poi_cols].values.astype(np.float32)  # (num_devices, 9)
        num_devices = poi_mat.shape[0]
        if num_devices < N:
            pad = np.zeros((N - num_devices, poi_mat.shape[1]), dtype=np.float32)
            poi_mat = np.concatenate([poi_mat, pad], axis=0)
        elif num_devices > N:
            poi_mat = poi_mat[:N, :]        
        poi_min = poi_mat.min(axis=0, keepdims=True)
        poi_max = poi_mat.max(axis=0, keepdims=True)
        denom = poi_max - poi_min
        denom[denom == 0] = 1.0
        poi_norm = (poi_mat - poi_min) / denom  # (N, 9)       
        poi_features = np.tile(poi_norm[None, :, :], (T, 1, 1)).astype(np.float32)
        return poi_features

    def _load_weather_good(self, T, N):
        
        base_dir = os.path.dirname(os.path.abspath(self.root_path))
        weather_file = os.path.join(base_dir, "weather.csv")
        weather_feat = np.zeros((T, N, 1), dtype=np.float32)
        if not os.path.exists(weather_file):
            print(f"[Dataset_PEMS] 未找到 weather.csv，天气特征通道将为 0。期望路径: {weather_file}")
            return weather_feat
        wdf = pd.read_csv(weather_file)
        if "weather_condition" not in wdf.columns:
            print("[Dataset_PEMS] weather.csv 中未找到 'weather_condition' 列，天气特征默认为 0。")
            return weather_feat
        wcond = wdf["weather_condition"].values.astype(np.float32)       
        L = len(wcond)
        if L < T:
            pad_len = T - L
            if pad_len > 0:
                wcond = np.concatenate([wcond, np.repeat(wcond[-1], pad_len)])
        elif L > T:
            wcond = wcond[:T]       
        max_code = 10.0
        good_weather = (max_code + 1.0 - wcond) / max_code  # 1(晴)→接近1, 10(沙尘暴)→接近0
        gw_min, gw_max = good_weather.min(), good_weather.max()
        gw_denom = gw_max - gw_min if gw_max != gw_min else 1.0
        good_weather_norm = (good_weather - gw_min) / gw_denom  # (T,)
        good_weather_norm = good_weather_norm.reshape(T, 1)    # (T, 1)
        
        weather_feat = np.tile(good_weather_norm[:, None, :], (1, N, 1)).astype(np.float32)
        return weather_feat

    
    def __read_data__(self):
        self.scaler = StandardScaler() 
        flow = self._load_flow()  # (T, N)
        T, N = flow.shape  
        poi_features = self._load_poi(T, N)              # (T, N, 9)
        weather_features = self._load_weather_good(T, N) # (T, N, 1)
        train_ratio = 0.7
        valid_ratio = 0.1
        train_end = int(train_ratio * T)
        val_end = int((train_ratio + valid_ratio) * T)

        train_flow = flow[:train_end]
        valid_flow = flow[train_end:val_end]
        test_flow = flow[val_end:]
        if self.scale:
            self.scaler.fit(train_flow.reshape(-1, 1))
            flow_scaled = self.scaler.transform(flow.reshape(-1, 1)).reshape(flow.shape)
        else:
            flow_scaled = flow
        flow_ch = flow_scaled[:, :, None]          
        data_x_full = np.concatenate(
            [flow_ch, poi_features, weather_features], axis=2
        ).astype(np.float32)                       
        train_x = data_x_full[:train_end]
        valid_x = data_x_full[train_end:val_end]
        test_x = data_x_full[val_end:]
        if self.set_type == 0:   # train
            self.data_x = train_x
            self.data_y = train_flow
        elif self.set_type == 1: # val
            self.data_x = valid_x
            self.data_y = valid_flow
        else:                    # test
            self.data_x = test_x
            self.data_y = test_flow 
        T_sub, N_sub, C_sub = self.data_x.shape
        df_x = pd.DataFrame(self.data_x.reshape(T_sub, -1))
        df_x = df_x.fillna(method="ffill", limit=T_sub)
        self.data_x = df_x.values.reshape(T_sub, N_sub, C_sub).astype(np.float32)
        df_y = pd.DataFrame(self.data_y)
        df_y = df_y.fillna(method="ffill", limit=len(df_y))
        self.data_y = df_y.values.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)
