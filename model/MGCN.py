import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import dgl
from dgl.nn import ChebConv

from layers.mamba_ssm.mixer2_seq_simple import MixerTSModel as Mamba



class CustomChebConv(ChebConv):
    def __init__(self, in_channels, out_channels, K):
        super().__init__(in_channels, out_channels, K)
        self.fusion_linear = nn.Linear(out_channels, in_channels)

    def forward(self, graph, feat):
        feat = super().forward(graph, feat) 
        feat = torch.relu(feat)
        feat = self.fusion_linear(feat)      
        return feat


class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(STGCNLayer, self).__init__()
        self.custom_graph_conv = CustomChebConv(in_channels, out_channels, K)

    def forward(self, g, x):
       
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            node_features = x[i]  
            out_graph = self.custom_graph_conv(g, node_features)
            outputs.append(out_graph.unsqueeze(0))
        return torch.cat(outputs, dim=0)  



class MGCN_block(nn.Module):
    def __init__(
        self,
        DEVICE,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        len_input,
        adj_mx,
        distance_df_filename,
        id_filename=None
    ):
        super(MGCN_block, self).__init__()

        self.device = DEVICE
        self.adj_mx = adj_mx

        
        self.aug_adj_mx = self._build_augmented_adj(adj_mx, distance_df_filename)

        self.enc_embedding = DataEmbedding_inverted(len_input, 512, 0.1)
        self.stgcn_layer = STGCNLayer(len_input, nb_chev_filter, K)

        self.residual_conv = nn.Conv2d(
            in_channels,
            nb_time_filter,
            kernel_size=(1, 1),
            stride=(1, time_strides)
        )
        self.ln = nn.LayerNorm(nb_time_filter)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=512,
                        n_layer=1,
                        d_intermediate=2048,
                        dropout=0.1,
                        use_casual_conv=True,
                        VPT_mode=1,
                        n_vars=290,
                        ATSP_solver="SA",
                        ssm_cfg={"layer": "Mamba1"},
                        device=DEVICE,
                        distance_df_filename=distance_df_filename,
                        id_filename=id_filename,
                    ),
                    d_model=512,
                    dropout=0.1,
                    activation="gelu",
                )
                for _ in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(512),
        )

        self.projector = nn.Linear(512, len_input, bias=True)
        self.projector1 = nn.Linear(nb_time_filter, in_channels, bias=True)
        self.projector2 = nn.Linear(in_channels, nb_time_filter, bias=True)

    
    def _build_augmented_adj(self, adj_mx, distance_df_filename, poi_threshold=0.5, poi_beta=0.5):
        
        try:
            
            base_dir = os.path.dirname(os.path.abspath(distance_df_filename))
            poi_path = os.path.join(base_dir, "poi.csv")
            if not os.path.exists(poi_path):
                print(f"[MGCN_block] poi.csv not found at {poi_path}, use original adjacency.")
                return adj_mx

            poi_df = pd.read_csv(poi_path)

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
                    print(f"[MGCN_block] poi.csv missing column '{col}', use original adjacency.")
                    return adj_mx

           
            poi_df_sorted = poi_df.sort_values("device")
            poi_mat = poi_df_sorted[poi_cols].values.astype(np.float32)  

            
            if isinstance(adj_mx, np.ndarray):
                N = adj_mx.shape[0]
            else:
                N = adj_mx.size(0)

            if poi_mat.shape[0] < N:
                pad = np.zeros((N - poi_mat.shape[0], poi_mat.shape[1]), dtype=np.float32)
                poi_mat = np.concatenate([poi_mat, pad], axis=0)
            elif poi_mat.shape[0] > N:
                poi_mat = poi_mat[:N, :]

            
            poi_min = poi_mat.min(axis=0, keepdims=True)
            poi_max = poi_mat.max(axis=0, keepdims=True)
            denom = poi_max - poi_min
            denom[denom == 0] = 1.0
            poi_norm = (poi_mat - poi_min) / denom  

            
            norms = np.linalg.norm(poi_norm, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            feat_norm = poi_norm / norms
            S = np.dot(feat_norm, feat_norm.T)  
            S = np.clip(S, 0.0, 1.0)           

            
            np.fill_diagonal(S, 0.0)

            
            S[S < poi_threshold] = 0.0

            if S.max() > 0:
                S = S / S.max()  

            
            if isinstance(adj_mx, np.ndarray):
                A = adj_mx.astype(np.float32)
            else:
                A = adj_mx.detach().cpu().numpy().astype(np.float32)

            if A.max() > 0:
                A = A / A.max()

            
            A_aug = A + poi_beta * S
           
            A_aug = 0.5 * (A_aug + A_aug.T)

            print("[MGCN_block] Built POI-augmented adjacency matrix.")
            return A_aug

        except Exception as e:
            print(f"[MGCN_block] POI-based adjacency augmentation failed: {e}")
            return adj_mx

    
    def forward(self, x):
        
        x0 = x
        if x0.dim() < 4:
            
            x0 = x0.permute(0, 2, 1)      
            x0 = torch.unsqueeze(x0, dim=2)  

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x0.shape

        
        x1 = torch.squeeze(x0, dim=2).to(self.device)  

        
        adj_mx_tensor = (
            torch.from_numpy(self.aug_adj_mx)
            if isinstance(self.aug_adj_mx, np.ndarray)
            else self.aug_adj_mx
        )
        edge_index = adj_mx_tensor.nonzero(as_tuple=False).T  
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_of_vertices)
        g = g.to(self.device)
        output_gcn = self.stgcn_layer(g, x1)             
        enc_out = self.enc_embedding(output_gcn)         
        mamba_output, _ = self.encoder(enc_out)          
        mamba_output = self.projector(mamba_output)      
        mamba_output = mamba_output.permute(0, 2, 1)[:, :, :num_of_vertices]
        mamba_output = mamba_output.permute(0, 2, 1)     
        mamba_output = torch.unsqueeze(mamba_output, dim=3)  
        mamba_output = self.projector2(mamba_output)           
        x_residual = self.residual_conv(x0.permute(0, 2, 1, 3))  
        x_residual = x_residual.permute(0, 2, 3, 1)             
        x_residual1 = self.ln(F.relu(x_residual + mamba_output))
        x_residual1 = self.ln(F.relu(x_residual1))
        x_residual2 = self.projector1(x_residual1)               
        x_residual2 = x_residual2.permute(0, 1, 3, 2)            
        return x_residual2



class MGCN_submodule(nn.Module):
    def __init__(
        self,
        DEVICE,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        num_for_predict,
        len_input,
        adj_mx,
        distance_df_filename,
        id_filename=None,
    ):
        super(MGCN_submodule, self).__init__()

        self.Block = MGCN_block(
            DEVICE,
            in_channels,
            K,
            nb_chev_filter,
            nb_time_filter,
            time_strides,
            len_input,
            adj_mx,
            distance_df_filename,
            id_filename=id_filename,
        )
        self.DEVICE = DEVICE
        self.projector3 = nn.Linear(len_input, num_for_predict, bias=True)
        self.to(DEVICE)

    def forward(self, x):
        
        x = self.Block(x)                 
        output = torch.squeeze(x, dim=2)  
        output = self.projector3(output)  
        output_final = output.permute(0, 2, 1) 
        return output_final



class MultiInputMGCN(nn.Module):
    def __init__(
        self,
        DEVICE,
        in_channels,        
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        adj_mx,
        num_for_predict,
        len_input,
        distance_df_filename,
        id_filename=None,
    ):
        super(MultiInputMGCN, self).__init__()

        self.DEVICE = DEVICE
        self.total_in_channels = in_channels
        self.num_for_predict = num_for_predict
        self.backbone = MGCN_submodule(
            DEVICE,
            in_channels=1,
            K=K,
            nb_chev_filter=nb_chev_filter,
            nb_time_filter=nb_time_filter,
            time_strides=time_strides,
            num_for_predict=num_for_predict,
            len_input=len_input,
            adj_mx=adj_mx,
            distance_df_filename=distance_df_filename,
            id_filename=id_filename,
        )
        exog_dim = max(in_channels - 1, 0)
        self.exog_dim = exog_dim
        self.poi_dim = min(9, exog_dim)
        self.weather_dim = max(exog_dim - self.poi_dim, 0)
        if self.poi_dim > 0:
            self.poi_embed_dim = 16
            self.poi_encoder = nn.Linear(self.poi_dim, self.poi_embed_dim)
        else:
            self.poi_embed_dim = 0
            self.poi_encoder = None
        if self.weather_dim > 0:
            self.weather_embed_dim = 8
            self.weather_encoder = nn.Linear(self.weather_dim, self.weather_embed_dim)
        else:
            self.weather_embed_dim = 0
            self.weather_encoder = None

        total_exog_embed_dim = self.poi_embed_dim + self.weather_embed_dim
        if total_exog_embed_dim > 0:
            self.exog_mlp = nn.Sequential(
                nn.Linear(total_exog_embed_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_for_predict), 
            )
            
            self.exog_gate = nn.Parameter(torch.tensor(0.05))
        else:
            self.exog_mlp = None
            self.exog_gate = None

        self.to(DEVICE)

    def forward(self, x):
       
        if x.dim() == 3:
            traffic = x                     
            poi = None
            weather = None
        elif x.dim() == 4:
            B, T_in, N, C = x.shape
            if C != self.total_in_channels:
                raise ValueError(
                    f"Expected input with {self.total_in_channels} channels, but got {C}"
                )
            traffic = x[..., 0]             
            exog = x[..., 1:]               

            poi = None
            weather = None
            if self.exog_dim > 0:
                if self.poi_dim > 0:
                    poi = exog[..., : self.poi_dim]  
                if self.weather_dim > 0:
                    weather = exog[..., self.poi_dim :]  
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        
        y_backbone = self.backbone(traffic)  
        B, T_out, N = y_backbone.shape
        if (self.exog_mlp is None) or (x.dim() == 3):
            return y_backbone
        exog_emb_list = []
        if (self.poi_encoder is not None) and (poi is not None):
            poi = torch.nan_to_num(poi)
            poi_avg = poi.mean(dim=1) 
            poi_emb = self.poi_encoder(poi_avg)  
        if (self.weather_encoder is not None) and (weather is not None):
            weather = torch.nan_to_num(weather)
            weather_node_mean = weather.mean(dim=2)   
            last_weather = weather_node_mean[:, -1, :]  
            weather_emb = self.weather_encoder(last_weather)  
            weather_emb_node = weather_emb.unsqueeze(1).expand(B, N, -1)  
            exog_emb_list.append(weather_emb_node)

        if len(exog_emb_list) == 0:
            return y_backbone

        exog_feat = torch.cat(exog_emb_list, dim=-1)  
        exog_pred = self.exog_mlp(exog_feat)          
        exog_pred = exog_pred.permute(0, 2, 1)       
        y = y_backbone + self.exog_gate * exog_pred
        return y

def make_model(
    DEVICE,
    in_channels,
    K,
    nb_chev_filter,
    nb_time_filter,
    time_strides,
    adj_mx,
    num_for_predict,
    len_input,
    distance_df_filename,
    id_filename=None,
):
    
    model = MultiInputMGCN(
        DEVICE=DEVICE,
        in_channels=in_channels,
        K=K,
        nb_chev_filter=nb_chev_filter,
        nb_time_filter=nb_time_filter,
        time_strides=time_strides,
        adj_mx=adj_mx,
        num_for_predict=num_for_predict,
        len_input=len_input,
        distance_df_filename=distance_df_filename,
        id_filename=id_filename,
    )

   
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        
        if x.dim() == 3:
            B, N, L = x.shape
            x_reshape = x.reshape(B * N, L)
            x_emb = self.value_embedding(x_reshape)  
            x_emb = x_emb.reshape(B, N, -1)         
        else:
            x_emb = self.value_embedding(x)

        pos_encoded = self._get_positional_encoding(x_emb)
        x_out = self.norm(pos_encoded)
        return self.dropout(x_out)

    def _get_positional_encoding(self, x):
       
        batch_size, seq_len, d_model = x.size()
        position = torch.arange(
            0, seq_len, dtype=torch.float, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float()
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        return x + pe


class EncoderLayer(nn.Module):
    def __init__(self, mamba, d_model, dropout=0.1, activation="gelu"):
        super(EncoderLayer, self).__init__()
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True
        )
        self.mamba = mamba
        self.fc_mamba = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation_dict = {
            "relu": F.relu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        if activation not in self.activation_dict:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose from {list(self.activation_dict.keys())}"
            )
        self.activation = self.activation_dict[activation]

    def forward(self, x):
       
        mamba_out, _ = self.mamba(x)
        mamba_out = self.fc_mamba(mamba_out)
        mamba_out = self.dropout(self.activation(mamba_out))
        mamba_lstm_out, _ = self.lstm(mamba_out)

        new_x = mamba_lstm_out + mamba_lstm_out
        attn = 1

        x = x + new_x
        x = self.norm1(x)
        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
