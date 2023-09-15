import torch
import torch.nn as nn
import numpy as np
from .feature_extractor import build_feature_extractor
from .bitrap_np import BiTraPNP
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from vrnntools.trajpred_models.modeling.mlp import MLP
from vrnntools.utils.common import dotdict
from vrnntools.trajpred_models.modeling.gat import GAT
from vrnntools.utils.adj_matrix import ego_dists
import pdb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=8)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 0.1), num_layers)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src)

        return output
    
class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, 0.1), num_layers)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        tgt = self.dropout(tgt)
        memory = memory.repeat(1, tgt.size(1), 1)
        output = self.transformer_decoder(tgt, memory)
        return output
    
class Transformer(nn.Module):
    def __init__(self, args, device):
        super(Transformer, self).__init__()
        self.device = device
        self.cvae = BiTraPNP(args)
        self.input_dim = args.input_dim # Input dim
        self.hidden_size = args.hidden_size # GRU hidden size
        self.enc_steps = args.enc_steps # observation step
        self.dec_steps = args.dec_steps # prediction step
        self.dropout = args.dropout
        self.layer_norm = 'layer_norm' in args and args.layer_norm
        self.feature_extractor = build_feature_extractor(args)
        self.pred_dim = args.pred_dim
        self.K = args.K
        self.map = False
        self.pred_dim = 2

        self.no_abs = args.no_abs if 'no_abs' in args else False
        self.use_corr = args.use_corr
        self.criterion = nn.MSELoss()

        self.sigma = None
        self.graph = None
        self.lg = None
        self.train_start_time = 0 if 'start_time' not in args else args.start_time

        feat_enc_resnet = dotdict({
                            "in_size": 2048,
                            "hidden_size": [512],
                            "out_size": 96,
                            "dropout": 0.0,
                            "layer_norm": True
                        },)
        embed_size = args.hidden_size
        self.feat_enc_resnet = MLP(feat_enc_resnet, device=self.device)
        self.combine = MLP(dotdict({'in_size': feat_enc_resnet.out_size + embed_size,
                                    'hidden_size': [embed_size],
                                    'out_size': embed_size,
                                    'dropout': 0.0}), device=self.device)
        def combine_input(input_x, input_resnet):
            # 1 layer: 4 -> 96 -> ReLu
            feat_x = self.feature_extractor(input_x)
            # 2 layers: 2048 -> 512 -> 96
            feat_resnet = self.feat_enc_resnet(input_resnet)
            # 192 wide
            combined = torch.cat([feat_x, feat_resnet], dim=-1)
            return combined
        self.combine_input = combine_input

        # the predict shift is in meter
        self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                    self.pred_dim))   
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.LayerNorm(1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.LayerNorm(1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.LayerNorm(self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                nn.LayerNorm(self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + args.LATENT_DIM,
                                                self.hidden_size),
                                                nn.LayerNorm(self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size),
                                                nn.LayerNorm(self.hidden_size),
                                                nn.ReLU(inplace=True))

        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                nn.LayerNorm(self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                nn.LayerNorm(self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                nn.LayerNorm(self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                nn.LayerNorm(self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.traj_enc_cell = nn.GRUCell(self.hidden_size*2 + self.hidden_size//4, self.hidden_size)
        self.drop_layer = nn.Dropout(self.dropout)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.encoder = TransformerEncoder(input_dim=self.hidden_size*2, 
                                          d_model=self.hidden_size, 
                                          nhead=8, 
                                          num_layers=3, 
                                          dim_feedforward=256)
        self.decoder = TransformerDecoder(input_dim=self.hidden_size, 
                                          d_model=self.hidden_size, 
                                          nhead=8, 
                                          num_layers=3, 
                                          dim_feedforward=256)
        
    def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)
       
        K = dec_hidden.shape[1]
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        for dec_step in range(self.dec_steps):
            # incremental goal for each time step
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.drop_layer(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            batch_traj = self.regressor(dec_hidden)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:,dec_step,:,:] = batch_traj
        return dec_traj

    def forward(self, inputs, map_mask=None, targets = None, start_index = 0, training=True,
                input_resnet=None, seq_start_end=None, hist_adj=None):
        # input given in T x B x d -> need to transpose
        inputs = inputs.permute(1, 0, 2)
        if input_resnet is not None:
            input_resnet = input_resnet.permute(1, 0, 2)
        self.training = training
        if torch.is_tensor(start_index):
            start_index = start_index[0].item()
        if self.feat_enc_resnet is not None:
            traj_input_temp = self.combine_input(inputs[:,start_index:,:], input_resnet[:,start_index:,:])
        else:
            # One layer: 4 -> 96 -> ReLu
            traj_input_temp = self.feature_extractor(inputs[:,start_index:,:])
        traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
        traj_input[:,start_index:,:] = traj_input_temp
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        total_KLD = 0
        for enc_step in range(start_index, self.enc_steps):
            traj_enc_hidden = self.traj_enc_cell(self.drop_layer(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            # Refine here
            if hist_adj is not None and self.graph is not None:
                hist_adj_t = hist_adj[enc_step]
                h_g = self.graph(traj_enc_hidden, hist_adj_t)
                traj_enc_hidden = self.lg(torch.cat([traj_enc_hidden, h_g], -1))
            enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
            # initial trajectory tensor
            goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
            goal_list = []
            for dec_step in range(self.dec_steps):
                goal_hidden = self.goal_cell(self.drop_layer(goal_input), goal_hidden)
                # next step input is generate by hidden
                goal_input = self.goal_hidden_to_input(goal_hidden)
                goal_list.append(goal_hidden)
                # regress goal traj for loss
                goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
                goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
            # get goal for decoder and encoder
            goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
            goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
            enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
            enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
            goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
            all_goal_traj[:,enc_step,:,:] = goal_traj
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            if self.training:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, inputs[:,enc_step,:], self.K, targets[:,enc_step,:,:])
            else:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, inputs[:,enc_step,:], self.K)
            total_KLD += KLD
            cvae_dec_hidden= self.cvae_to_dec_hidden(cvae_hidden)

            all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
        return all_goal_traj, all_cvae_dec_traj, KLD 
