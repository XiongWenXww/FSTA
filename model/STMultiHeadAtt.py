import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.MultiHeadAtt import MultiHeadAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach() # [1, n_position, d_hid]

class STMultiHeadAtt(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        # d_model:embedding size
        super(STMultiHeadAtt, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.f_gat = nn.Linear(d_model, d_model)
        self.f_att = nn.Linear(d_model, d_model)

        nodes = 5
        self.f_gat_adj = nn.Linear(nodes, nodes)
        self.f_att_adj = nn.Linear(nodes, nodes)
    def forward(self, enc_input, slf_attn_mask=None):
        # enc_input: [B, len1, len2, d_model]
        # temporal:[B, N, T, d_model], spatial:[B, T, N, d_model], len1、len2 correspond to T or N
        out, attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask) # [B, l1, l2, d_model], [B, l1, h, l2, l2]
        attn = torch.mean(torch.mean(torch.mean(attn, dim=0), dim=0), dim=0)  # [B, l1, h, l2, l2]->[l2, l2]
        return out, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x