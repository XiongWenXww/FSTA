import torch.nn as nn
import torch
import copy
from model.STMultiHeadAtt import PositionalEncoding, STMultiHeadAtt, PositionwiseFeedForward

from model.FourierAtt import FourierAtt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FSTA(nn.Module):
    def __init__(self, opt, time_num, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        # d_model:embedding size
        super(FSTA, self).__init__()
        self.FA = FourierAtt(opt)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=1)  # (in_channels, out_channels, kernel_size)
        self.position_enc = PositionalEncoding(d_hid=d_model, n_position=time_num)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.slf_attn = STMultiHeadAtt(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout) # d_model->d_inner->d_model

        self.conv2 = nn.Conv2d(d_model, 1, 1)
        self.norm_adj = nn.InstanceNorm2d(1)  # Normalize the adjacency matrix

        self.complex_weight = nn.Parameter(
            torch.randn(1, time_num // 2 + 1, opt.nodes_num, d_model, 2, dtype=torch.float32) * 0.02)
    def forward(self, input, slf_attn_mask=None):
        # [B, T, N]
        # embedding
        emb_input = input.unsqueeze(0)  # [1, B, T, N]
        emb_input = emb_input.permute(1, 0, 2, 3)  # [B, 1, T, N]
        emb_output = self.conv1(emb_input)  # [B, 1, T, N]->[B, d_model, T, N]
        emb_output = emb_output.permute(0, 2, 3, 1)  # [B, d_model, T, N]->[B, T, N, d_model]

        # position encoding
        pos_enc = self.position_enc(emb_output) # [1, T, d_model]
        pos_enc = pos_enc.unsqueeze(2).expand(emb_output.shape) # [1, T, 1, d_model]->[B, T, N, d_model]
        enc_output = self.dropout(emb_output + pos_enc)
        enc_output = self.layer_norm(enc_output)

        FA_output = self.FA(enc_output) # [B, T, N, d_model]
        X_spa = FA_output # X_spa doesn't change with irfft_output

        # temporal attention
        tep_input = FA_output.transpose(1, 2)  # [B, N, T, d_model]
        tep_output, _ = self.slf_attn(tep_input, slf_attn_mask)  # [B, N, T, d_model], [B, N, h, T, T]
        tep_output = self.pos_ffn(tep_output)
        # spatiotemporal fusion attention
        tep_output = tep_output.transpose(1, 2)  # [B, T, N, d_model]

        _, spa_attn = self.slf_attn(X_spa, slf_attn_mask)  # [B, T, N, d_model], [N, N]
        tep_output = tep_output.transpose(2, 3) # [B, T, d_model, N]
        st_output = torch.matmul(tep_output, spa_attn) # [B, T, d_model, N]

        st_output = st_output.transpose(2, 3) # [B, T, N, d_model]
        st_output += X_spa # residual
        st_output = self.pos_ffn(st_output)

        # processing the dimension
        output = st_output.permute(0, 3, 1, 2) # [B, T, N, d_model]->[B, d_model, T, N]
        output = self.conv2(output) # [B, d_model, T, N]->[B, 1, T, N]
        output = output.squeeze(1)  # [B, 1, T, N]->[B, T, N]
        return output, spa_attn
