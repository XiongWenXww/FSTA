import numpy as np
import torch
import torch.nn as nn
from model.Modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # d_model is embed_size
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # temporal:[B, N, T, d_model], spatial:[B, T, N, d_model], len1、len2 correspond to T or N
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len1_q, len1_k, len1_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len2_q, len2_v, len2_k = q.size(2), k.size(2), v.size(2)
        residual = q

        # [B, l1, l2, d_model]->[B, l1, l2, h*dv]->[B, l1, l2, h, dv]
        q = self.w_qs(q).view(sz_b, len1_q, len2_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len1_k, len2_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len1_v, len2_v, n_head, d_v)

        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3) # [B, l1, h, l2, dv]

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask) # [B, l1, h, l2, dv], [B, l1, h, l2, l2]

        # Transpose to move the head dimension back: [B, l1, l2, h, dv]
        # Combine the last two dimensions to concatenate all the heads together: [B, l1, l2, h*dv]
        q = q.transpose(2, 3).contiguous().view(sz_b, len1_q, len2_q, -1)
        q = self.dropout(self.fc(q)) # [B, l1, l2, d_model]
        q += residual  # residual doesn't change with Q

        q = self.layer_norm(q)

        return q, attn
