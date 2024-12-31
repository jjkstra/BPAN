import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention / self.temperature
        attention = self.softmax(attention)
        output = torch.bmm(attention, v)
        return output, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_q, d_k, d_v, dropout=0.1, flag_norm=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_q, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_q, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_q, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=math.sqrt(2.0 / (d_q + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=math.sqrt(2.0 / (d_q + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=math.sqrt(2.0 / (d_q + d_v)))

        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))
        self.layer_norm = nn.LayerNorm(d_q)

        self.fc = nn.Linear(n_head * d_v, d_q)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.flag_norm = flag_norm

    def forward(self, q, k, v):
        """
        Go through the multi-head attention module.
        """
        sz_q, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q

        q = self.w_qs(q).view(sz_q, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)

        output, _ = self.attention(q, k, v)
        output = output.view(self.n_head, sz_q, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_q, len_q, -1)
        output = self.dropout(self.fc(output))

        if self.flag_norm:
            output = self.layer_norm(output + residual)

        return output
