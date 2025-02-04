import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):

    def __init__(self, config, in_channels, num_heads, dropout=0.5):
        super().__init__()
        embed_dim = in_channels
        d_model = embed_dim
        if d_model % num_heads != 0:
            self.pad = True
            self.original_d_model = d_model
            self.d_model = ((d_model // num_heads) + 1) * num_heads
        else:
            self.pad = False
            self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model,
                             self.original_d_model if self.pad else d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads,
                      self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                   self.d_model)

    def forward(self, Q, K, V, mask=None):
        if self.pad:
            padding_length = self.d_model - self.original_d_model
            Q = F.pad(Q, (0, padding_length))
            K = F.pad(K, (0, padding_length))
            V = F.pad(V, (0, padding_length))
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        if self.pad:
            output = output[:, :, :self.original_d_model]
        return output, output
