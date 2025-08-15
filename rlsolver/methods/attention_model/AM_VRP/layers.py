import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot product attention."""
    
    def __init__(self, n_heads, d_model):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_depth = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_depth).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_depth)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        attention = torch.matmul(attention_weights, V)
        
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_out(attention)
        
        return output