# layers.py

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEmbedding(nn.Module):
    """Linear embedding for graph nodes."""
    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_size, embedding_size)

    def forward(self, inputs):
        return self.embedding(inputs)


class AttentionLayer(nn.Module):
    """Self-attention layer with Pre-LN and feed-forward network using Flash Attention."""
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.ln1(x)
        batch_size, seq_len, _ = normed.shape
        q = self.q_proj(normed).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        x = x + self.dropout(attn_output)
        normed = self.ln2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout(ffn_output)
        return x
