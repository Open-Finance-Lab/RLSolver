"""Neural network layers for TSP solver."""

from math import sqrt
import torch
import torch.nn as nn


class GraphEmbedding(nn.Module):
    """Linear embedding for graph nodes."""
    
    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_size, embedding_size)

    def forward(self, inputs):
        return self.embedding(inputs)


class Glimpse(nn.Module):
    """Multi-head dot-product attention for glimpsing."""
    
    def __init__(self, input_size, hidden_size, n_head):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.single_dim = hidden_size // n_head
        self.scale = 1.0 / sqrt(self.single_dim)
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, input_size)

    def forward(self, query, target, mask=None):
        """
        Args:
            query: FloatTensor [batch_size, input_size]
            target: FloatTensor [batch_size, seq_len, input_size]
            mask: BoolTensor [batch_size, seq_len]
        
        Returns:
            alpha: attention weights [batch_size, n_head, seq_len]
            output: attended features [batch_size, input_size]
        """
        batch_size, seq_len, _ = target.shape

        # Compute Q, K, V
        q = self.W_q(query).reshape(batch_size, self.n_head, self.single_dim)
        k = self.W_k(target).reshape(batch_size, seq_len, self.n_head, self.single_dim)
        v = self.W_v(target).reshape(batch_size, seq_len, self.n_head, self.single_dim)
        
        # Rearrange for batched matrix multiplication
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        
        # Compute attention scores
        scores = torch.einsum("ijl,ijkl->ijk", q, k) * self.scale
        
        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).repeat(1, self.n_head, 1)
            scores.masked_fill_(mask_expanded, float('-inf'))
        
        # Compute attention weights
        alpha = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.einsum("ijk,ijkl->ijl", alpha, v)
        
        # Output projection
        output = self.W_out(attended.reshape(batch_size, -1))
        
        return alpha, output


class Pointer(nn.Module):
    """Pointer network for selecting next node."""
    
    def __init__(self, input_size, hidden_size, n_head=1, C=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.C = C
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

    def forward(self, query, target, mask=None):
        """
        Args:
            query: FloatTensor [batch_size, input_size]
            target: FloatTensor [batch_size, seq_len, input_size]
            mask: BoolTensor [batch_size, seq_len]
            
        Returns:
            probs: selection probabilities [batch_size, seq_len]
            attended: attended features [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = target.shape
        
        q = self.W_q(query)  # [batch_size, hidden_size]
        k = self.W_k(target)  # [batch_size, seq_len, hidden_size]
        v = self.W_v(target)  # [batch_size, seq_len, hidden_size]
        
        # Compute attention scores
        scores = torch.einsum("ik,ijk->ij", q, k)  # [batch_size, seq_len]
        scores = self.C * torch.tanh(scores)
        
        # Apply mask
        if mask is not None:
            scores.masked_fill_(mask, float('-inf'))
        
        # Compute probabilities
        probs = torch.softmax(scores, dim=-1)
        
        # Compute attended features
        attended = torch.einsum("ij,ijk->ik", probs, v)
        
        return probs, attended


class AttentionLayer(nn.Module):
    """Self-attention layer with feed-forward network."""
    
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        
        # Feed-forward with residual connection
        x = x + self.ffn(x)
        
        return x


class AttentionModule(nn.Module):
    """Stack of self-attention layers."""
    
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionLayer(embed_dim, n_heads, feed_forward_hidden)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x