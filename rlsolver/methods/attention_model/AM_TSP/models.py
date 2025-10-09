"""TSP Model Definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from layers import GraphEmbedding, AttentionModule


class ManualCrossAttention(nn.Module):
    """Manual implementation of cross-attention."""
    
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: [batch_size, 1, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            key_padding_mask: [batch_size, seq_len]
        Returns:
            output: [batch_size, 1, embed_dim]
            attn_weights: [batch_size, num_heads, 1, seq_len]
        """
        batch_size = query.size(0)
        seq_len = key.size(1)
        
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) * self.scale
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
        output = self.out_proj(context)
        
        return output, attn_weights


class AutoregressiveTSP(nn.Module):
    """Autoregressive TSP model."""
    
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.C = C
        
        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = AttentionModule(embedding_size, n_head, n_layers=3)
        
        self.h_context_embed = nn.Linear(embedding_size, embedding_size)
        self.current_embed = nn.Linear(embedding_size, embedding_size)
        self.first_embed = nn.Linear(embedding_size, embedding_size)
        
        self.cross_attention = ManualCrossAttention(embedding_size, n_head)
        self.output_projection = nn.Linear(embedding_size, embedding_size)
    
    def forward(self, observation):
        """Single-step forward pass.
        
        Args:
            observation: dict with:
                - nodes: [batch_size, seq_len, 2]
                - current_node: [batch_size] or None
                - first_node: [batch_size] or None
                - action_mask: [batch_size, seq_len]
                - encoded: [batch_size, seq_len, embedding_size] (optional)
        Returns:
            logits: [batch_size, seq_len]
        """
        nodes = observation['nodes']
        current_node = observation.get('current_node')
        first_node = observation.get('first_node')
        action_mask = observation['action_mask']
        encoded = observation.get('encoded')
        
        batch_size = nodes.size(0)
        
        if encoded is not None:
            pass
        else:
            embedded = self.embedding(nodes)
            encoded = self.encoder(embedded)
        
        h_mean = encoded.mean(dim=1)
        h_context = self.h_context_embed(h_mean)
        
        query = h_context.clone()
        
        if current_node is not None:
            current_idx = current_node.unsqueeze(1).unsqueeze(2).expand(
                -1, -1, self.embedding_size
            )
            current_h = encoded.gather(1, current_idx).squeeze(1)
            query = query + self.current_embed(current_h)
        
        if first_node is not None:
            first_idx = first_node.unsqueeze(1).unsqueeze(2).expand(
                -1, -1, self.embedding_size
            )
            first_h = encoded.gather(1, first_idx).squeeze(1)
            query = query + self.first_embed(first_h)
        
        query = query.unsqueeze(1)
        context_vector, _ = self.cross_attention(
            query, encoded, encoded,
            key_padding_mask=~action_mask
        )
        context_vector = context_vector.squeeze(1)
        
        context_vector = self.output_projection(context_vector)
        
        logits = torch.einsum('bnd,bd->bn', encoded, context_vector)
        logits = logits / torch.sqrt(torch.tensor(self.embedding_size, dtype=torch.float32))
        logits = self.C * torch.tanh(logits)
        
        logits = logits.masked_fill(~action_mask, -1e4)
        
        return logits


class TSPActor(nn.Module):
    """Actor network wrapper for policy."""
    
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        self.network = AutoregressiveTSP(embedding_size, hidden_size, seq_len, n_head, C)
    
    def forward(self, observation):
        """Get action distribution.
        
        Returns:
            dist: Categorical distribution
            logits: Raw logits
        """
        logits = self.network(observation)
        dist = Categorical(logits=logits)
        return dist, logits
    
    def get_action(self, observation, deterministic=False):
        """Sample or select action.
        
        Returns:
            action: [batch_size]
            log_prob: [batch_size]
        """
        dist, logits = self.forward(observation)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob