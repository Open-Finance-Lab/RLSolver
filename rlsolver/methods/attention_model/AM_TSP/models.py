# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from layers import GraphEmbedding, AttentionLayer


class ManualCrossAttention(nn.Module):
    """Manual implementation of cross-attention with POMO support."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: [batch_size, pomo_size, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            key_padding_mask: [batch_size, pomo_size, seq_len]
        Returns:
            output: [batch_size, pomo_size, embed_dim]
        """
        batch_size = query.size(0)
        pomo_size = query.size(1)
        seq_len = key.size(1)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q = Q.view(batch_size, pomo_size, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.einsum('bhpd,bhsd->bhps', Q, K) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.einsum('bhps,bhsd->bhpd', attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, pomo_size, self.embed_dim)
        output = self.out_proj(context)
        return output


class AutoregressiveTSP(nn.Module):
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.C = C
        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = self._build_encoder(embedding_size, n_head)
        self.h_context_embed = nn.Linear(embedding_size, embedding_size)
        self.current_embed = nn.Linear(embedding_size, embedding_size)
        self.first_embed = nn.Linear(embedding_size, embedding_size)
        self.cross_attention = ManualCrossAttention(embedding_size, n_head)
        self.output_projection = nn.Linear(embedding_size, embedding_size)

    def _build_encoder(self, embed_dim, n_heads):
        """Build encoder with standard attention layers."""
        return nn.Sequential(
            AttentionLayer(embed_dim, n_heads, feed_forward_hidden=512),
            AttentionLayer(embed_dim, n_heads, feed_forward_hidden=512),
            AttentionLayer(embed_dim, n_heads, feed_forward_hidden=512)
        )

    def forward(self, observation):
        """Forward pass with POMO support using shared encoding.
        
        Args:
            observation: dict with:
                - nodes: [batch_size, seq_len, 2]
                - current_node: [batch_size, pomo_size] or None
                - first_node: [batch_size, pomo_size] or None
                - action_mask: [batch_size, pomo_size, seq_len]
                - encoded: [batch_size, seq_len, embedding_size] (optional, shared)
                
        Returns:
            logits: [batch_size, pomo_size, seq_len]
        """
        nodes = observation['nodes']
        current_node = observation.get('current_node')
        first_node = observation.get('first_node')
        action_mask = observation['action_mask']
        encoded = observation.get('encoded')
        batch_size = nodes.size(0)
        pomo_size = action_mask.size(1)
        if encoded is None:
            embedded = self.embedding(nodes)
            encoded = self.encoder(embedded)
        h_mean = encoded.mean(dim=1)
        h_context = self.h_context_embed(h_mean)
        query = h_context.unsqueeze(1).expand(-1, pomo_size, -1).clone()
        if current_node is not None:
            batch_idx = torch.arange(batch_size, device=encoded.device)[:, None].expand(-1, pomo_size)
            current_h = encoded[batch_idx, current_node]
            query = query + self.current_embed(current_h)
        if first_node is not None:
            batch_idx = torch.arange(batch_size, device=encoded.device)[:, None].expand(-1, pomo_size)
            first_h = encoded[batch_idx, first_node]
            query = query + self.first_embed(first_h)
        context_vector = self.cross_attention(
            query, encoded, encoded,
            key_padding_mask=~action_mask
        )
        context_vector = self.output_projection(context_vector)
        logits = torch.einsum('bse,bpe->bps', encoded, context_vector)
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
            logits: Raw logits [batch_size, pomo_size, seq_len]
        """
        logits = self.network(observation)
        batch_size, pomo_size, seq_len = logits.shape
        logits_flat = logits.view(batch_size * pomo_size, seq_len)
        dist = Categorical(logits=logits_flat)
        return dist, logits

    def get_action(self, observation, deterministic=False):
        """Sample or select action.
        
        Returns:
            action: [batch_size, pomo_size]
            log_prob: [batch_size, pomo_size]
        """
        dist, logits = self.forward(observation)
        batch_size, pomo_size, seq_len = logits.shape
        if deterministic:
            action = logits.argmax(dim=-1)
            log_prob = dist.log_prob(action.view(-1)).view(batch_size, pomo_size)
        else:
            action_flat = dist.sample()
            action = action_flat.view(batch_size, pomo_size)
            log_prob = dist.log_prob(action_flat).view(batch_size, pomo_size)
        return action, log_prob
