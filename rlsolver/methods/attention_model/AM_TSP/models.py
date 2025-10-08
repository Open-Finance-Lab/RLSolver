# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from layers import GraphEmbedding, AttentionModule, AttentionLayer, GatedAttentionLayer


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
            query: [batch_size, pomo_size, embed_dim] or [batch_size, 1, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            key_padding_mask: [batch_size, pomo_size, seq_len] if query has pomo_size
        
        Returns:
            output: [batch_size, pomo_size, embed_dim]
            attn_weights: [batch_size, pomo_size, num_heads, seq_len]
        """
        batch_size = query.size(0)
        pomo_size = query.size(1)
        seq_len = key.size(1)
        
        # Project Q, K, V
        Q = self.q_proj(query)  # [batch, pomo, embed_dim]
        K = self.k_proj(key)    # [batch, seq_len, embed_dim]
        V = self.v_proj(value)  # [batch, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, pomo_size, self.num_heads, self.head_dim).transpose(1, 2)
        # K and V are shared across POMO - no expansion needed
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores - broadcast automatically handles POMO dimension
        scores = torch.einsum('bhpd,bhsd->bhps', Q, K) * self.scale
        
        # Apply mask if provided
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1)  # [batch, 1, pomo, seq_len]
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.einsum('bhps,bhsd->bhpd', attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, pomo_size, self.embed_dim)
        output = self.out_proj(context)
        
        return output, attn_weights


class AutoregressiveTSP(nn.Module):
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.C = C
        
        # Node embedding
        self.embedding = GraphEmbedding(2, embedding_size)
        
        # Shared encoder with mixed architecture
        self.encoder = self._build_encoder(embedding_size, n_head)
        
        # Context processing
        self.h_context_embed = nn.Linear(embedding_size, embedding_size)
        
        # Query construction
        self.current_embed = nn.Linear(embedding_size, embedding_size)
        self.first_embed = nn.Linear(embedding_size, embedding_size)
        
        # Cross attention
        self.cross_attention = ManualCrossAttention(embedding_size, n_head)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_size, embedding_size)
    
    def _build_encoder(self, embed_dim, n_heads):
        """构建混合编码器：标准注意力 + 门控注意力"""
        layers = nn.ModuleList([
            AttentionLayer(embed_dim, n_heads, feed_forward_hidden=512),
            GatedAttentionLayer(embed_dim, n_heads, feed_forward_hidden=512),  # 门控层
            AttentionLayer(embed_dim, n_heads, feed_forward_hidden=512)
        ])
        
        class MixedEncoder(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return MixedEncoder(layers)
    
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
        seq_len = nodes.size(1)
        
        # Determine POMO size from action_mask
        if action_mask.dim() == 3:
            pomo_size = action_mask.size(1)
        else:
            pomo_size = 1
            action_mask = action_mask.unsqueeze(1)
        
        # Encode only once per batch (shared across POMO)
        if encoded is not None:
            pass  # Use cached encoding
        else:
            embedded = self.embedding(nodes)
            encoded = self.encoder(embedded)  # [batch, seq_len, embed_dim]
        
        # Compute context (shared across POMO)
        h_mean = encoded.mean(dim=1)  # [batch, embed_dim]
        h_context = self.h_context_embed(h_mean)  # [batch, embed_dim]
        
        # Build query vector for each POMO
        query = h_context.unsqueeze(1).expand(-1, pomo_size, -1).clone()  # [batch, pomo, embed_dim]
        
        # Add current node embedding using gather with proper indexing
        if current_node is not None:
            # current_node: [batch, pomo]
            batch_idx = torch.arange(batch_size, device=encoded.device)[:, None].expand(-1, pomo_size)
            current_h = encoded[batch_idx, current_node]  # [batch, pomo, embed_dim]
            query = query + self.current_embed(current_h)
        
        # Add first node embedding
        if first_node is not None:
            # first_node: [batch, pomo]
            batch_idx = torch.arange(batch_size, device=encoded.device)[:, None].expand(-1, pomo_size)
            first_h = encoded[batch_idx, first_node]  # [batch, pomo, embed_dim]
            query = query + self.first_embed(first_h)
        
        # Cross-attention (encoded is shared, query is per-POMO)
        context_vector, _ = self.cross_attention(
            query, encoded, encoded,
            key_padding_mask=~action_mask
        )
        
        # Project context
        context_vector = self.output_projection(context_vector)  # [batch, pomo, embed_dim]
        
        # Compute logits using einsum (broadcasting handles POMO dimension)
        # encoded: [batch, seq_len, embed_dim]
        # context_vector: [batch, pomo, embed_dim]
        logits = torch.einsum('bse,bpe->bps', encoded, context_vector)
        
        # Scale and clip
        logits = logits / torch.sqrt(torch.tensor(self.embedding_size, dtype=torch.float32))
        logits = self.C * torch.tanh(logits)
        
        # Apply action mask
        logits = logits.masked_fill(~action_mask, -1e4)
        
        return logits  # [batch, pomo, seq_len]


class TSPActor(nn.Module):
    """Actor network wrapper for policy."""
    
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        self.network = AutoregressiveTSP(embedding_size, hidden_size, seq_len, n_head, C)
    
    def forward(self, observation):
        """Get action distribution.
        
        Returns:
            dist: Categorical distribution
            logits: Raw logits [batch, pomo, seq_len]
        """
        logits = self.network(observation)
        
        # Handle both 2D and 3D logits
        if logits.dim() == 3:
            # [batch, pomo, seq_len] -> reshape for Categorical
            batch_size, pomo_size, seq_len = logits.shape
            logits_flat = logits.view(batch_size * pomo_size, seq_len)
            dist = Categorical(logits=logits_flat)
            return dist, logits
        else:
            dist = Categorical(logits=logits)
            return dist, logits
    
    def get_action(self, observation, deterministic=False):
        """Sample or select action.
        
        Returns:
            action: [batch_size, pomo_size] or [batch_size]
            log_prob: [batch_size, pomo_size] or [batch_size]
        """
        dist, logits = self.forward(observation)
        
        if logits.dim() == 3:
            batch_size, pomo_size, seq_len = logits.shape
            
            if deterministic:
                action = logits.argmax(dim=-1)  # [batch, pomo]
                log_prob = dist.log_prob(action.view(-1)).view(batch_size, pomo_size)
            else:
                action_flat = dist.sample()  # [batch*pomo]
                action = action_flat.view(batch_size, pomo_size)
                log_prob = dist.log_prob(action_flat).view(batch_size, pomo_size)
            
            return action, log_prob
        else:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action, log_prob