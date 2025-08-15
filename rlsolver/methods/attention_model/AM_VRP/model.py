import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import MultiHeadAttention
from environment import AgentVRP


class MultiHeadAttentionLayer(nn.Module):
    """MHA layer with feed-forward network and skip connections."""
    
    def __init__(self, input_dim, num_heads, feed_forward_hidden=512):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, input_dim)
        self.ff1 = nn.Linear(input_dim, feed_forward_hidden)
        self.ff2 = nn.Linear(feed_forward_hidden, input_dim)
        
    def forward(self, x, mask=None):
        mha_out = self.mha(x, x, x, mask)
        x = torch.tanh(x + mha_out)
        
        ff_out = self.ff2(F.relu(self.ff1(x)))
        x = torch.tanh(x + ff_out)
        
        return x


class GraphAttentionEncoder(nn.Module):
    """Graph encoder using multi-head attention layers."""
    
    def __init__(self, input_dim, num_heads, num_layers, feed_forward_hidden=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        self.init_embed_depot = nn.Linear(2, input_dim)
        self.init_embed = nn.Linear(3, input_dim)
        
        self.mha_layers = nn.ModuleList([
            MultiHeadAttentionLayer(input_dim, num_heads, feed_forward_hidden)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None, cur_num_nodes=None):
        depot, loc, demand = x
        batch_size = loc.size(0)
        
        depot_embedding = self.init_embed_depot(depot).unsqueeze(1)
        loc_with_demand = torch.cat([loc, demand.unsqueeze(-1)], dim=-1)
        loc_embedding = self.init_embed(loc_with_demand)
        
        x = torch.cat([depot_embedding, loc_embedding], dim=1)
        
        for layer in self.mha_layers:
            x = layer(x, mask)
        
        if mask is not None:
            graph_embedding = x.sum(dim=1) / cur_num_nodes
        else:
            graph_embedding = x.mean(dim=1)
        
        return x, graph_embedding


class AttentionDynamicModel(nn.Module):
    """Dynamic attention model for VRP with encoder-decoder architecture."""
    
    def __init__(self, embedding_dim, n_encode_layers=2, n_heads=8, tanh_clipping=10.):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.decode_type = "sampling"
        
        self.problem = AgentVRP
        
        self.embedder = GraphAttentionEncoder(
            input_dim=embedding_dim,
            num_heads=n_heads,
            num_layers=n_encode_layers
        )
        
        self.output_dim = embedding_dim
        self.head_depth = self.output_dim // self.n_heads
        
        self.wq_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wq_step_context = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
        
        self.wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wk_tanh = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.w_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
    def set_decode_type(self, decode_type):
        self.decode_type = decode_type
        
    def _select_node(self, logits):
        """Select next node based on decoding strategy."""
        if self.decode_type == "greedy":
            selected = logits.argmax(dim=-1).squeeze(-1)
        elif self.decode_type == "sampling":
            probs = F.softmax(logits.squeeze(1), dim=-1)
            selected = torch.multinomial(probs, 1).squeeze(-1)
        else:
            raise ValueError(f"Unknown decode type: {self.decode_type}")
        return selected
    
    def get_step_context(self, state, embeddings):
        """Get context for current decoding step."""
        prev_node = state.prev_a.long()
        cur_embedded_node = torch.gather(
            embeddings, 1, 
            prev_node.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
        )
        
        remaining_capacity = self.problem.VEHICLE_CAPACITY - state.used_capacity
        step_context = torch.cat([cur_embedded_node, remaining_capacity.unsqueeze(-1)], dim=-1)
        
        return step_context
    
    def decoder_mha(self, Q, K, V, mask=None):
        """Multi-head attention for decoder."""
        batch_size = Q.size(0)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_depth).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_depth)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(attention_weights, V)
        
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.output_dim
        )
        
        return self.w_out(attention)
    
    def get_log_p(self, Q, K, mask=None):
        """Compute log probabilities for node selection."""
        compatibility = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.output_dim)
        compatibility = torch.tanh(compatibility) * self.tanh_clipping
        
        if mask is not None:
            compatibility = compatibility.masked_fill(mask, float('-inf'))
        
        return F.log_softmax(compatibility, dim=-1)
    
    def get_log_likelihood(self, log_p, actions):
        """Calculate log likelihood of selected actions."""
        log_p_selected = torch.gather(log_p, 2, actions.long().unsqueeze(-1))
        return log_p_selected.squeeze(-1).sum(1)
    
    def get_projections(self, embeddings, context_vectors):
        """Pre-compute projections for decoding."""
        batch_size = embeddings.size(0)
        
        K = self.wk(embeddings)
        K_tanh = self.wk_tanh(embeddings)
        V = self.wv(embeddings)
        Q_context = self.wq_context(context_vectors.unsqueeze(1))
        
        K = K.view(batch_size, -1, self.n_heads, self.head_depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_depth).transpose(1, 2)
        
        return K_tanh, Q_context, K, V
    
    def forward(self, inputs, return_pi=False):
        embeddings, mean_graph_emb = self.embedder(inputs)
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        outputs = []
        sequences = []
        
        state = self.problem(inputs)
        K_tanh, Q_context, K, V = self.get_projections(embeddings, mean_graph_emb)
        
        i = 0
        while not state.all_finished():
            if i > 0:
                state.i = torch.zeros(1, dtype=torch.long, device=device)
                att_mask, cur_num_nodes = state.get_att_mask()
                embeddings, context_vectors = self.embedder(inputs, att_mask, cur_num_nodes)
                K_tanh, Q_context, K, V = self.get_projections(embeddings, context_vectors)
            
            while not state.partial_finished():
                step_context = self.get_step_context(state, embeddings)
                Q_step_context = self.wq_step_context(step_context)
                Q = Q_context + Q_step_context
                
                mask = state.get_mask()
                
                mha = self.decoder_mha(Q, K, V, mask)
                
                log_p = self.get_log_p(mha, K_tanh, mask)
                
                selected = self._select_node(log_p)
                state.step(selected)
                
                outputs.append(log_p.squeeze(1))
                sequences.append(selected)
            
            i += 1
        
        log_p_stack = torch.stack(outputs, dim=1)
        pi = torch.stack(sequences, dim=1).float()
        
        cost = self.problem.get_costs(inputs, pi)
        ll = self.get_log_likelihood(log_p_stack, pi)
        
        if return_pi:
            return cost, ll, pi
        return cost, ll