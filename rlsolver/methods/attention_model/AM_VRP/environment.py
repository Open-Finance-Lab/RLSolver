import torch


class AgentVRP:
    """VRP environment for reinforcement learning."""
    
    VEHICLE_CAPACITY = 1.0
    
    def __init__(self, inputs):
        depot, loc, demand = inputs
        self.batch_size, self.n_loc, _ = loc.shape
        
        self.coords = torch.cat([depot.unsqueeze(1), loc], dim=1)
        self.demand = demand.float()
        
        self.ids = torch.arange(self.batch_size, device=loc.device).unsqueeze(1)
        
        self.prev_a = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=loc.device)
        self.from_depot = self.prev_a == 0
        self.used_capacity = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=loc.device)
        
        self.visited = torch.zeros((self.batch_size, 1, self.n_loc + 1), dtype=torch.uint8, device=loc.device)
        
        self.i = torch.zeros(1, dtype=torch.long, device=loc.device)
        
    def get_att_mask(self):
        """Get attention mask for encoder."""
        att_mask = self.visited.squeeze(1)[:, 1:].float()
        cur_num_nodes = self.n_loc + 1 - att_mask.sum(-1, keepdim=True)
        att_mask = torch.cat([torch.zeros(att_mask.size(0), 1, device=att_mask.device), att_mask], dim=-1)
        
        att_mask = (att_mask.unsqueeze(-1) + att_mask.unsqueeze(-2) - 
                   att_mask.unsqueeze(-1) * att_mask.unsqueeze(-2))
        
        return att_mask.bool(), cur_num_nodes
    
    def all_finished(self):
        """Check if all tours are finished."""
        return self.visited.all()
    
    def partial_finished(self):
        """Check if all agents returned to depot."""
        return self.from_depot.all() and self.i.item() != 0
    
    def get_mask(self):
        """Get mask for available actions."""
        visited_loc = self.visited[:, :, 1:]
        exceeds_cap = (self.demand + self.used_capacity) > self.VEHICLE_CAPACITY
        
        mask_loc = visited_loc.bool() | exceeds_cap.unsqueeze(1) | \
                  ((self.i > 0) & self.from_depot.unsqueeze(1))
        
        mask_depot = self.from_depot & ((~mask_loc).sum(dim=-1) > 0)
        
        return torch.cat([mask_depot.unsqueeze(-1), mask_loc], dim=-1)
    
    def step(self, action):
        """Execute action and update state."""
        if action.dim() == 2:
            action = action.squeeze(-1)
        
        selected = action.unsqueeze(1)
        
        self.prev_a = selected.float()
        self.from_depot = self.prev_a == 0
        
        indices = torch.clamp(self.prev_a - 1, 0, self.n_loc - 1).long()
        selected_demand = torch.gather(self.demand, 1, indices)
        
        self.used_capacity = (self.used_capacity + selected_demand) * (~self.from_depot).float()
        
        self.visited.scatter_(2, self.prev_a.long().unsqueeze(1), 1)
        
        self.i = self.i + 1
    
    @staticmethod
    def get_costs(dataset, pi):
        """Calculate total tour costs."""
        depot, loc = dataset[0], dataset[1]
        loc_with_depot = torch.cat([depot.unsqueeze(1), loc], dim=1)
        
        d = torch.gather(loc_with_depot, 1, pi.long().unsqueeze(-1).expand(-1, -1, 2))
        
        distances = torch.norm(d[:, 1:] - d[:, :-1], p=2, dim=2).sum(1)
        distances = distances + torch.norm(d[:, 0] - depot, p=2, dim=1)
        distances = distances + torch.norm(d[:, -1] - depot, p=2, dim=1)
        
        return distances