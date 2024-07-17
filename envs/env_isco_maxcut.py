import torch
from scipy.stats import poisson
import torch
from config import *
from torch.func import vmap

class iSCO:
    def __init__(self,data):
        self.data_directory =DATA_ROOT
        self.batch_size = BATCH_SIZE
        self.device = torch.device(DEVICE)
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE,device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE,device=self.device)
        self.device = torch.device(DEVICE)
        self.edge_from = data['edge_from']
        self.edge_to = data['edge_to']      
        # self.edge_from = data['edge_from'].unsqueeze(0).repeat(self.batch_size, 1)
        # self.edge_to = data['edge_to'].unsqueeze(0).repeat(self.batch_size, 1)
        self.max_num_nodes = data['num_nodes']
        self.max_num_edges = data['num_edges']

    def model(self,x):
        delta_x = (x * 2 - 1)
        gather2src = torch.gather(delta_x, -1,  self.edge_from)
        gather2dst = torch.gather(delta_x, -1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = - torch.sum(is_cut,dim=-1)
        return energy


    def get_init_sample(self):
        sample = torch.bernoulli(torch.full((self.batch_size, self.max_num_nodes,), 0.5, device=self.device))
        return sample


class iSCO_fast_vmap(iSCO):
    def __init__(self,data):
        super().__init__(data)
        self.sampler = 'iSCO_fast_vmap'
    
    def flip(self,x,i_n):
        return x*(1-2*i_n)+i_n

    def x2y(self,x,I_N,idx_list,traj,path_length,temperature):
        with torch.no_grad():
            cur_x = x.clone()
            energy_x = self.model(x)

            for step in range(path_length):
                neighbor_x = vmap(self.flip,in_dims = (None,0))(x,I_N)
                neighbor_x_energy = vmap(self.model,in_dims=0)(neighbor_x)
                score_change_x = -(neighbor_x_energy - energy_x) / (2 * temperature)
                prob_x_local = torch.log_softmax(score_change_x, dim=-1)
                traj[:,step] = prob_x_local
                index = torch.multinomial(prob_x_local.exp(), 1).view(-1)
                idx_list[step] = index
                cur_x[index] = 1.0 - cur_x[index]

        return cur_x,idx_list,traj
    
    def y2x(self,x,energy_x,I_N,y,energy_y,grad_y,idx_list,traj,path_length,temperature):
        with torch.no_grad():
            r_idx = torch.arange(path_length,device=self.device).view(1, -1)
            neighbor_y = vmap(self.flip,in_dims = (None,0))(y,I_N)
            neighbor_y_energy = vmap(self.model,in_dims=0)(neighbor_y)
            score_change_y = -(neighbor_y_energy - energy_y) / (2 * temperature)
            prob_y_local = torch.log_softmax(score_change_y, dim=-1)

            # fwd from x -> y
            log_fwd = torch.sum(traj[:,:-1][idx_list,r_idx], dim=-1) + energy_x.view(-1)

            # backwd from y -> x
            delta_y = 1.0 - 2.0 * y
            traj[:,path_length] = prob_y_local
            log_backwd = torch.sum(traj[:,1:][idx_list, r_idx], dim=-1) + energy_y.view(-1)

            log_acc = log_backwd - log_fwd
            accs = torch.clamp(log_acc.exp(), max=1)
            mask = accs >= torch.rand_like(accs)
            new_x = torch.where(mask,y,x)
            energy = torch.where(mask,energy_y,energy_x)

        return new_x,energy,accs    
    
    def step(self,path_length,temperature,sample):
        x = sample.detach().requires_grad_(True)
        energy_x = self.model(x[0])
        idx_list = torch.empty(self.batch_size,path_length, device=self.device,dtype=torch.int)
        traj = torch.empty(self.batch_size,self.max_num_nodes,path_length+1,device=self.device)
        I_N = torch.eye(self.max_num_nodes,device=self.device)
        cur_x,idx_list,traj = self.x2y(x,I_N,idx_list,traj,path_length,temperature)
        y = cur_x.detach().requires_grad_(True)
        energy_y, grad_y = self.model(y)
        new_x,energy,accs = self.y2x(x,energy_x,grad_x,y,energy_y, grad_y,idx_list,traj,path_length,temperature)
        accs = torch.mean(accs)
        return new_x,energy,accs
    