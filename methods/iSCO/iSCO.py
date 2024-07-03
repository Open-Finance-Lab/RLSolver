import torch
import tqdm
from scipy.stats import poisson
import torch
import time
import os
import networkx as nx
import math
from torch.func import vmap,grad
import until

class iSCO:
    def __init__(self,config,data):
        self.data_directory =config['data_root']
        self.batch_size = config['batch_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chain_length = config['chain_length']
        self.init_temperature = torch.tensor(config['init_temperature'],device=self.device)
        self.final_temperature = torch.tensor(config['final_temperature'],device=self.device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_from = data['edge_from'].unsqueeze(0).repeat(self.batch_size, 1)
        self.edge_to = data['edge_to'].unsqueeze(0).repeat(self.batch_size, 1)
        self.max_num_nodes = data['num_nodes']
        self.max_num_edges = data['num_edges']
        self.b_idx = torch.arange(self.batch_size,device=self.device)

    def get_energy(self,x):
        delta_x = (x * 2 - 1)
        gather2src = torch.gather(delta_x,1,self.edge_from)
        gather2dst = torch.gather(delta_x,1, self.edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = - torch.sum(is_cut,dim=-1)
        grad = torch.autograd.grad(outputs=energy, inputs=x, grad_outputs=torch.ones_like(energy), create_graph=False,retain_graph=False)[0]
        with torch.no_grad():
            grad = grad.detach()
            energy = energy.detach()
        return energy,grad

    def get_result(self):
            sample = self.get_init_sample()
            mu = 10.0
            result = torch.empty(self.batch_size,self.max_num_nodes,device=self.device)
            energy = torch.empty(self.batch_size,device=self.device)
            average_acc = 0
            average_path_length = 0
            if self.sampler == 'iSCO_fast_vmap':               
                self.x2y,self.y2x = until.parallelization(self.x2y,self.y2x)
            start_time = time.time()
            for step in tqdm.tqdm(range(0,self.chain_length)):
                poisson_dist = poisson(mu)
                path_length = max(1,int(poisson_dist.rvs(size=1)))
                average_path_length += path_length
                temperature = self.init_temperature  - step / self.chain_length * (self.init_temperature -self.final_temperature)
                sample,new_energy,acc = self.step(path_length,temperature,sample)
                acc = acc.item()
                mu = min(self.max_num_nodes,max(1,(mu + 0.01*(acc - 0.574))))
                average_acc+=acc

            obj, obj_index = torch.min(new_energy, dim=0)
            obj = obj.item()
            result = sample[obj_index].squeeze()
            average_acc,average_path_length = average_acc/self.chain_length,average_path_length/self.chain_length
            end_time = time.time()
            running_duration = end_time - start_time
            until.write_result(self.data_directory,result,obj,running_duration,self.max_num_nodes)

    def get_init_sample(self):
        sample = torch.bernoulli(torch.full((self.batch_size, self.max_num_nodes,), 0.5, device=self.device))
        return sample
class iSCO_fast(iSCO):
    def __init__(self,config,data):
        super().__init__(config,data)
    def step():
        pass

class iSCO_fast_vmap(iSCO):
    def __init__(self,config,data):
        super().__init__(config,data)
        self.sampler = 'iSCO_fast_vmap'

    def x2y(self,x,grad_x,idx_list,traj,path_length,temperature):
        with torch.no_grad():
            cur_x = x.clone()

            for step in range(path_length):
                delta_x = 1.0 - 2.0 * cur_x
                traj[:,step] = delta_x
                score_change_x = -(delta_x * grad_x) / (2 * temperature)
                score_change_x = score_change_x - torch.logsumexp(score_change_x, dim=-1, keepdim=True)
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                idx_list[step] = index
                cur_x[index] = 1.0 - cur_x[index]

        return cur_x,idx_list,traj
    
    def y2x(self,x,energy_x,grad_x,y,energy_y,grad_y,idx_list,traj,path_length,temperature):
        with torch.no_grad():
            r_idx = torch.arange(path_length,device=self.device).view(1, -1)

            # fwd from x -> y
            score_fwd = (traj[:,:-1] * grad_x.unsqueeze(1)) / (2 * temperature)
            log_fwd = torch.log_softmax(score_fwd, dim=-1)
            log_fwd = torch.sum(log_fwd[idx_list,r_idx], dim=-1) + energy_x.view(-1)

            # backwd from y -> x
            delta_y = 1.0 - 2.0 * y
            traj[:,path_length] = delta_y
            score_backwd = (traj[:,1:] * grad_y.unsqueeze(1)) / (2 * temperature)
            log_backwd = torch.log_softmax(score_backwd, dim=-1)
            log_backwd = torch.sum(log_backwd[idx_list, r_idx], dim=-1) + energy_y.view(-1)

            log_acc = log_backwd - log_fwd
            accs = torch.clamp(log_acc.exp(), max=1)
            mask = accs >= torch.rand_like(accs)
            new_x = torch.where(mask,y,x)
            energy = torch.where(mask,energy_y,energy_x)

        return new_x,energy,accs    
    
    def step(self,path_length,temperature,sample):
    
        x = sample.detach().requires_grad_(True)
        energy_x, grad_x = self.get_energy(x)
        idx_list = torch.empty(self.batch_size,path_length, device=self.device,dtype=torch.int)
        traj = torch.empty(self.batch_size,self.max_num_nodes,path_length+1,device=self.device)
        cur_x,idx_list,traj = self.x2y(x,grad_x,idx_list,traj,path_length,temperature)
        y = cur_x.detach().requires_grad_(True)
        energy_y, grad_y = self.get_energy(y)
        new_x,energy,accs = self.y2x(x,energy_x,grad_x,y,energy_y, grad_y,idx_list,traj,path_length,temperature)
        accs = torch.mean(accs)
        return new_x,energy,accs