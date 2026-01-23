"""
Copyright (c) 2024 Cheng Chen, Ruitao Chen, Tianyou Li, Zaiwen Wen
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from config import DEVICE
class simple(torch.nn.Module):
    def __init__(self, output_num):
        super(simple, self).__init__()
        self.lin = torch.nn.Linear(1,  output_num)
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, alpha = 0.1, start_samples=None, value=None, 
                device=DEVICE):
        x = torch.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x-0.5) * 0.6 + 0.5
        probs = x
        probs = probs.squeeze()
        retdict = {}
        reg = probs * torch.log(probs) + (1-probs) * torch.log(1-probs )
        reg = torch.mean(reg)
        if start_samples == None:
            retdict["output"] = [probs.squeeze(-1), "hist"]  # output
            retdict["reg"] = [reg, "sequence"]
            retdict["loss"] = [alpha * reg, "sequence"]
            return retdict

        res_samples = value.t().detach()

        start_samples_idx = start_samples * \
            probs + (1 - start_samples) * (1 - probs)
        log_start_samples_idx = torch.log(start_samples_idx)
        log_start_samples = log_start_samples_idx.sum(dim=1)
        loss_ls = torch.mean(log_start_samples * res_samples)
        loss = loss_ls + alpha * reg

        retdict["output"] = [probs.squeeze(-1), "hist"]  # output
        retdict["reg"] = [reg, "sequence"]
        retdict["loss"] = [loss, "sequence"]
        return retdict

    def __repr__(self):
        return self.__class__.__name__
