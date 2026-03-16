
import torch


use_cuda = torch.cuda.is_available()
# use_cuda = False

torch.cuda.set_device(0)

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
Tensor = FloatTensor

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
