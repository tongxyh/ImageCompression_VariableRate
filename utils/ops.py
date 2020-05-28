import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as f
import numpy as np

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

# Low_bound make the numerical calculation close to the bound
class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lower_bound=1e-6):
        ctx.save_for_backward(x)
        ctx.lower_bound = lower_bound
        x = torch.clamp(x, min=lower_bound)
        return x

    @staticmethod
    def backward(ctx, g):
        [x] = ctx.saved_tensors
        pass_through_if = (x>=ctx.lower_bound) + (g<0.0) * (g>-20.0)
        return g * pass_through_if.float(), None

class GDN(nn.Module):
    def __init__(self,channel_num,inverse=False,gama_init=0.1,beta_min=1e-6,reparam_offset=2**-18):
        super(GDN,self).__init__()

        self.inverse = bool(inverse)
        self.beta_min = float(beta_min)
        self.channel_num = int(channel_num)
        self.gama_init = float(gama_init)
        self.reparam_offset = float(reparam_offset)
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**0.5
        self.gama_bound = self.reparam_offset

        beta_initializer = torch.sqrt(torch.ones(self.channel_num)+self.pedestal)
        init_matrix = torch.eye(channel_num, channel_num)
        init_matrix = torch.unsqueeze(init_matrix, dim=-1)
        init_matrix = torch.unsqueeze(init_matrix, dim=-1)
        gamma_initializer = torch.sqrt(self.gama_init*init_matrix+self.pedestal)

        self.beta = Parameter(torch.Tensor(channel_num))
        self.beta.data.copy_(beta_initializer)

        self.gama = Parameter(torch.Tensor(self.channel_num, self.channel_num, 1, 1))
        self.gama.data.copy_(gamma_initializer)

    def forward(self, x):
        gama = Low_bound.apply(self.gama, self.gama_bound)

        gama = gama ** 2 - self.pedestal
        beta = Low_bound.apply(self.beta, self.beta_bound)

        beta = beta ** 2 - self.pedestal

        norm_pool = f.conv2d(x ** 2.0, weight=gama, bias=beta)
        if self.inverse:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.rsqrt(norm_pool)

        return x * norm_pool




