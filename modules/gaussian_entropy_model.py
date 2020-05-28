import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from utils import ops

class Distribution_for_entropy2(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy2,self).__init__()

    def forward(self, x, p_dec, quan_step = 1.):

        mean = p_dec[:, 0, :, :, :]
        scale = p_dec[:, 1, :, :, :]

        ## to make the scale always positive
        # scale[scale == 0] = 1e-9
        scale = ops.Low_bound.apply(torch.abs(scale), 1e-9)
        #scale1 = torch.clamp(scale1,min = 1e-9)
        m1 = torch.distributions.normal.Normal(mean,scale)
        lower = m1.cdf(x - 0.5 * quan_step)
        upper = m1.cdf(x + 0.5 * quan_step)

        likelihood = torch.abs(upper - lower)

        likelihood = ops.Low_bound.apply(likelihood,1e-6)
        return likelihood