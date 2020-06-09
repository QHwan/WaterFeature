from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, n_fea_in, n_fea_out):
        super(GraphConvolution, self).__init__()
        self.W_self = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.W_nei = Parameter(torch.FloatTensor(n_fea_in, n_fea_out))
        self.bias = Parameter(torch.FloatTensor(n_fea_out))
        
        self.reset_parameters_uniform(self.W_self)
        self.reset_parameters_uniform(self.W_nei)
        self.reset_parameters_uniform(self.bias)

    def reset_parameters_uniform(self, x):
        stdv = 1. / math.sqrt(x.size(0))
        x.data.uniform_(-stdv, stdv)

    def forward(self, H, A):
        H_filtered_self = torch.einsum("aij,jk->aik", (H, self.W_self)) 

        H_filtered_nei = torch.einsum("aij,jk->aik", (H, self.W_nei)) 
        H_filtered_nei = torch.bmm(A, H_filtered_nei) 
        H1 = H_filtered_self + H_filtered_nei + self.bias
        return(H1)