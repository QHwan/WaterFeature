from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layers import *


class NN(nn.Module):
    def __init__(self, n_in, n_out):
        super(NN, self).__init__()

        self.act = nn.ReLU

        self.fc1 = nn.Linear(n_in, 128)
        self.act1 = self.act()

        #self.fc2 = nn.Linear(128, 64)
        #self.act2 = self.act()

        self.enc1 = nn.Linear(128, 2)
        self.act3 = nn.LeakyReLU()

        self.fc3 = nn.Linear(2, n_out)

        self.dropout = nn.Dropout(.2)

    def forward(self, X):
        X = self.fc1(X)
        X = self.act1(X)
        X = self.dropout(X)

        X_enc = self.enc1(X)

        X = self.fc3(X_enc)
        return(X, X_enc)


class GCN(nn.Module):
    def __init__(self, n_in, n_out):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_in, n_in)
        self.gc2 = GraphConvolution(n_in, n_in)
        self.nn1 = nn.Linear(n_in, 32)
        self.nn2 = nn.Linear(32, n_out)
        

    def forward(self, X, A):
        X = F.relu(self.gc1(X, A))
        X = F.relu(self.nn1(X))
        X = self.nn2(X)
        return(X)

