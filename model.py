from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class NN(nn.Module):
    def __init__(self, n_in, n_out):
        super(NN, self).__init__()

        self.act = nn.ReLU

        self.fc1 = nn.Linear(n_in, 64)
        self.act1 = self.act()

        self.fc2 = nn.Linear(64, 32)
        self.act2 = self.act()

        self.fc3 = nn.Linear(32, n_out)

        self.dropout = nn.Dropout(.1)

    def forward(self, X):
        X = self.fc1(X)
        X = self.act1(X)
        X = self.dropout(X)

        X = self.fc2(X)
        X = self.act2(X)

        X = self.fc3(X)
        return(X)

