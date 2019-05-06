# Library
# standard library
import math
from collections import namedtuple

# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# home-brew python file
import opt


# Define RNN.
class MLP(nn.Module):

    def __init__(self, num_input=721, num_hidden=100, num_output=721):
        super(MLP, self).__init__()

        self._input_ = nn.Linear(num_input, num_hidden)
        self.hidden0 = nn.Sequential(
            nn.BatchNorm1d(num_hidden, momentum=0.5),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden1 = nn.Sequential(
            nn.BatchNorm1d(num_hidden, momentum=0.5),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.BatchNorm1d(num_hidden, momentum=0.5),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self._output_ = nn.Linear(num_hidden, num_output)

        self._initialize_weights()

    def forward(self, x):
        x = self._input_(x)

        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)

        x = self._output_(x)

        return x

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(0, 1)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)
