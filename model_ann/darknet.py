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
class ANN(nn.Module):

    def __init__(self, num_input=721, num_hidden=100, num_output=721):
        super(ANN, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(inplace=True)
        )

        self._output_ = nn.Linear(num_hidden, num_output)

        self._initialize_weights()

    def forward(self, x):
        x = self.hidden(x)

        x = self._output_(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.fill_(0)
