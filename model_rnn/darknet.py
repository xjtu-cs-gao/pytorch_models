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
class RNN(nn.Module):

    def __init__(self, num_input=1, num_hidden=10, num_layer=1, num_output=1):
        super(RNN, self).__init__()

        self.num_hidden = num_hidden
        self.num_layer = num_layer

        self.rnn = nn.GRU(
            num_input,
            num_hidden,
            num_layer,
            batch_first=True
        )
        self.fc = nn.Linear(num_hidden, num_output)

        self._initialize_weights()

    def forward(self, x):
        h0 = None

        x, h = self.rnn(x, h0)

        outs = []
        for time_step in range(x.size(1)):
            outs.append(self.fc(x[:, time_step, :]))

        return torch.stack(outs, dim=1)

    def _initialize_weights(self):
        self.fc.weight.data.uniform_(0, 0.1)
        self.fc.bias.data.fill_(0)
