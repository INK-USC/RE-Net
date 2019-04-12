import math
import torch as th
import torch
from torch import nn

import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, in_dim, n_layer):
        super(LSTMModel, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = in_dim
        # self.lstm = nn.LSTM(in_dim, self.hidden_dim, n_layer, batch_first=True)
        self.lstm = LSTMLinear(in_dim, self.hidden_dim)

    def forward(self, x):
        out, h = self.lstm(x)
        return h[0]


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.linear_acti = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.contiguous()
        if hidden is None:
            hidden = self._init_hidden(x, self.hidden_size)

        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)

        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        # g_t = preact[:, 3 * self.hidden_size:].tanh()
        g_t = preact[:, 3 * self.hidden_size:]
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)
        h_t = th.mul(o_t, c_t)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, c_t

    @staticmethod
    def _init_hidden(input_, hidden_size):
        h = th.zeros_like(input_.view(1, input_.size(0), -1))
        c = th.zeros_like(input_.view(1, input_.size(0), -1))
        return h, c


class LSTMLinear(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMLinear, self).__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.batch_first = True

    def forward(self, input_, hidden=None):

        if self.batch_first:
            input_ = input_.transpose(0, 1)

        outputs = []
        steps = range(input_.size(0))
        for i in steps:
            hidden = self.lstm_cell(input_[i], hidden)
            if isinstance(hidden, tuple):
                outputs.append(hidden[0])
            else:
                outputs.append(hidden)

        outputs = torch.stack(outputs, dim=0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, hidden
