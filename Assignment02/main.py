# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

from data_loader import data_loader

import torch.nn as nn

data = data_loader('music-affect_v2')


class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, classes_num):
        super(LSTM_net, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, 1)
        self.fc = nn.Linear(hidden_size, classes_num)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_seq, hprev, cprev):
        # fixme: processing on input with .view()
        output, hc = self.LSTM(input, (hprev, cprev))
        output = self.fc(output)
        output = self.softmax(output)
        return output
