# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

from data_loader import data_loader
from RNN_model import RNN_model

import torch.nn as nn

input_dim = 255 # Fake.

hidden_dim = 30
layer_dim = 2
output_dim = 3  # Four kinds of genres within 12 songs.

# Return size: [batch_size, length]
input_train, label_train, input_test, label_test = data_loader()





#
#rnn = RNN_model(input_dim, hidden_dim, layer_dim, output_dim)
#
#x = nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=False)
#x, h = rnn(x)
#x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
#
#
#print(rnn)
#
#
#print(rnn.rnn.all_weights)
#print("+++++++++++++++++++++++++++++++++++++")
#print(rnn.fc.weight)
#

