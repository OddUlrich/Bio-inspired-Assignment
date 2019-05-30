# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

from data_loader import data_loader
from RNN_model import RNN_model

import torch.nn as nn

'''
Dimension Statement:
    1  :  (1028, 16)
    2  :  (980, 16)
    3  :  (932, 16)
    4  :  (964, 16)
    5  :  (868, 16)
    6  :  (836, 16)
    7  :  (808, 16)
    8  :  (840, 16)
    9  :  (956, 16)
    10 :  (1104, 16)
    11 :  (956, 16)
    12 :  (965, 16)
'''
input_dim = 255 # Fake.

hidden_dim = 30
layer_dim = 2
output_dim = 3  # Four kinds of genres within 12 songs.

sample_num = 16 * 12
target_num = 3
max_seq_len = 1104

seqs, labels = data_loader('music-affect_v2', sample_num, max_seq_len, targer_num)

print(seqs.size)
print(labels.shape)
    
nn.utils.rnn.pack_padded_sequence()

rnn = RNN_model(input_dim, hidden_dim, layer_dim, output_dim)

nn.utils.rnn.pad_packed_sequence()

print(rnn)
print(rnn.hidden_dim)
print(rnn.layer_dim)
print(rnn.rnn.all_weights)
print(rnn.fc.weight)
print(rnn.softmax.dim)
