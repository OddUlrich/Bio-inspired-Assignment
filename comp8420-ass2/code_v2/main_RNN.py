# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
import numpy as np
from data_loader import seq_loader
from RNN_model import RNN_model, train_model, test_model
from utils import confusion, F1_score, saveNNParas, saveDataset
import time

############################################
# Hyper parameters
input_dim = 1
hidden_dim = 50
layer_dim = 1
output_dim = 3  # Four kinds of genres within 12 songs.

epochs = 200
lr = 0.003
############################################


# Return size: [batch_size, length]
input_train, label_train, input_test, label_test = seq_loader()

rnn = RNN_model(input_dim, hidden_dim, layer_dim, output_dim)


# Various sequence length used for padding sequence and packed sequence in rnn modol.
l = [1104, 1028, 980, 964, 960, 956, 956, 932, 868, 840, 836, 808]
train_seq_lens = np.zeros((12*12))
for i in range(len(l)):
    train_seq_lens[i*12: (i+1)*12] = l[i]
test_seq_lens = np.zeros((4*12))
for i in range(len(l)):
    test_seq_lens[i*4: (i+1)*4] = l[i]   

# Unsqueeze from 2-dimension to 3-dimension to match the rnn model.
flat_input_train = input_train.unsqueeze(-1)
train_model(rnn, flat_input_train, label_train, train_seq_lens, lr=lr, epochs=epochs)

flat_input_test = input_test.unsqueeze(-1)
start_time = time.time()
acc, pred = test_model(rnn, flat_input_test, label_test, test_seq_lens)
stop_time = time.time()
print("Execution time: %s ms" % ((stop_time - start_time)*1000))
            
if acc > 35:
    torch.save(rnn.state_dict(), 'rnn_model.pt')
    saveNNParas(rnn, input_test, hidden_dim)
    saveDataset(input_test, label_test, 'testing_sequence')
    
mat = confusion(input_test.size(0), output_dim, pred, label_test)
print("Confusion Matrix：")
print(mat)
F1_score(mat)
