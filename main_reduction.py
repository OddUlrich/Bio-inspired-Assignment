# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
from Net import Net, test_model
from utils import confusion, F1_score, loadDataset

# Reload the parameters of the trained model.
load_net = Net(11, 30, 3)
load_net.load_state_dict(torch.load('net_model.pt'))
load_net.eval()

"""
Scheme for units removal:
    #5, 14   ->  2
    18      -> 11
    #29      -> 16
    
The units that will be removed are 5, 14, 18, #29.
"""


## Operation of addition.
#load_net.hidden.weight[0] += (load_net.hidden.weight[10] + load_net.hidden.weight[29])
#load_net.hidden.weight[5] += load_net.hidden.weight[3]
#load_net.hidden.weight[19] += (load_net.hidden.weight[21] + load_net.hidden.weight[22])
#load_net.hidden.weight[26] += load_net.hidden.weight[24]
#
## Slicing the remained weight values and bias values in a new-sized network.
#new_net = Net(11, 24, 3)
#new_net.hidden.weight[: 3] = load_net.hidden.weight[0: 3]        # Unit 1 - 3
#new_net.hidden.weight[3: 9] = load_net.hidden.weight[4: 10]     # Unit 5 - 10
#new_net.hidden.weight[9: 19] = load_net.hidden.weight[11: 21]     # Unit 12 - 21
#new_net.hidden.weight[19] = load_net.hidden.weight[23]         # Unit 24
#new_net.hidden.weight[20:] = load_net.hidden.weight[25: 29]     # Unit 26 - 29
#
#new_net.hidden.bias[: 3] = load_net.hidden.bias[0: 3]        # Unit 1 - 3
#new_net.hidden.bias[3: 9] = load_net.hidden.bias[4: 10]     # Unit 5 - 10
#new_net.hidden.bias[9: 19] = load_net.hidden.bias[11: 21]     # Unit 12 - 21
#new_net.hidden.bias[19] = load_net.hidden.bias[23]         # Unit 24
#new_net.hidden.bias[20:] = load_net.hidden.bias[25: 29]     # Unit 26 - 29
#
#new_net.output.weight[:, : 3] = load_net.output.weight[:, 0: 3]        # Unit 1 - 3
#new_net.output.weight[:, 3: 9] = load_net.output.weight[:, 4: 10]     # Unit 5 - 10
#new_net.output.weight[:, 9: 19] = load_net.output.weight[:, 11: 21]     # Unit 12 - 21
#new_net.output.weight[:, 19] = load_net.output.weight[:, 23]         # Unit 24
#new_net.output.weight[:, 20:] = load_net.output.weight[:, 25: 29]     # Unit 26 - 29
#
#new_net.output.bias[:] = load_net.output.bias[:]
#
#new_net.eval()
#
## Reload the test dateset and evaluate the shrinked network.
#x_train, y_train, x_test, y_test = loadDataset()
#acc, pred = test_model(new_net, x_test, y_test)
#
#mat = confusion(x_test.size(0), 3, pred, y_test)
#print("Confusion Matrix (after pruning)：")
#print(mat)
#F1_score(mat)
#

