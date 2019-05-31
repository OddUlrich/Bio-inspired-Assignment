# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
from Net import Net, test_model
from utils import confusion, F1_score, loadDataset

# Reload the parameters of the trained model.
load_net = Net(11, 30, 3)
load_net.load_state_dict(torch.load('net_model_subjective_rating.pt'))
load_net.eval()

"""
Manual operation on network reduction.

Scheme for units removal:
    17 ->  3
    24  -> 6
    23 ->  9
The units that will be removed are 17, 23, 24.
"""

# Operation of addition.
load_net.hidden.weight[2] += load_net.hidden.weight[16]
load_net.hidden.weight[5] += load_net.hidden.weight[23]
load_net.hidden.weight[8] += load_net.hidden.weight[22]

# Slicing the remained weight values and bias values in a new-sized network.
new_net = Net(11, 27, 3)
new_net.hidden.weight[: 16] = load_net.hidden.weight[: 16]        
new_net.hidden.weight[16: 21] = load_net.hidden.weight[17: 22]        
new_net.hidden.weight[21: ] = load_net.hidden.weight[24: ]       

new_net.hidden.bias[: 16] = load_net.hidden.bias[0: 16]        
new_net.hidden.bias[16: 21] = load_net.hidden.bias[17: 22]        
new_net.hidden.bias[21: ] = load_net.hidden.bias[24: ]       

new_net.output.weight[:, : 16] = load_net.output.weight[:, 0: 16]        
new_net.output.weight[:, 16: 21] = load_net.output.weight[:, 17: 22]       
new_net.output.weight[:, 21: ] = load_net.output.weight[:, 24: ]       

new_net.output.bias[:] = load_net.output.bias[:]
new_net.eval()

# Reload the test dateset and evaluate the shrinked network.
x_test, y_test = loadDataset()
acc, pred = test_model(new_net, x_test, y_test)

mat = confusion(x_test.size(0), 3, pred, y_test)
print("Confusion Matrix (after pruning)：")
print(mat)
F1_score(mat)

