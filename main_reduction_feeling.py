## -*- coding: utf-8 -*-
#"""
#@author: Ulrich
#"""
#
## -*- coding: utf-8 -*-
#"""
#@author: Ulrich
#"""
#
#import torch
#from Net import Net, test_model
#from utils import confusion, F1_score, loadDataset
#
## Reload the parameters of the trained model.
#load_net = Net(11, 30, 3)
#load_net.load_state_dict(torch.load('net_model_genre.pt'))
#load_net.eval()
#
#"""
#Scheme for units removal:
#    16 ->  2
#    17  -> 1
#    25 ->  29
#The units that will be removed are 16, 17, 25.
#"""
#
## Operation of addition.
#load_net.hidden.weight[0] += load_net.hidden.weight[16]
#load_net.hidden.weight[1] += load_net.hidden.weight[15]
#load_net.hidden.weight[28] += load_net.hidden.weight[24]
#
## Slicing the remained weight values and bias values in a new-sized network.
#new_net = Net(11, 27, 3)
#new_net.hidden.weight[: 15] = load_net.hidden.weight[: 15]        
#new_net.hidden.weight[15: 22] = load_net.hidden.weight[17: 24]        
#new_net.hidden.weight[22: ] = load_net.hidden.weight[25: ]       
#
#new_net.hidden.bias[: 15] = load_net.hidden.bias[0: 15]        
#new_net.hidden.bias[15: 22] = load_net.hidden.bias[17: 24]        
#new_net.hidden.bias[22: ] = load_net.hidden.bias[25: ]       
#
#new_net.output.weight[:, : 15] = load_net.output.weight[:, 0: 15]        
#new_net.output.weight[:, 15: 22] = load_net.output.weight[:, 17: 24]       
#new_net.output.weight[:, 22: ] = load_net.output.weight[:, 25: ]       
#
#new_net.output.bias[:] = load_net.output.bias[:]
#new_net.eval()
#
## Reload the test dateset and evaluate the shrinked network.
#x_test, y_test = loadDataset()
#acc, pred = test_model(new_net, x_test, y_test)
#
#mat = confusion(x_test.size(0), 3, pred, y_test)
#print("Confusion Matrix (after pruning)：")
#print(mat)
#F1_score(mat)
#
