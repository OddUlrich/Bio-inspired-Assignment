# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""


from Net import Net
from RNN_model import RNN_model

def reduced_ann_net(old_net, unit_in, unit_remove, new_hidden_num):
    old_net.hidden.weight[unit_in] += old_net.hidden.weight[unit_remove]
    old_net.hidden.bias[unit_in] += old_net.hidden.bias[unit_remove]

    # Slicing the remained weight values and bias values in a new-sized network.
    new_net = Net(11, new_hidden_num, 3)
    new_net.hidden.weight[: unit_remove] = old_net.hidden.weight[: unit_remove]
    new_net.hidden.weight[unit_remove:] = old_net.hidden.weight[unit_remove + 1:]

    new_net.hidden.bias[: unit_remove] = old_net.hidden.bias[0: unit_remove]
    new_net.hidden.bias[unit_remove:] = old_net.hidden.bias[unit_remove + 1:]

    new_net.output.weight[:, : unit_remove] = old_net.output.weight[:, 0: unit_remove]
    new_net.output.weight[:, unit_remove:] = old_net.output.weight[:, unit_remove + 1:]

    new_net.output.bias[:] = old_net.output.bias[:]
    new_net.eval()

    return new_net

def reduced_rnn_net(old_rnn, unit_a, unit_b, new_hidden_dim):
    new_rnn = RNN_model(1, new_hidden_dim, 1, 3)
    
    # Slicing the remained weight values and bias values in a new-sized network.
    new_rnn.rnn.weight_ih_l0[: unit_a] = old_rnn.rnn.weight_ih_l0[: unit_a]
    new_rnn.rnn.weight_ih_l0[unit_a: unit_b-1] = old_rnn.rnn.weight_ih_l0[unit_a+1: unit_b]
    new_rnn.rnn.weight_ih_l0[unit_b-1: ] = old_rnn.rnn.weight_ih_l0[unit_b+1: ]
    
    new_rnn.rnn.weight_hh_l0[: unit_a, : unit_a] = old_rnn.rnn.weight_hh_l0[: unit_a, : unit_a]
    new_rnn.rnn.weight_hh_l0[unit_a: unit_b-1, unit_a: unit_b-1] = old_rnn.rnn.weight_hh_l0[unit_a+1: unit_b, unit_a+1: unit_b]
    new_rnn.rnn.weight_hh_l0[unit_b-1: , unit_b-1: ] = old_rnn.rnn.weight_hh_l0[unit_b+1: , unit_b+1: ]
    
    new_rnn.rnn.bias_ih_l0[: unit_a] = old_rnn.rnn.bias_ih_l0[: unit_a]
    new_rnn.rnn.bias_ih_l0[unit_a: unit_b-1] = old_rnn.rnn.bias_ih_l0[unit_a+1: unit_b]
    new_rnn.rnn.bias_ih_l0[unit_b-1: ] = old_rnn.rnn.bias_ih_l0[unit_b+1: ]
    
    new_rnn.rnn.bias_hh_l0[: unit_a] = old_rnn.rnn.bias_hh_l0[: unit_a]
    new_rnn.rnn.bias_hh_l0[unit_a: unit_b-1] = old_rnn.rnn.bias_hh_l0[unit_a+1: unit_b]
    new_rnn.rnn.bias_hh_l0[unit_b-1: ] = old_rnn.rnn.bias_hh_l0[unit_b+1: ]
    
    new_rnn.fc.weight[: , : unit_a] = old_rnn.fc.weight[: , : unit_a]
    new_rnn.fc.weight[: , unit_a: unit_b-1] = old_rnn.fc.weight[: , unit_a+1: unit_b]
    new_rnn.fc.weight[: , unit_b-1: ] = old_rnn.fc.weight[: , unit_b+1: ]
    
    new_rnn.fc.bias[:] = old_rnn.fc.bias[:]
    
    return new_rnn

