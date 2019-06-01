# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

from Net import Net


def reduced_ann_net(old_net, unit_in, unit_remove, new_hidden_num):
    old_net.hidden.weight[unit_in] += old_net.hidden.weight[unit_remove]

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

