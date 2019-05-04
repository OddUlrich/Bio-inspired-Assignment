# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
from Net import Net
import numpy as np
import pandas as pd

load_net = Net(11, 30, 3)
load_net.load_state_dict(torch.load('net_model.pt'))
load_net.eval()


#print(load_net.hidden.weight.data[-1])
#print(load_net.hidden.weight.data[-2])
#print(load_net.hidden.weight.data[-3])
#print(load_net.hidden.bias.shape)
#
#
#new_net = Net(11, 29, 3)
#new_net.hidden.weight[:-1] = load_net.hidden.weight[:-2]
#new_net.hidden.weight[-1] = load_net.hidden.weight[-2] + load_net.hidden.weight[-1]
#
#print(new_net.hidden.weight.data[-1])
#print(new_net.hidden.weight.data[-2])
#print(new_net.hidden.bias.shape)
#
#


    


