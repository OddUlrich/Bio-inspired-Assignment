# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

from data_loader import data_loader
from RNN_model import RNN_model, train_model, test_model
from utils import confusion, F1_score


input_dim = 1
hidden_dim = 20
layer_dim = 1
output_dim = 3  # Four kinds of genres within 12 songs.

epochs = 100
lr = 0.001

# Return size: [batch_size, length]
input_train, label_train, input_test, label_test = data_loader()

rnn = RNN_model(input_dim, hidden_dim, layer_dim, output_dim)

flat_input_train = input_train.unsqueeze(-1)
train_model(rnn, flat_input_train, label_train, lr=lr, epochs=epochs)

flat_input_test = input_test.unsqueeze(-1)
acc, pred = test_model(rnn, flat_input_test, label_test)

#if accuracy > 40:
#    saveParas(net, X_test, hidden_num+1)
#    torch.save(net.state_dict(), 'net_model.pt')
#    saveDataset(X_test, Y_test)

mat = confusion(input_test.size(0), output_dim, pred, label_test)
print("Confusion Matrix：")
print(mat)
F1_score(mat)
