# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:21:27 2019

@author: Ulrich
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

############################################
# Hyper parameters
features_num = 14
hidden_num = 30  # TODO: Design a function to choose the optimum size of hidden layer.
classes_num = 3
epochs_num = 500
learning_rate = 0.01

# Data parameters
genre_loc = 14 # Ignore the index column
############################################

"""
Step 1: Load data and pre-process data.
"""
# Load all data
data = pd.read_excel('music-features.xlsx', header=None)
# Drop the first column as it is references to the participants.
data.drop(data.columns[0], axis=1, inplace=True)

# Print out the first 10 rows of the data.
# print(data[:10])

# Shuffle data
data = data[1:].sample(frac=1).reset_index(drop=True)

# Randomly split data into training set (80%) and testing set (20%).
mask = np.random.rand(len(data)) < 0.8
train_data = data[mask]
test_data = data[~mask]

# Split trainning data and testing data into input and output
# The first 14 columns are features, the 15th column is target genre.
train_input = train_data.iloc[:, :features_num]
train_output = train_data.iloc[:, genre_loc]
test_input = test_data.iloc[:, :features_num]
test_output = test_data.iloc[:, genre_loc]

train_input = train_input.astype(dtype='float32')
test_input = test_input.astype(dtype='float32')
train_output = train_output.astype(dtype='int64')
test_output = test_output.astype(dtype='int64')

# TODO: Convert the first 13 features from time domain into frequent domain as that of the 14th feature.

# Create Tensors to hold inputs and outputs.
X = torch.Tensor(train_input.values).float()
Y = torch.Tensor(train_output.values - 1).long()  # Target values 0-2.
X_test = torch.Tensor(test_input.values).float()
Y_test = torch.Tensor(test_output.values - 1).long()  # Target values 0-2.

"""
Step 2: Define a neural network

To implement a bidirectional neural network, here we build two model.
- Forward direction NN
    input layer: 14 neurons, representing the 14 features.
    hidden layer: 30 neurons temparory, using ReLU as activation function.
    output layer: 3 neurons, representing the categories of genres.
- Reverse direction NN
    input layer: 3 neurons, representing the categories of genres.
    hidden layer: 30 neurons temparory, using ReLU as activation function.
    output layer: 14 neurons, representing the 14 features.

TODO: Update the numbers of neurons in hidden layer after getting the optimum number.

Both the network will be trained with Stochastic Gradient Descent (SGD) as an optimizer,
that will hold the current state and will update the parameters based on the computed gradients.

Its performance will be evaluated using cross-entropy.
"""
# Neural network model.
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        z_hidden = self.hidden(input)
        a_hidden = F.relu(z_hidden)
        out_pred = self.out(a_hidden)
        return out_pred
 
# Define a forward direction network.
forward_net = Net(features_num, hidden_num, classes_num)
forward_criterion = torch.nn.CrossEntropyLoss()
forward_optimizer = torch.optim.SGD(forward_net.parameters(), lr=learning_rate)
forward_all_losses = []
# Define a reverse direction network.
reverse_net = Net(classes_num, hidden_num, features_num)
reverse_criterion = torch.nn.MSELoss()
reverse_optimizer = torch.optim.SGD(reverse_net.parameters(), lr=learning_rate)
reverse_all_losses = []

# Transfer the class number into a one-shot matrix.
Y_matrix = torch.zeros([Y.size(0), classes_num], dtype=torch.float32)
for idx in range(Y.size(0)):    
    Y_matrix[idx][Y[idx]] = 1.0
    
# Train the model in both direction.
for epoch in range(epochs_num):
    # Forward propagation.
    output = forward_net(X)
    forward_loss = forward_criterion(output, Y)
    forward_all_losses.append(forward_loss.item())
    
    _, predicted = torch.max(output, 1)
    total = predicted.size(0)
    correct = predicted.data.numpy() == Y.data.numpy()
    
    if epoch % 50 == 0:
        print('Forward direction - epoch [%3d/%3d] loss: %.4f  Accuracy: %.2f %%'
          % (epoch+1, epochs_num, forward_loss.item(), 100 * sum(correct)/total))

    # Back propagation.
    forward_net.zero_grad()
    forward_loss.backward()
    forward_optimizer.step()

    # Switch the layout order of network: hidden -> out, out -> hidden.
    with torch.no_grad():
        reverse_net.hidden.weight.copy_(torch.t(forward_net.out.weight))
        reverse_net.out.weight.copy_(torch.t(forward_net.hidden.weight))

    # Forward propagation.
    output = reverse_net(Y_matrix)
    reverse_loss = reverse_criterion(output, X)
    reverse_all_losses.append(reverse_loss.item())
    
    if epoch % 50 == 0:
       print('Reverse direction - epoch [%3d/%3d] loss: %.4f'
             % (epoch+1, epochs_num, reverse_loss.item()))
    
    # Back propagation.
    reverse_net.zero_grad()
    reverse_loss.backward()
    reverse_optimizer.step()
    
    # Switch the layout order of network: hidden -> out, out -> hidden.
    with torch.no_grad():
        forward_net.hidden.weight.copy_(torch.t(reverse_net.out.weight))
        forward_net.out.weight.copy_(torch.t(reverse_net.hidden.weight))

# Plot the historical loss.
# import matplotlib as plt
#
# plt.figure()
# plt.plot(forward_all_losses)
# plt.show()

"""
Step 3: Test the neural network
"""

# create Tensors to hold inputs and outputs


# test the neural network using testing data
# It is actually performing a forward pass computation of predicted y
# by passing x to the model.
# Here, Y_pred_test contains three columns, where the index of the
# max column indicates the class of the instance
Y_pred_test = forward_net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
