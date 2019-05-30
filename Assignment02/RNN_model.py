# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

class RNN_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out, hn
    

def train_model(model, input, label, lr=0.01, epochs=500):  
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_losses = []
    
    for epoch in range(1, epochs+1):
        output, hidden = model(input)
        
        loss = criterion(output, torch.max(label, 1)[1])
        all_losses.append(loss)
        
        _, prediction = torch.max(output, 1)
        _, target = torch.max(label, 1)
        
        correct = sum(prediction.data.numpy() == target.data.numpy())
        total = len(prediction)
        
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()), end=' ')
            print("Accuracy: {:.2f}".format(100*correct/total))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    plt.figure()
    plt.plot(all_losses)
    plt.show()
    
    
def test_model(model, input, label):  
    output, hidden = model(input)
    
    _, prediction = torch.max(output, 1)
    _, target = torch.max(label, 1)
    
    correct = sum(prediction.data.numpy() == target.data.numpy())
    total = len(prediction)
    accuracy = 100*correct/total
    print('Testing Accuracy: %.2f %%' % (accuracy))
 
    return accuracy, prediction

    