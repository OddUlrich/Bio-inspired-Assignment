# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

class RNN_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, seq_lens):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # When using padded sequence, do not shuffle the date set in loading data.
#        x = pack_padded_sequence(x, seq_lens, batch_first = True)    
        out, hn = self.rnn(x, h0.detach())
#        out, _ = pad_packed_sequence(out, batch_first = True) 
        
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out, hn
    

def train_model(model, input, label, seq_lens, lr=0.01, epochs=500):  
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_losses = []
    
    for epoch in range(1, epochs+1):
        output, hidden = model(input, seq_lens)
        
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
    
    
def test_model(model, input, label, seq_lens):  
    model.eval()
    output, hidden = model(input, seq_lens)
    
    _, prediction = torch.max(output, 1)
    _, target = torch.max(label, 1)
    
    correct = sum(prediction.data.numpy() == target.data.numpy())
    total = len(prediction)
    accuracy = 100*correct/total
    print('Testing Accuracy: %.2f %%' % (accuracy))
 
    return accuracy, prediction
