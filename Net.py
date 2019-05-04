# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
import matplotlib.pyplot as plt

# Build a neural network structure.
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim = 1)
        
    def forward(self, input):
        z_hidden = self.hidden(input)
        a_hidden = torch.sigmoid(z_hidden)
        out = self.output(a_hidden)
        out = self.softmax(out)
        return out
    
    def getActivationVec(self, input):
        z_hidden = self.hidden(input)
        a_hidden = torch.sigmoid(z_hidden)
        return a_hidden

def train_model(model, input, label, lr=0.01, epochs=500, loss_bound=None):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_losses = []
    
    if loss_bound == None:
        for epoch in range(epochs):
            out = model(input)
            loss = criterion(out, label)
            all_losses.append(loss.item())
        
            _, prediction = torch.max(out, 1)
            _, target = torch.max(label, 1)
            
            correct = sum(prediction.data.numpy() == target.data.numpy())
            total = len(prediction)

            if epoch % 100 == 0:
                print('epoch [%3d/%3d] loss: %.4f Accuracy: %.2f' 
                      % (epoch + 1, epochs, loss.item(), 100*correct/total))
    
            model.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        epoch = 0
        while True: 
            epoch += 1
            
            out = model(input)
            loss = criterion(out, label)
            all_losses.append(loss.item())
            
            _, prediction = torch.max(out, 1)
            _, target = torch.max(label, 1)
            
            correct = sum(prediction.data.numpy() == target.data.numpy())
            total = len(prediction)

            if epoch % 100 == 0:
                print('epoch %3d loss: %.4f Accuracy: %.2f' 
                      % (epoch + 1, loss.item(), 100*correct/total))
                if (loss <= loss_bound):
                    break
    
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
    plt.figure()
    plt.plot(all_losses)
    plt.show()

def test_model(model, input, label):
    out_test = model(input)
      
    _, label_pred = torch.max(out_test, 1)
    _, target = torch.max(label, 1)            
    correct = sum(label_pred.data.numpy() == target.data.numpy())
    total = len(label_pred)
    accuracy = 100*correct/total
    print('Testing Accuracy: %.2f %%' % (accuracy))
 
    return accuracy, label_pred
