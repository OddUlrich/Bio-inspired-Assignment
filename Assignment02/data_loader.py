# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import numpy as np
import pandas as pd
import torch


def data_loader():
    ''' Loading training data '''
    # Load all training data.
    raw_train_data = pd.read_excel('Training data.xlsx', header=None)
    
    # Drop the first column as it is references to the time sequence indices.
    raw_train_data.drop(raw_train_data.columns[0], axis=1, inplace=True)
    
#    print(raw_train_data.iloc[0:5, 0:6])

    # Skip the first two columns of song number and genre classes.
    x_train = raw_train_data.T.iloc[:, 2:].sample(frac=1).reset_index(drop=True).to_numpy()
#    x_train = raw_train_data.T.iloc[:, 2:].to_numpy()

    # Create one-hot vector for y_train with the second column of genre classes.
    class_train = raw_train_data.T.iloc[:, 1].to_numpy()
    y_train = np.zeros((class_train.shape[0], 3))
    for i in range(class_train.shape[0]):
        y_train[i][int(class_train[i])-1] = 1

    x_train = x_train.astype(dtype='float32')
    y_train = y_train.astype(dtype='int64')
    input_train = torch.Tensor(x_train).float()
    label_train = torch.Tensor(y_train).long()
    
#    print(input_train.shape)
#    print(label_train.shape)
#    print(input_train[0:5, 0:6])
#    print(label_train[0:5])
    
    ''' Loading testing data '''
    # Load all training data.
    raw_test_data = pd.read_excel('Testing data.xlsx', header=None)
    
    # Drop the first column as it is references to the time sequence indices.
    raw_test_data.drop(raw_test_data.columns[0], axis=1, inplace=True)
    
#    print(raw_test_data.iloc[0:5, 0:6])

    # Skip the first two columns of song number and genre classes.
    x_test = raw_test_data.T.iloc[:, 2:].sample(frac=1).reset_index(drop=True).to_numpy()
#    x_test = raw_test_data.T.iloc[:, 2:].to_numpy()
    
    # Create one-hot vector for y_train with the second column of genre classes.
    class_test = raw_test_data.T.iloc[:, 1].to_numpy()
    y_test = np.zeros((class_test.shape[0], 3))
    for i in range(class_test.shape[0]):
        y_test[i][int(class_test[i])-1] = 1

    # Create Tensors.
    x_test = x_test.astype(dtype='float32')
    y_test = y_test.astype(dtype='int64')
    input_test = torch.Tensor(x_test).float()
    label_test = torch.Tensor(y_test).long()
    
#    print(input_test.shape)
#    print(label_test.shape)
#    print(input_test[0:5, 0:6])
#    print(label_test[0:5])
    
    return input_train, label_train, input_test, label_test
    