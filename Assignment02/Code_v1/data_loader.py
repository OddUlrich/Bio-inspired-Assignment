# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import pandas as pd
import numpy as np
import torch

# Min-max normalization into the range of 0 to 1.
def min_max_norm(data, features_num):
    for i in range(features_num):
        vec = data.iloc[:, i]
        delta = vec - vec.min()
        new_range = vec.max() -  vec.min()
        data.iloc[:, i] = delta / new_range
    return data

"""
Loading data from excel file.
    1. Remove the title of columns and rows.
    2. Shuffle the data into random order.
    3. Transform data features with min-max normalization into range of 0 to 1.
    4. Select the features that are filtered by selection algorithm.
    5. Randomly divide data into training set and testing set.
    6. Transform data into Tensor form, and then return them,
"""
def load_data(file_dir, features_num, label_index, features_selector=[], spliting_ratio=0.8):
    # Load all data.
    raw_data = pd.read_excel(file_dir, header=None)
    # Drop the first column as it is references to the data.
    raw_data.drop(raw_data.columns[0], axis=1, inplace=True)
    
    # Remove the headings of features.
    raw_data = raw_data[1:]

    data = np.zeros((raw_data.shape[0], features_num + 1))
    
    ## FIXME: Check the index for features selection.
    
    if features_selector != []:
        # The features_selector is a list of index of the selected features.
        for i in range(len(features_selector)):
            data[:, i-1] = raw_data.iloc[:, features_selector[i-1]]
    else:
        data[:, :features_num] = raw_data.iloc[:, :features_num]
    
    data[:, -1] = raw_data.iloc[:, label_index]
       
    print(data)

    genre_1 = []
    genre_2 = []
    genre_3 = []
    
    for row in data:
        print(row)
        print(row.shape)
        break
    
#    
#    
#    label_set = set(data[:, -1])
#    print(label_set)
#    
#    
#    # Randomly split data into training set and testing set.
#    mask = np.random.rand(len(data)) < spliting_ratio
#    train_data = data[mask]
#    test_data = data[~mask]
#            
#    train_input = train_data[:, :-1]
#    train_output = train_data[:, -1]
#    test_input = test_data[:, :-1]
#    test_output = test_data[:, -1]
#
#    train_input = train_input.astype(dtype='float32')
#    test_input = test_input.astype(dtype='float32')
#    train_output = train_output.astype(dtype='int64')
#    test_output = test_output.astype(dtype='int64')
#
#    # Create Tensors to hold inputs and outputs.
#    X_train = torch.Tensor(train_input).float()
#    Y_train = torch.Tensor(train_output - 1).long()
#    X_test = torch.Tensor(test_input).float()
#    Y_test = torch.Tensor(test_output - 1).long()
#
#    # Form label from a single number in a one-shot vector.
#    label_train = torch.zeros(len(Y_train), 3)
#    for i in range(len(Y_train)):
#        label_train[i][Y_train[i]] = 1
#
#    label_test = torch.zeros(len(Y_test), 3)
#    for i in range(len(Y_test)):
#        label_test[i][Y_test[i]] = 1
#        
#    return X_train, label_train, X_test, label_test
    return None, None, None, None
