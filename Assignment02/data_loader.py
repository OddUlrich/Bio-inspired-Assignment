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
    2. Transform data features with min-max normalization into range of 0 to 1.
    3. Select the features that are filtered by selection algorithm.
    4. Divide data into training set and testing set to keep the same distribution.
    5. Shuffle the data into random order.
    6. Transform data into Tensor form, and then return them,
"""


def load_data(file_dir, features_num, label_index, features_selector=[], spliting_ratio=0.8):
    # Load all data.
    raw_data = pd.read_excel(file_dir, header=None)
    # Drop the first column as it is references to the data.
    raw_data.drop(raw_data.columns[0], axis=1, inplace=True)

    # Remove the headings of features.
    all_data = raw_data[1:]
    data = np.zeros((all_data.shape[0], features_num + 1))
    if features_selector != []:
        # The features_selector is a list of index of the selected features.
        for i in range(features_num):
            data[:, i] = all_data.iloc[:, features_selector[i]-1]
    else:
        data[:, :features_num] = all_data.iloc[:, :features_num]

    data[:, -1] = all_data.iloc[:, label_index]
        
    # Divide data into training set and testing set to keep the same distribution.
    sorted_data = data[data[:,-1].argsort()]
    train_data = np.zeros((144, sorted_data.shape[1]))
    test_data = np.zeros((48, sorted_data.shape[1]))
    for i in range(3):
        train_data[i*48: (i+1)*48, :] = sorted_data[i*64: i*64+48, :]
        test_data[i*16: (i+1)*16, :] = sorted_data[i*64+48: (i+1)*64, :]
        
    # Shuffle data.
    permutation = np.random.permutation(train_data.shape[0])
    train_data = train_data[permutation, :]
    permutation = np.random.permutation(test_data.shape[0])
    test_data = test_data[permutation, :]
    
    train_input = train_data[:, :-1]
    train_output = train_data[:, -1]
    test_input = test_data[:, :-1]
    test_output = test_data[:, -1]

    train_input = train_input.astype(dtype='float32')
    test_input = test_input.astype(dtype='float32')
    train_output = train_output.astype(dtype='int64')
    test_output = test_output.astype(dtype='int64')

    # Create Tensors to hold inputs and outputs.
    X_train = torch.Tensor(train_input).float()
    Y_train = torch.Tensor(train_output - 1).long()
    X_test = torch.Tensor(test_input).float()
    Y_test = torch.Tensor(test_output - 1).long()

    # Form label from a single number in a one-shot vector.
    label_train = torch.zeros(len(Y_train), 3)
    for i in range(len(Y_train)):
        label_train[i][Y_train[i]] = 1

    label_test = torch.zeros(len(Y_test), 3)
    for i in range(len(Y_test)):
        label_test[i][Y_test[i]] = 1
        
    return X_train, label_train, X_test, label_test


def seq_loader():
    ''' Loading training data '''
    # Load all training data.
    raw_train_data = pd.read_excel('Training sequence.xlsx', header=None)

    # Drop the first column as it is references to the time sequence indices.
    raw_train_data.drop(raw_train_data.columns[0], axis=1, inplace=True)

    # Skip the first two columns of song number and genre classes.
    x_train = raw_train_data.T.iloc[:, 2:].sample(frac=1).reset_index(drop=True).to_numpy()
#    x_train = raw_train_data.T.iloc[:, 2:].to_numpy()

#     Create one-hot vector for y_train with the second column of genre classes.
    class_train = raw_train_data.T.iloc[:, 1].to_numpy()
    y_train = np.zeros((class_train.shape[0], 3))
    for i in range(class_train.shape[0]):
        y_train[i][int(class_train[i]) - 1] = 1

    x_train = x_train.astype(dtype='float32')
    y_train = y_train.astype(dtype='int64')
    input_train = torch.Tensor(x_train).float()
    label_train = torch.Tensor(y_train).long()

    ''' Loading testing data '''
    # Load all training data.
    raw_test_data = pd.read_excel('Testing sequence.xlsx', header=None)

    # Drop the first column as it is references to the time sequence indices.
    raw_test_data.drop(raw_test_data.columns[0], axis=1, inplace=True)

    # Skip the first two columns of song number and genre classes.
    x_test = raw_test_data.T.iloc[:, 2:].sample(frac=1).reset_index(drop=True).to_numpy()
#    x_test = raw_test_data.T.iloc[:, 2:].to_numpy()

    # Create one-hot vector for y_train with the second column of genre classes.
    class_test = raw_test_data.T.iloc[:, 1].to_numpy()
    y_test = np.zeros((class_test.shape[0], 3))
    for i in range(class_test.shape[0]):
        y_test[i][int(class_test[i]) - 1] = 1

    # Create Tensors.
    x_test = x_test.astype(dtype='float32')
    y_test = y_test.astype(dtype='int64')
    input_test = torch.Tensor(x_test).float()
    label_test = torch.Tensor(y_test).long()

    return input_train, label_train, input_test, label_test

