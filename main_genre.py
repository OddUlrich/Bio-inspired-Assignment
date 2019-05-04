# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
from data_loader import load_data, min_max_norm
from Net import Net, train_model, test_model
from utils import confusion, F1_score, saveParas, saveDataset


############################################
# Hyper parameters
all_features_num = 14
hidden_num = 30
classes_num = 3
epochs_num = 1000
learning_rate = 0.01

genre_loc = 14 # Ignore the index column
disturbing_conforting_loc = 15
depressing_exiting_loc = 16

# Data parameters
all_features_list = list(range(1, all_features_num+1))
GA_features_list = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 14]
SD_MRMR_features_list = [1, 2, 3, 5, 6, 7, 9, 10, 12, 14]

SFS_features_list = [3, 4, 7, 10, 11]
SFFS_features_list = [3, 4, 10, 11]

############################################
selector = GA_features_list
features_num = len(selector)

label_loc = genre_loc


# Normalizing all features values into the range of 0 to 1.
#
#raw_data = pd.read_excel('music-features.xlsx', header=None)
#raw_data.drop(raw_data.columns[0], axis=1, inplace=True)
#raw_data.drop(0, axis=0, inplace=True)
#
#normalized = min_max_norm(raw_data, 14)
#normalized.to_excel('music-features-processed.xlsx')


X_train, Y_train, X_test, Y_test = load_data('music-features-processed.xlsx', 
                                             features_num, 
                                             label_loc,
                                             features_selector = selector,
                                             spliting_ratio=0.8)

net = Net(features_num, hidden_num, classes_num)
train_model(net, X_train, Y_train, lr=learning_rate, epochs=epochs_num)

accuracy, Y_pred = test_model(net, X_test, Y_test)
if accuracy > 40:
    saveParas(net, X_test, hidden_num+1)
    torch.save(net.state_dict(), 'net_model.pt')
    saveDataset(X_train, Y_train, X_test, Y_test)

mat = confusion(X_test.size(0), classes_num, Y_pred, Y_test)
print("Confusion Matrix：")
print(mat)
F1_score(mat)


print("\n========================== END ==================================")



