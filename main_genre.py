# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

from data_loader import load_data
from Net import Net, train_model, test_model
from utils import confusion, F1_score, saveParas

############################################
# Hyper parameters
all_features_num = 14
hidden_num = 30
classes_num = 3
epochs_num = 500
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


#X_train, Y_train, X_test, Y_test = data_loader.load_data('music-features-processed.xlsx', 
#                                                         features_num, 
#                                                         genre_loc)

X_train, Y_train, X_test, Y_test = load_data('music-features-processed.xlsx', 
                                             features_num, 
                                             label_loc,
                                             features_selector = selector,
                                             spliting_ratio=0.8)

#X_train, Y_train, X_test, Y_test = data_loader.load_data('music-features-processed.xlsx', 
#                                                         features_num, 
#                                                         label_loc,
#                                                         features_selector = selector)


net = Net(features_num, hidden_num, classes_num)
train_model(net, X_train, Y_train, lr=0.02, epochs=epochs_num, loss_bound=0.1)
#train_model(net, X_train, Y_train, lr=0.01, epochs=epochs_num)

accuracy, Y_pred = test_model(net, X_test, Y_test)
#if accuracy > 45:
#    saveParas(net, X_test, hidden_num+1)
#    torch.save(net.state_dict(), 'net_model.pt')

mat = confusion(X_test.size(0), classes_num, Y_pred, Y_test)
print("Confusion Matrix：")
print(mat)
F1_score(mat)

