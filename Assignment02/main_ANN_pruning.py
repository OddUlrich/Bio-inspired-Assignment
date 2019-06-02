# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
import pandas as pd
from reducing_net import reduced_ann_net
from Net import Net, test_model
from utils import confusion, F1_score, loadDataset, saveNNParas
import time

# Loading the previous network status.
feature_num = 11
hidden_num = 30
output_num = 3

load_net = Net(feature_num, hidden_num, output_num)
load_net.load_state_dict(torch.load('ann_net_model_genre.pt'))
#load_net.load_state_dict(torch.load('net_model_subjective_rating.pt'))
load_net.eval()

# Loading testing dataset to evaluate new network.
x_test, y_test = loadDataset('testing')

# Loading the information of vector.
vectors = pd.read_excel('ann_vector_angle_sample.xls', header=None)
raw_df = pd.DataFrame({'row': vectors.iloc[:, 0],
                       'col': vectors.iloc[:, 1],
                       'vector': vectors.iloc[:, 2]})

# Sorting by the values of vector angle in ascending order.
increase_res = raw_df.sort_values('vector', ascending=True)
unique_row = increase_res.row.unique()
unique_col = increase_res.col.unique()

# Initialize all the status parameters.
nets = []
times = []
old_net = load_net
cnt = 0
hidden_num = 30

# To maintain normal performance of the network, we only remove less than 10 hidden units.
while cnt < 10:
    # No simple similar vector pair.
    if len(unique_row) == 0 or len(unique_col) == 0:
        print("\n Finished! \n")
        break
        
    for index, row in increase_res.iterrows():
        if row['row'] in unique_row and row['col'] in unique_col and row['row'] not in unique_col:
            hidden_num -= 1
            new_net = reduced_ann_net(old_net, int(row['row']), int(row['col']), hidden_num)

            print("\n======= Net hidden size: {}==========\n".format(hidden_num))

            start_time = time.time()
            acc, pred = test_model(new_net, x_test, y_test)
            stop_time = time.time()
            print("Execution time: %s ms" % ((stop_time - start_time)*1000))
            times.append((stop_time - start_time)*1000)

            mat = confusion(x_test.size(0), 3, pred, y_test)
            F1_score(mat)

            # Save the new network and evaluate its vector angle.
            nets.append(new_net)
            old_net = new_net

            saveNNParas(new_net, x_test, hidden_num)
            vectors = pd.read_excel('vector_angle.xls', header=None)
            if (vectors.empty):
                cnt = 10
                print("\n Finished: Vectors are empty! \n")
                break

            df = pd.DataFrame({'row': vectors.iloc[:, 0],
                               'col': vectors.iloc[:, 1],
                               'vector': vectors.iloc[:, 2]})

            increase_res = df.sort_values('vector', ascending=True)

            unique_row = increase_res.row.unique()
            unique_col = increase_res.col.unique()
            
            cnt += 1
            break
            