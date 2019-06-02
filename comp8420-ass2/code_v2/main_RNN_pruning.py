# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""


import torch
import pandas as pd
import numpy as np
from RNN_model import RNN_model, test_model
from reducing_net import reduced_rnn_net
from utils import confusion, F1_score, loadDataset, saveNNParas
import time

# Loading the previous network status.
input_dim = 1
hidden_dim = 50
layer_dim = 1
output_dim = 3  # Four kinds of genres within 12 songs.

load_rnn = RNN_model(input_dim, hidden_dim, layer_dim, output_dim)
load_rnn.load_state_dict(torch.load('rnn_model.pt'))
load_rnn.eval()

# Loading testing dataset to evaluate new network.
x_test, y_test = loadDataset('testing_sequence')
flat_input_test = x_test.unsqueeze(-1)

# Various sequence length used for padding sequence and packed sequence in rnn modol.
l = [1104, 1028, 980, 964, 960, 956, 956, 932, 868, 840, 836, 808]
test_seq_lens = np.zeros((4*12))
for i in range(len(l)):
    test_seq_lens[i*4: (i+1)*4] = l[i]   

# Loading the information of vector.
vectors = pd.read_excel('rnn_vector_angle_sample.xls', header=None)
raw_df = pd.DataFrame({'row': vectors.iloc[:, 0],
                       'col': vectors.iloc[:, 1],
                       'angle': vectors.iloc[:, 2]})

# Sorting by the values of vector angle in ascending order.
decrease_res = raw_df.sort_values('angle', ascending=False)
unique_row = decrease_res.row.unique()
unique_col = decrease_res.col.unique()

# Initialize all the status parameters.
rnns = []
times = []
old_rnn = load_rnn
cnt = 0
hidden_num = hidden_dim

# To maintain normal performance of the network, we only remove less than 10 hidden units.
while cnt < 10:
    # No simple similar vector pair.
    if len(unique_row) == 0 or len(unique_col) == 0:
        print("\n Finished! \n")
        break
        
    for index, row in decrease_res.iterrows():
        if row['row'] in unique_row and row['col'] in unique_col and row['row'] not in unique_col:
            hidden_num -= 2
            new_rnn = reduced_rnn_net(old_rnn, int(row['row']), int(row['col']), hidden_num)

            print("\n======= RNN hidden size: {}==========\n".format(hidden_num))

            start_time = time.time()
            # Unsqueeze from 2-dimension to 3-dimension to match the rnn model.
            acc, pred = test_model(new_rnn, flat_input_test, y_test, test_seq_lens)
            stop_time = time.time()
            print("Execution time: %s ms" % ((stop_time - start_time)*1000))
            times.append((stop_time - start_time)*1000)

            mat = confusion(x_test.size(0), 3, pred, y_test)
            F1_score(mat)

            # Save the new network and evaluate its vector angle.
            rnns.append(new_rnn)
            old_rnn = new_rnn

            saveNNParas(new_rnn, x_test, hidden_num)
            vectors = pd.read_excel('vector_angle.xls', header=None)
            if (vectors.empty):
                cnt = 10
                print("\n Finished: Vectors are empty! \n")
                break

            df = pd.DataFrame({'row': vectors.iloc[:, 0],
                               'col': vectors.iloc[:, 1],
                               'angle': vectors.iloc[:, 2]})

            decrease_res = df.sort_values('angle', ascending=False)

            unique_row = decrease_res.row.unique()
            unique_col = decrease_res.col.unique()
            
            cnt += 1
            break
        
print(times)