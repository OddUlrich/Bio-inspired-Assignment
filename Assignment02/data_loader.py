# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import pandas as pd
import numpy as np
import torch


def data_loader(file_dir, sample_num, max_seq_len, target_num):
    # Using a dictionary with song number as a key to store the sequences.
    data_seq_dic = {}

    seqs = np.zeros((sample_num, max_seq_len))
    labels = np.zeros((sample_num, target_num))

    group_size = 16
    # Song number is from 0 to 11.
    for song_num in range(12):
        # Loading dataset.
        file_path = file_dir + '/S' + str(song_num+1) + '.xlsx'
        raw_data = pd.read_excel(file_path, header=None)
        
        # Drop the first column as it is references to the data.
        raw_data.drop(raw_data.columns[0], axis=1, inplace=True)
    
        # Remove the headings of features.
        data_seq = raw_data[1:]    
        
        # Insert the sequences with its song number as a key.
        data_seq_dic[song_num+1] = data_seq.T.to_numpy()
        
        seqs[group_size*(song_num): group_size*(song_num+1)] = data_seq.T.to_numpy()
        labels[group_size*(song_num): group_size*(song_num+1), song_num] = 1
        
    for key, value in data_seq_dic.items():
         print(key, ': ', value.shape)  
         
    return seqs, labels
