# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import pandas as pd
import numpy as np

def data_loader(file_dir):
    # Using a dictionary with song number as a key to store the sequences.
    data_seq_dic = {}

    seqs = []
    labels = []

    # Song number is from 1 to 12.
    for song_num in range(1, 13):
        # Loading dataset.
        file_path = file_dir + '/S' + str(song_num) + '.xlsx'
        raw_data = pd.read_excel(file_path, header=None)
        
        # Drop the first column as it is references to the data.
        raw_data.drop(raw_data.columns[0], axis=1, inplace=True)
    
        # Remove the headings of features.
        data_seq = raw_data[1:]    
        
        # Insert the sequences with its song number as a key.
        data_seq_dic[song_num] = data_seq.T
        
        seqs.append(data_seq.T)
        
        label = np.zeros((4))
        label[int(song_num / 4)] = 1
        labels.append(label)
        
    for key, value in data_seq_dic.items():
         print(key, ': ', value.shape)  
         
    return seqs, labels
