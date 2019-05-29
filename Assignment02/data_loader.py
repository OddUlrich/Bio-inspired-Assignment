# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import pandas as pd


def data_loader(file_dir):
    # Using a dictionary with song number as a key to store the sequences.
    data_seq_dic = {}

    # Song number is from 1 to 12.
    for song_num in range(1, 13):
        print(song_num)
        # Loading dataset.
        file_path = file_dir + '/S' + str(song_num) + '.xlsx'
        raw_data = pd.read_excel(file_path, header=None)
        
        # Drop the first column as it is references to the data.
        raw_data.drop(raw_data.columns[0], axis=1, inplace=True)
    
        # Remove the headings of features.
        data_seq = raw_data[1:]    
        
        # Insert the sequences with its song number as a key.
        data_seq_dic[song_num] = data_seq
        
    return data_seq_dic
