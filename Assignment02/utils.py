# -*- coding: utf-8 -*-
"""
@author: Ulrich
"""

import torch
import numpy as np
import pandas as pd
import xlwt 

# Define a function to plot confusion matrix
def confusion(sample_num, labels_num, label_pred, label):
    matrix = torch.zeros(labels_num, labels_num)
    _, target = torch.max(label, 1)
    for i in range(sample_num):
        actual_label = target[i]
        predicted_label = label_pred[i]

        matrix[actual_label][predicted_label] += 1

    return matrix

# Calculate precision, recall and F1 score with an input of 3*3 confusion matrix.
def F1_score(m):
    TP, FP, FN = [], [], []
    precision, recall, F1 = [], [], []
    m = m.data.numpy()
    
    TP.append(m[0][0])
    TP.append(m[0][1])
    TP.append(m[0][2])
    
    FP.append(m[1][0] + m[2][0])
    FP.append(m[1][1] + m[2][1])
    FP.append(m[1][2] + m[2][2])
    
    FN.append(m[0][1] + m[0][2])
    FN.append(m[1][0] + m[1][2])
    FN.append(m[2][0] + m[2][1])
    
    for i in range(3):    
        if TP[i] + FP[i] == 0:
            precision.append(0)
        else:
            precision.append(TP[i] / (TP[i] + FP[i]))
        if TP[i] + FN[i] == 0:
            recall.append(0)
        else:
            recall.append(TP[i] / (TP[i] + FN[i]))
        if precision[i] + recall[i] == 0:
            F1.append(0)
        else:            
            F1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
        
    print("\nPrecision: {} ; \nAverage precision: {}".format(precision, np.mean(precision)))
    print("\nRecall: {} ; \nAverage recall: {}".format(recall, np.mean(recall)))
    print("\nF1 Score: {} ; \nAverage F1 Score: {}".format(F1, np.mean(F1)))
    
# Write matrix data into an excel file.
# The cells with value less than 15 will be filled with blue for finding the similarity.
def saveExcel(data, path, sheet_num, bVector=False):
    file = xlwt.Workbook()
    sheet1 = file.add_sheet(sheet_num, cell_overwrite_ok=True)
    
    # Color of cells.
    st = xlwt.easyxf('pattern: pattern solid;')
    st.pattern.pattern_fore_colour = 4

    sd = xlwt.easyxf('pattern: pattern solid;')
    sd.pattern.pattern_fore_colour = 8

    row = 0

    [h, l] = data.shape
    for i in range(h):
        for j in range(l):
            if bVector:
                if i != 0 and j != 0 and i < j:
                    if (data[i, j] < 10) or (data[i, j] > 170):
                        sheet1.write(row, 0, i)
                        sheet1.write(row, 1, j)
                        sheet1.write(row, 2, data[i, j].item())
                        row += 1
            else:
                sheet1.write(i, j, data[i, j].item())
    file.save(path)


# The input vectors should be numpy vector form.
def vector_angle(vec1, vec2):
    list1 = vec1.data.numpy()
    list2 = vec2.data.numpy()

    val = np.dot(list1, list2) / np.linalg.norm(list1) / np.linalg.norm(list2)
    radian = np.arccos(val)  # 0 - PI
    
    angle = radian * 180 / np.pi
    angle = np.nan_to_num(angle)

    return angle


# Save relevant parameter for further observation on reduce the neural network.
def saveNNParas(model, input_data, vec_size):
    a_hidden = model.getActivationVec(input_data)
#    saveExcel(a_hidden[:, :] - 0.5, 'activation.xls', u'sheet1')
    
    vectorangle_mat = np.zeros((vec_size, vec_size))
    for i in range(vec_size):
        for j in  range(vec_size):
            if i == j:
                vectorangle_mat[i][j] = 0
            else:
                vectorangle_mat[i][j] = vector_angle(a_hidden[:, i], a_hidden[:, j])
    saveExcel(vectorangle_mat, 'vector_angle.xls', u'sheet1', bVector=True)
    
    print("Vector angles has been successfully saved in Excel files!\n")
    
# Save test data for validating the reduced network.
def saveDataset(x, y, sub_title):
    features = pd.DataFrame(x.data.numpy())
    features.to_excel(sub_title + '_features.xlsx', index=False)
        
    classes = pd.DataFrame(y.data.numpy())
    classes.to_excel(sub_title + '_class.xlsx', index=False)
    
# Load test data for validating the reduced network.
def loadDataset(sub_title):
    x = pd.read_excel(sub_title + '_features.xlsx', header=None)
    y = pd.read_excel(sub_title + '_class.xlsx', header=None)

    x.drop(0, axis=0, inplace=True)
    y.drop(0, axis=0, inplace=True)

    # Create Tensors to hold inputs and outputs.
    x = torch.Tensor(x.to_numpy()).float()
    y = torch.Tensor(y.to_numpy()).long()
    
    return x, y