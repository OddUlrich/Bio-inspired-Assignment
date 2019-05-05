# Bio-inspired-NN

## Overview
<p> The code files implement the reduction techniques for an artificial neural network based on the similarity of hidden units. 
There are totally 3 executive files with "main" in the file name, which implement different individual function. 
Other files provide necessary classes and methods for being called.

## Code Structure
<p> Code in this directory basically contains three categories: pre-processing, model training and network reducing.

### 1. Data processing
<p> The first step is to pre-process data set by removing irrelevant data from the data set 
and normalizing all features values into the same range. 
Removal can be done by manual operation in excel files.
Normalization is conducted by the method min-max-norm() in data_loader.py.
This part only executes for once at the beginning, so the caller of this method is commented in main.py.
<p> The next step is to load data from excel file and format them into proper data structure.
In data_loader.py, the code implements the following tasks:

   - Remove the title of columns and rows.
   - Shuffle the data into random order.
   - Select the features that are filtered by selection algorithm.
   - Randomly divide data into training set and testing set.
   - Transform data into Tensor form, and then return them,
   
### 2. Model training
<p> In this part, we create a three layers neural network.
All functionalities and structures are defined in Net.py.
<p> During the process of training and predicting, it is required to evaluate or visualize the performance of model,
so the mothods for confusion matrix, precision, recall and F1 score are included in utils.py.
Others for storing important parameters are also implemented in this file.

### 3. Network reducing
<p> Since the procedure of network reduction is limited with research environment, 
so the implement is basically dependent on manual decision and hard code programming.
The code for reduce network for genre class and subjective rating (Depressing -> Neural -> Exciting) class 
is respectively in main_reduction_genre.py and main_reduction_subjective_rating.py.
