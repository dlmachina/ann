#implementing the recurrent neural network

# data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

#feaature scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)



X_train = []
for i in range(training_set.shape[0]-60):
    X_train.append(training_set_scaled[i:i+60])
    
    



#building the rnn






#making the predictions and visualising the results



