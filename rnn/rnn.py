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
y_train = []
for i in range(training_set.shape[0]-60):
    X_train.append(training_set_scaled[i:i+60, 0])
    y_train.append(training_set_scaled[i+60, 0])
    
# reshaping the train data
X_train, y_train = np.array(X_train), np.array(y_train)

X_train_reshaped = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))    


#building the rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout




#making the predictions and visualising the results



