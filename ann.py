#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 18:25:08 2019

@author: nithin
"""
# data preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,-1]
n = 
#encoding categorical variables 
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
column_trans = ColumnTransformer([("onehot",OneHotEncoder(sparse="False",dtype=np.int),[1]),("ord",OrdinalEncoder(dtype=np.int),[2])],remainder="passthrough")
X_mod = column_trans.fit_transform(X)  

# to take care of the dummy variable trap
X  = X_mod[:,1:]

# splitting to train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#fitting the classifier model to the training dataset
#making  a model of ANN 
#importing the keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN
ann_classifier =  Sequential()

ann_classifier.add(Dense(units = 6,kernel_initializer = "uniform", activation = "relu",input_dim = 11))
ann_classifier.add(Dense(units = 6,kernel_initializer = "uniform", activation = "relu"))
ann_classifier.add(Dense(units=1,kernel_initializer = "uniform", activation = "sigmoid"))

# running the ANN
ann_classifier.compile(optimizer="adam",loss = "binary_crossentropy",metrics=["accuracy"])

ann_classifier.fit(X_train,y_train,batch_size = 10, nb_epoch =100 )
    

y_pred = ann_classifier.predict(X_test) 
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(" The accuracy on test dataset is "+str(np.trace(cm)*100/np.shape(X_test)[0])+ "%" )

# using the model on a new sample x_sample 

 """ Use our ANN model to predict if the customer with the following informations will leave the bank: 

     Geography: France
     Credit Score: 600
     Gender: Male
     Age: 40 years old
     Tenure: 3 years
     Balance: $60000
     Number of Products: 2
     Does this customer have a credit card ? Yes
     Is this customer an Active Member: Yes
     Estimated Salary: $50000
"""

x_sample = np.array([ [0,0,1,600,40,3,60000,2,1,1,50000] ])
x_sample_std = sc_X.transform(x_sample)
y_sample = ann_classifier.predict(x_sample) > 0.5

                    
