



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:16:43 2020

@author: nithin
"""
import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



# reading dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#encoding categorical variables 
column_trans = ColumnTransformer([("onehot",OneHotEncoder(sparse="False",dtype=np.int),[1]),("ord",OrdinalEncoder(dtype=np.int),[2])],remainder="passthrough")
X_mod = column_trans.fit_transform(X)  
# to take care of the dummy variable trap
X  = X_mod[:,1:]

# splitting to train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def build_classifier(optimizer):
    ann_classifier =  Sequential() 
    ann_classifier.add(Dense(units = 6,kernel_initializer = "uniform", activation = "relu",input_dim = 11))
    ann_classifier.add(Dense(units = 6,kernel_initializer = "uniform", activation = "relu"))
    ann_classifier.add(Dense(units=1,kernel_initializer = "uniform", activation = "sigmoid"))
    ann_classifier.compile(optimizer= optimizer,loss = "binary_crossentropy",metrics=["accuracy"])
    return ann_classifier
classifier = KerasClassifier(build_fn = build_classifier)  
parameters = {"batch_size":[25,32],"epochs":[100,500],"optimizer":["adam","rmsprop"]}
grid_search = GridSearchCV(estimator=classifier,param_grid = parameters,scoring ="accuracy",cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_