# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:34:11 2023

@author: 26526
"""

import numpy as np
import os, sys, random, glob
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from data_test1 import pre_data_for_svdata

def read_data(path):
    data_x = []
    data_y = []
    data_list = []
    
    data_list.extend(glob.glob(os.path.join(path, "*.npy"))) 
    print(data_list)
    for i in range(len(data_list)):
        data_tmp = np.load(data_list[i])
        data_x.append(data_tmp[:-1])
        data_y.append(data_tmp[-1])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    return data_x, data_y
  
    
def pre_data():
    p0 = "..//Deep_model//datasets//s_data//train"
    p1 = "..//Deep_model//datasets//s_data//validation"
    
    X_Train, Y_Train = read_data(p0)
    
    print(X_Train)
    print(Y_Train)
    
    X_Test, Y_Test = read_data(p1)
    
    print("The shape of training datasets is:", X_Train.shape)
    print("The shape of test datasets is:", Y_Train.shape)
    
    return X_Train, Y_Train, X_Test, Y_Test

def Linear_train():
    #model fitting
    X_Train, Y_Train, X_Test, Y_Test = pre_data_for_svdata()
    lr = LinearRegression()
    lr.fit(X_Train, Y_Train)
    
    print("lr.coef_ is:", lr.coef_)
    print("lr.intercept is =:", lr.intercept_)
    
    #model evaluation
    Y_Pred = lr.predict(X_Test)
    MSE = metrics.mean_squared_error(Y_Test, Y_Pred)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_Test, Y_Pred))
    
    print("MSE is:", MSE)
    print("RMSE is:", RMSE)
    
    plt.figure(figsize=(16,7))
    plt.subplot(121)
    plt.plot(range(len(Y_Test)), Y_Test, "r", label="test data")
    plt.plot(range(len(Y_Test)), Y_Pred, "b", label="pred data")
    plt.legend()
    
    plt.subplot(122)
    plt.scatter(Y_Test, Y_Pred)
    plt.plot([Y_Test.min(), Y_Test.max()], [Y_Test.min(), Y_Test.max()], "k--")
    
    

if __name__=="__main__":
    Linear_train()
    