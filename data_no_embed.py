# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:32:09 2024

@author: Curry
"""

import numpy as np
import os, glob
import matplotlib.pyplot as plt
from config import *

#function to obtain the dataets including strain and v 

def pre_for_svdata(vth): 
    
    """
    vth represent the nth of the v-data that we want to fit by ML 
    vth ~ 1,2,3,4,5,6
    
    return: training and validation datasets
    """
    
    path = ".//datasets//s_v_data//strain_z.txt"
    data = [] 
    with open(path, "r") as f:
        a = f.readlines() 
    
    for i in range(len(a)):
        v = a[i].split() 
        vs = [float(s) for s in v]
        print(len(vs))
        data.append(vs)
    
    plt.scatter(data[0], data[vth])
    plt.show()
    
    x_data = np.array(data[0]).reshape(1,-1)[0]
    y_data = np.array(data[vth]).reshape(1,-1)[0]
    
    train_data = [] 
    for i in range(len(x_data)):
        #Notice that we enlarge the values of x with 100 times
        train_data.append([x_data[i]*100, y_data[i]])
    
    train_data = np.array(train_data)
    
    np.random.shuffle(train_data)
    print("train_data is:", train_data.shape)
    
    x_data, y_data = [], [] 
    x_data_val, y_data_val = [], [] 
    
    train_nums = int(train_r*len(train_data))
    
    for i in range(len(train_data)):
        if i <= train_nums:
            x_data.append(train_data[i][0])
            y_data.append(train_data[i][1])
        else:
            x_data_val.append(train_data[i][0])
            y_data_val.append(train_data[i][1])
    
    x_data = np.array(x_data).reshape(-1, 1)
    y_data = np.array(y_data).reshape(-1, 1)
    x_data_val = np.array(x_data_val).reshape(-1,1)
    y_data_val = np.array(y_data_val).reshape(-1,1)
    
    #plt.scatter(x_data_val, y_data_val)
    #plt.show()
    
    return x_data, y_data, x_data_val, y_data_val



if __name__=="__main__":
    pre_for_svdata(3)