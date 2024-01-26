# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:27:32 2023

@author: Curry
"""
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from config import *

def embeding(pre, x):
    
    tmp = 0
    
    embed_x = round(x*100, 6)
    
    if "strain_x.txt" in pre:
        return [embed_x, tmp, tmp] 
        
    elif "strain_y.txt" in pre:
        return [tmp, embed_x, tmp]
        
    elif "strain_z.txt" in pre:
        return [tmp, tmp, embed_x]
        
    elif "strain_xy.txt" in pre:
        return [embed_x, embed_x, tmp]
        
    elif "strain_xz.txt" in pre:
        return [embed_x, tmp, embed_x]
        
    elif "strain_yz.txt" in pre:
        return [tmp, embed_x, embed_x]
        
    else:
        return [embed_x, embed_x, embed_x]

#function to obtain the dataets including vs and strain with embeding xyz direction
def pre_for_svdata_xyz(vth):
    """
    vth represent the nth of the v-data that we want to fit by ML 
    vth ~ 1,2,3,4,5,6
    
    return: training and validation datasets
    """
     
    p0 = ".//datasets//s_v_data"
    path = ["_x.txt", "_y.txt", "_z.txt", "_xy.txt", "_xz.txt", "_yz.txt", "_xyz.txt"]
    
    train_data = [] 
    for p in path:
        pre = os.path.join(p0, "strain" + p)
        with open(pre, "r") as f:
            a = f.readlines()
        
        data_x = [float(s) for s in a[0].split()] #read the strain parameters
        data_y = [float(s) for s in a[vth].split()] #read vth data of v data
        #for i in range(len(data_y[0])):
        #    num_zeros[0][i] = np.random.uniform(-1, 1)*0.02
        
        #changing the column order of the data array
        nums = len(data_x)
        
        for i in range(nums):
            embed_x = embeding(pre, data_x[i])
           #print("now embed_x is:", embed_x, data_y[i])
            embed_data = [embed_x[0], embed_x[1], embed_x[-1], data_y[i]]
           #print("embed_data is:", embed_data)
            train_data.append(embed_data)
            #print("now train_data is:", train_data)
    train_data = np.array(train_data)
    np.random.shuffle(train_data)
        
    print("the nums of train_data is:", train_data.shape)
    
    x_data, y_data = [], [] 
    x_data_val, y_data_val = [], []
    
    train_nums = int(train_r*len(train_data))
    
    for i in range(len(train_data)):
        if i <= train_nums:
            x_data.append([train_data[i][s] for s in range(0,3)])
            #print("nth x data is:", x_data)
            y_data.append(train_data[i][-1])
        else:
            x_data_val.append([train_data[i][s] for s in range(0,3)])
            y_data_val.append(train_data[i][-1])
    
    
    x_data = np.array(x_data).reshape(-1,3)
    y_data = np.array(y_data).reshape(-1,1)
    x_data_val = np.array(x_data_val).reshape(-1,3)
    y_data_val = np.array(y_data_val).reshape(-1,1)
    
    print(x_data_val.shape)
    print(y_data_val.shape)
    
    return x_data, y_data, x_data_val, y_data_val 
   

if __name__=="__main__":
    x_data, y_data, x_data_val, y_data_val  = pre_for_svdata_xyz(1)
            
            