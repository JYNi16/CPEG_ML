# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:27:32 2023

@author: Curry
"""
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import random

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
     
    p0 = "..//data_3//s_v_data"
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
            print("now embed_x is:", embed_x, data_y[i])
            embed_data = embed_x
            print("embed_data is:", embed_data)
            train_data.append(embed_data)
            #print("now train_data is:", train_data)
    
    train_data = np.random.shuffle(np.array(train_data))
        
    print("train_data is:", train_data)
        
        #for j in range(data_tmp.shape[0]):
        #    data = data_tmp[j,:]
        #    path_name = p.split(".")[0]
        #    if (j // 5 == 0):
        #        np.save("datasets//data_3//s_data//validation//{}_{:d}.npy".format(path_name,j), data)
        #    else:
        #        np.save("datasets//data_3//s_data//train//{}_{:d}.npy".format(path_name,j), data)

if __name__=="__main__":
    pre_for_svdata_xyz(1)
            
            