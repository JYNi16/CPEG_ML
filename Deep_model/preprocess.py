# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:27:32 2023

@author: 26526
"""
import numpy as np
import os, glob

def pre_data_for_vdata():
    p0 = "..//data_2//v_data"
    path = ["_x.txt", "_y.txt", "_z.txt", "_xy.txt", "_xz.txt", "_yz.txt", "_xyz.txt"]
    
    for p in path:
        pre = p0 + "//strain" + p
        data = []
        with open(pre, "r") as f:
            a = f.readlines()
        
        for i in range(len(a)):
            v = a[i].split()
            vs = [float(s) for s in v]
            data.append(vs)
        
        data_tmp = np.transpose(np.array(data))
        print("data_tmp.shape is:", data_tmp.shape)
        
        for j in range(data_tmp.shape[0]):
            data = data_tmp[j,:]
            #print("data is :", data)
            path_name = pre.split("//")[-1].split(".")[0]
            #print(path_name)
            if (j // 5 == 0):
                np.save("datasets//data_2//v_data//validation//{}_{:d}.npy".format(path_name,j), data)
            else:
                np.save("datasets//data_2//v_data//train//{}_{:d}.npy".format(path_name,j), data)

def pre_data_for_sdata():
    p0 = "..\data_2\s_data"
    path = ["_x.txt", "_y.txt", "_z.txt", "_xy.txt", "_xz.txt", "_yz.txt", "_xyz.txt"]
    for p in path:
        data_x = []
        data_B1 = []
        data_B2 = []
        pre = os.path.join(p0, "strain" + p)
        with open(pre, "r") as f:
            a = f.readlines()
        
        data_x.append([float(1+float(s)) for s in a[0].split()])
        data_B1.append([float(s) for s in a[1].split()])
        data_B2.append([float(s) for s in a[2].split()])
        
        
        nums = len(data_B1[0])
        num_ones = np.ones((1,nums))
        #for i in range(len(data_y[0])):
        #    num_zeros[0][i] = np.random.uniform(-1, 1)*0.02
        
        #changing the column order of the data array
        o_data = np.array(data_x)
        if "strain_x.txt" in pre:
            print("path is:", pre)
            data_xe = np.concatenate((o_data, num_ones), axis=0)
            data_xe = np.concatenate((np.array(data_xe), num_ones), axis=0)

        elif "strain_y.txt" in pre:
            print("path is:", pre)
            data_xe = np.concatenate((num_ones, o_data), axis=0)
            data_xe = np.concatenate((data_xe, num_ones), axis=0)
            
        elif "strain_z.txt" in pre:
            print("path is:", pre)
            data_xe = np.concatenate((num_ones, o_data), axis=0)
            data_xe = np.concatenate((num_ones, data_xe), axis=0)
        
        elif "strain_xy.txt" in pre:
            print("path is:", pre)
            data_xe = np.concatenate((o_data, np.array(o_data)), axis=0)
            data_xe = np.concatenate((data_xe, num_ones), axis=0)
        
        elif "strain_xz.txt" in pre:
            print("path is:", pre)
            data_xe = np.concatenate((o_data, num_ones), axis=0)
            data_xe = np.concatenate((data_xe, o_data), axis=0)
        
        elif "strain_yz.txt" in pre:
            print("path is:", pre)
            data_xe = np.concatenate((num_ones, o_data), axis=0)
            data_xe = np.concatenate((data_xe, o_data), axis=0)
        
        else:
            print("path is:", pre)
            data_xe = np.concatenate((o_data, o_data), axis=0)
            data_xe = np.concatenate((data_xe, o_data), axis=0)
            
            
        data_tmp = np.concatenate((data_xe, np.array(data_B1)), axis=0)
            
        data_tmp = np.transpose(data_tmp)
        
        print("data_tmp is:", data_tmp)
        
        for j in range(data_tmp.shape[0]):
            data = data_tmp[j,:]
            path_name = p.split(".")[0]
            if (j // 5 == 0):
                np.save("datasets//data_2//s_data//validation//{}_{:d}.npy".format(path_name,j), data)
            else:
                np.save("datasets//data_2//s_data//train//{}_{:d}.npy".format(path_name,j), data)
            

if __name__=="__main__":
    pre_data_for_vdata()
            
            