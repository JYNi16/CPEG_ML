# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:21:55 2023

@author: 26526
"""

import os, glob 
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader 

class data_set(Dataset):
    def __init__(self, path, train=True):
        self.path = path 
        self.train = train
        data_list = []
        data_list.extend(glob.glob(os.path.join(self.path, "*.npy")))
        #print(data_list)
        self.data_list = data_list
    
    def __getitem__(self,index):
        data_path = self.data_list[index]
        #print(data_path)
        data = np.load(data_path)
        x_data = data[0:-1] 
        y_data = data[-1]
        
        return x_data, y_data 
    
    def __len__(self):
        return len(self.data_list)


if __name__=="__main__":
    train_data = data_set("datasets//s_data//train")
    train_loader = DataLoader(train_data, batch_size = 4, shuffle=True)
    for x, y in train_loader:
        print("x.shape is:", x.shape)
        print("y.shape is:", y.shape)
        