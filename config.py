# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:01:46 2024

@author: Curry
"""
import torch
from torch import nn

train_r = 0.75 #the ratio of the training sets
val_r = 0.25  #the ratio of the validation sets

vth = 4 #nth of the v-data that we want to fit by ML v~1,2,3,4,5,6 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#device = torch.device("cuda:0")
loss_function = nn.MSELoss()
epoch = 4000 
learning_rate = 0.01
gam = 0.98 
step_s = 50

#font setting
font = {'family': "Times New Roman", "weight":"normal", "size":20,}

#save_path 
save_path = "./training_results/N1_300_N2_600/Lr_{}_step_{}".format(learning_rate, step_s)
