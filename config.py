# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:01:46 2024

@author: Curry
"""
import torch
from torch import nn

train_r = 0.75 #the ratio of the training sets
val_r = 0.25  #the ratio of the validation sets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
loss_function = nn.MSELoss()
epoch = 1000 
learning_rate = 0.01
gam = 0.98 
step_s = 10