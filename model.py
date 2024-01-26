# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:36:07 2024

@author: Curry

"""
import torch 
from torch.nn import functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from config import *

class Fit_net(nn.Module):
    def __init__(self, c1, c2, m, num):
        """
        Parameters
        ----------
        c1 : channels of input layer.
        c2 : channels of output layer.
        m : channels of middle layer.
        num : numbers of the middle layers
        """
        super(Fit_net, self).__init__()
        self.in_layer = nn.Linear(3,c1)
        self.mid_layer1 = nn.Linear(c1, m)
        self.mid_layer2 = nn.Linear(m, m)
        self.mid_layer3 = nn.Linear(m, c2)
        self.out_layer = nn.Linear(c2, 1)
        self.num = num
    
    def forward(self,x):
        x = self.in_layer(x)
        x = F.relu(x)
        x = self.mid_layer1(x)
        x = F.relu(x)
        for i in range(self.num):
            x = self.mid_layer2(x)
            x = F.relu(x)
        x = self.mid_layer3(x)
        x = F.relu(x)
        y = self.out_layer(x)
        
        return y 

if __name__=="__main__":
    net = Fit_net(200, 400, 200, 1)

    input = torch.tensor(np.array([1.2, 1.3, 1.1]).reshape(-1,3), dtype = torch.float32)

    out = net(input)

    print(out)
