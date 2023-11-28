# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:32:56 2023

@author: 26526
"""

import numpy as np
import matplotlib.pyplot as plt 
import random 
import torch 
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR

data_nums = 100

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Linear(3,100)
        self.l2 = nn.Linear(100,100)
        self.l3 = nn.Linear(100, 1)
    
    def forward(self,x):
        x = self.l1(x)
        x = F.relu(x)
        for i in range(2):
            x = self.l2(x)
            x = F.relu(x)
        y = self.l3(x)
        
        return y 

def generate_data():
    train_data = np.zeros((data_nums,4))

    for i in range(data_nums):
        train_data[i][0] = random.uniform(-1,1)
        train_data[i][1] = random.uniform(-1,1)
        train_data[i][2] = random.uniform(-1,1)
        train_data[i][3] = -0.02*train_data[i][0] **2 + 0.015*np.exp(train_data[i][1]) + -0.04*np.cos(train_data[i][3])

    x_data = train_data[:70,0:3]
    y_data = train_data[:70, 3].reshape(70, -1)

    x_data_val = train_data[70:85,0:3]
    y_data_val = train_data[70:85,3].reshape(15,-1)

    x_data_test = train_data[85:,0:3]
    y_data_test = train_data[85:,3].reshape(15,-1)

    print(x_data.shape, y_data.shape)
    
    return x_data, y_data, x_data_val, y_data_val, x_data_test, y_data_test
    


x_data, y_data, x_data_val, y_data_val, x_data_test, y_data_test = generate_data()

model = net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_function = nn.MSELoss()
epoch = 2000

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
x_data = torch.tensor(x_data, dtype = torch.float32)
y_data = torch.tensor(y_data, dtype = torch.float32)

x_data_val = torch.tensor(x_data_val, dtype = torch.float32)
y_data_val = torch.tensor(y_data_val, dtype = torch.float32)

x_data_test = torch.tensor(x_data_test, dtype = torch.float32)
#y_data_test = torch.tensor(y_data_test, dtype = torch.float32)

scheduler = StepLR(optimizer, step_size=10, gamma=0.99)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

plt.ion()
def train_(model, data,  train=True):
    x, y = data
    #print("x.shape is:", x.shape)
    if train:
        torch.set_grad_enabled(True)
        model.train()
        optimizer.zero_grad()
        y_pre = model(x)
    else:
        torch.set_grad_enabled(False)
        model.eval()
        y_pre = model(x)
    #print("y_pre.shape is:", y_pre.shape)
    loss = loss_function(y_pre, y)
    
    if train:
        loss.backward()
        optimizer.step()
    
    else:
        plt.scatter(y.data.numpy(), y_pre.data.numpy())
        plt.plot([y.data.numpy().min(), y.data.numpy().max()], [y.data.numpy().min(), y.data.numpy().max()], "k--")
        plt.pause(0.001)
    return loss 

def main():
    
    val_loss = []
    train_loss = []
    step_x = []
    
    for step in range(epoch):
        loss_train = train_(model, [x_data, y_data], train=True)
        loss_val = train_(model, [x_data_val, y_data_val], train=False)
        step_x.append(step)
        train_loss.append(float(loss_train))
        val_loss.append(float(loss_val))
        
        if step % 10 == 0:
            print("epoch is {} | train loss is:{}".format(step, loss_train))
            print("epoch is {} | val loss is:{}".format(step, loss_val))
            
            lr = get_lr(optimizer)
            print("learning rate is:", lr)
        
        scheduler.step()
     
    #plt.ioff()
    
    #test process
    plt.figure(1, figsize=(16,7))
    plt.subplot(121)
    y_test_pre = model(x_data_test)
    plt.scatter(y_data_test, y_test_pre.data.numpy())
    plt.plot([y_data_test.min(), y_data_test.max()], [y_data_test.min(), y_data_test.max()], "k--")
    
    plt.subplot(122)
    plt.plot(step_x, train_loss, "o-")
    plt.plot(step_x, val_loss, "o-")
    plt.show()
    

if __name__=="__main__":
    main()
        
