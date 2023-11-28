# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:05:44 2023

@author: 26526
"""

import torch 
from torch import nn
from dataset import *
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import StepLR



def get_data(path):
    data_x = []
    data_y = []
    data_list = []
    data_list.extend(glob.glob(os.path.join(path, "*.npy"))) 
    random.shuffle(data_list)
    
    #print(data_list)
    for i in range(len(data_list)):
        data_tmp = np.load(data_list[i])
        data_x.append(data_tmp[:-1])
        data_y.append(data_tmp[-1])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    return data_x, data_y
    

class Fit_model(nn.Module):
    def __init__(self):
        super(Fit_model,self).__init__()
        self.l1 = nn.Linear(6,100)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(100,100)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(100,100)
        self.r3 = nn.ReLU()
        self.l4 = nn.Linear(100,1)
        self.r4 = nn.ReLU()
        
    def forward(self, input):
        y = self.l1(input)
        y = self.r1(y)
        y = self.l2(y)
        y = self.r2(y)
        y = self.l3(y)
        y = self.r3(y)
        y = self.l4(y)
        y = self.r4(y)
        #y = self.r3(y)
        return y

p0 = "..//Deep_model//datasets//data_2/v_data//train"
p1 = "..//Deep_model//datasets//data_2//v_data//validation"
data_x, data_y = get_data(p0)
test_x, test_y = get_data(p1) 
data_x, data_y = torch.tensor(data_x, dtype=torch.float32),torch.tensor(data_y, dtype=torch.float32)
test_x, test_y = torch.tensor(test_x, dtype=torch.float32),torch.tensor(test_y, dtype=torch.float32) 
#train_loader = DataLoader(train_data, batch_size = 48, shuffle=True)
#val_data = data_set("datasets//s_data//validation")
#val_loader = DataLoader(train_data, batch_size = 1, shuffle=True)
print("data_x is:", data_x)
print("data_y is:", data_y)

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
model = Fit_model()
model.to(device)
epoch = 5000

optimizer = torch.optim.SGD(model.parameters(), lr = 0.008)
scheduler = StepLR(optimizer, step_size=50, gamma=0.99)
loss_function = nn.MSELoss()

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
        loss_train = train_(model, [data_x, data_y], train=True)
        loss_val = train_(model, [test_x, test_y], train=False)
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
    #plt.figure(1, figsize=(16,7))
    #plt.subplot(121)
    #y_test_pre = model(x_data_test)
    #plt.scatter(y_data_test, y_test_pre.data.numpy())
    #plt.plot([y_data_test.min(), y_data_test.max()], [y_data_test.min(), y_data_test.max()], "k--")
    
    #plt.subplot(122)
    #plt.plot(step_x, train_loss, "o-")
    #plt.plot(step_x, val_loss, "o-")
    #plt.show()


if __name__=="__main__":
    main()