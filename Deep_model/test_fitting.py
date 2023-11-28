# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:43:08 2023

@author: 26526
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as t
from torch.autograd import Variable as var

def get_data(x,w,b,d):
    c,r = x.shape
    y = (w * x * x + b*x + d)+ (0.1*(2*np.random.rand(c,r)-1))
    return(y)


xs_data = np.arange(-3,3,0.1).reshape(-1,1)
np.random.shuffle(xs_data)
xs = xs_data[:40]
ys = get_data(xs,1,-2,3)

xs_test = xs_data[40:]
ys_test = get_data(xs_test,1,-2,3)

xs = var(t.Tensor(xs))
ys = var(t.Tensor(ys))
xs_test = var(t.Tensor(xs_test))

class Fit_model(t.nn.Module):
    def __init__(self):
        super(Fit_model,self).__init__()
        self.l1 = t.nn.Linear(1,32)
        self.s1 = t.nn.ReLU()
        self.l2 = t.nn.Linear(32,1)
        #self.s2 = t.nn.Sigmoid()
        #self.l3 = t.nn.Linear(32,1)

        self.criterion = t.nn.MSELoss()
        self.opt = t.optim.SGD(self.parameters(),lr=0.01)
    def forward(self, input):
        y = self.l1(input)
        y = self.s1(y)
        y = self.l2(y)
        #y = self.s2(y)
        #y = self.l3(y)
        return y

model = Fit_model()
for e in range(2000):
    y_pre = model(xs)

    loss = model.criterion(y_pre,ys)
    print(e,loss.data)
    
    # Zero gradients
    model.opt.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    model.opt.step()

ys_pre = model(xs_test)

plt.title("curve")
#plt.plot(xs.data.numpy(),ys.data.numpy())
#plt.plot(xs.data.numpy(),ys_pre.data.numpy())
#plt.legend("ys","ys_pre")
plt.scatter(ys_test, ys_pre.data.numpy())
plt.plot([ys_test.min(), ys_test.max()], [ys_test.min(), ys_test.max()], "k--")
plt.show()