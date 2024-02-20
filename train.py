# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:32:56 2023

@author: Curry
"""
import os
import numpy as np
import matplotlib.pyplot as plt 
import random 
import data_no_embed as data_NE
import data_embed as data_Emb
import torch 
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR
from config import *

from model import Fit_net

    
#model setting

print("device is:", device)
model = Fit_net(N1, N2, N1, layers)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
scheduler = StepLR(optimizer, step_size= step_s, gamma = gam)

def save_results():
    if os.path.exists(save_path):
        print("save path {} exist".format(save_path))
    else:
        print("save path {} not exist".format(save_path))
        os.makedirs(save_path)
        print("now makefir the save_path")

def inference_results(): 
    print("test now")
    r_s = -10 #range start 
    r_f = 10 #range final 
    d = 0.4
    input_x, input_y, input_z = np.arange(r_s, r_f, d), np.arange(r_s, r_f, d), np.arange(r_s, r_f, d)
    #print("input_x is:", input_x) 
    X, Y, Z = np.meshgrid(input_x, input_y, input_z)
    xx, yy, zz = X.flatten(), Y.flatten(), Z.flatten() 
    model_infer = Fit_net(N1, N2, N1, layers)
    print("load model is ok!")
    model_infer.load_state_dict(torch.load(save_path + "/model_v{}.pth".format(vth), map_location=device))
    print("load infer model is ok!")
    model_infer.to(device)

    nums = len(xx)

    print("the number of datasets is:", nums)

    #write the inference results into files
    with open(save_path + "/infer_v{}_{}_epoch_{}_{}_{}.dat".format(vth, epoch, r_s*0.01, r_f*0.01, int((r_f-r_s)/d)), "w") as f1:
        print("Inference results by NN model", file = f1)
        print("1st is strain parameter along x, 2nd is y, 3rd is z and last term is v{}".format(vth), file = f1)
        print("strin_x"+ "    " + "strain_y" + "    " + "strain_z" + "    " + "v{}".format(vth), file = f1)
        for i in range(nums):            
            input = np.array([xx[i], yy[i], zz[i]]).reshape(-1, 3)
            print("input is:", input*0.01)
            input = torch.tensor(input, dtype = torch.float32)
            out = model_infer(input.to(device, torch.float32))
            print("infer result is:", out.data.cpu().numpy()[0])

            y_pred = out.data.cpu().numpy()[0][0]

            print(str(round(xx[i]*0.01, 6))+"   "+str(round(yy[i]*0.01, 6))+"   "+str(round(zz[i]*0.01,6))+"  "+str(y_pred), file=f1)
    
    f1.close()

def generate_data(): 
    
    x_data, y_data, x_data_val, y_data_val  = data_Emb.pre_for_svdata_xyz(vth)
    
    print("the training dataset size is:", x_data.shape)
    
    x_data = torch.tensor(x_data, dtype = torch.float32)
    y_data = torch.tensor(y_data, dtype = torch.float32)

    x_data_val = torch.tensor(x_data_val, dtype = torch.float32)
    y_data_val = torch.tensor(y_data_val, dtype = torch.float32) 
    
    return x_data, y_data, x_data_val, y_data_val



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
        y_pre = model(x.to(device, torch.float32))
    else:
        torch.set_grad_enabled(False)
        model.eval()
        y_pre = model(x.to(device, torch.float32))
    #print("y_pre.shape is:", y_pre.shape)
    loss = loss_function(y_pre, y.to(device, torch.float32))
    
    #procedure for training
    if train:
        loss.backward()
        optimizer.step()
        #plt.scatter(y.data.numpy(), y_pre.data.numpy())
        #plt.plot([y.data.numpy().min(), y.data.numpy().max()], [y.data.numpy().min(), y.data.numpy().max()], "k--")
        #plt.pause(0.001)    
    
    #procedure for validation
    #else:
    #    plt.scatter(y.data.numpy(), y_pre.data.numpy())
    #    plt.plot([y.data.numpy().min(), y.data.numpy().max()], [y.data.numpy().min(), y.data.numpy().max()], "k--")
    #    plt.pause(0.001)

    return loss 

def main():
    
    val_loss = []
    train_loss = []
    step_x = []
    
    x_data, y_data, x_data_val, y_data_val = generate_data()
    
    print("Now the training process start !!!")
    
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
    
    print("The training process finished !!!")   

    print("now test start !!!")
    #test process
    
    plt.figure(1, figsize=(16,7))
    plt.subplot(121)
    y_test_pre = model(x_data_val.to(device, torch.float32))
    plt.scatter(y_data_val, y_test_pre.data.cpu().numpy())
    plt.plot([y_data_val.min(), y_data_val.max()], [y_data_val.min(), y_data_val.max()], "k--")
    plt.xlabel(r"$v^{}$".format(vth) + "(DFT)", font)
    plt.ylabel(r"$v^{}$".format(vth) + "(Pred)", font)
    
    #plt.figure(1, figsize=(6,6))
    plt.subplot(122)
    plt.plot(step_x, train_loss, "o-",)
    plt.plot(step_x, val_loss, "o-",)
    plt.xlabel("epoch", font)
    plt.ylabel("loss", font)
    save_results()
    plt.savefig(save_path + "/v{}_{}_epoch.png".format(vth, epoch), dpi=500)
    #plt.show()

    
    #save model 
    torch.save(model.state_dict(), save_path + "/model_v{}.pth".format(vth))

    #inferrence start !!! 
    print("inference process start !!!")

    #start inference result 
    #inference_results()


if __name__=="__main__":
    inference_results()
