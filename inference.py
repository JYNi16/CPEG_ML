import os
import numpy as np
from config import *
from model import Fit_net
from train import save_results

def inference_results_mesh():
    print("test now")
    r_s = -5 #range start
    r_f = 5 #range final
    d = 0.1
    input_x, input_y, input_z = np.arange(r_s, r_f, d), np.arange(r_s, r_f, d), np.arange(r_s, r_f, d)
    #print("input_x is:", input_x)
    X, Y, Z = np.meshgrid(input_x, input_y, input_z)
    xx, yy, zz = X.flatten(), Y.flatten(), Z.flatten()
    model_infer = Fit_net(N1, N2, N1, layers)
    print("load model is ok!")
    model_infer.load_state_dict(torch.load(save_path + "/model_v{}_{}.pth".format(vth, epoch), map_location=device))
    print("load infer model is ok!")
    model_infer.to(device)

    nums = len(xx)
    print("the number of datasets is:", nums)
    
    save_results(save_infer_path)
    #write the inference results into files
    with open(save_infer_path + "/infer_v{}_{}_epoch_{}_{}_{}_test.dat".format(vth, epoch, r_s*0.01, r_f*0.01, int((r_f-r_s)/d)), "w") as f1:
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

            #print(str(round(xx[i]*0.01, 6))+"   "+str(round(yy[i]*0.01, 6))+"   "+str(round(zz[i]*0.01,6))+"  "+str(y_pred), file=f1)
            print(str(round(xx[i]*0.01, 6))+"   "+str(round(yy[i]*0.01, 6))+"   "+str(round(zz[i]*0.01, 6))+"  "+str(y_pred), file=f1)
    f1.close()

def inference_results_uniform(input_d):
    print("test now")
    r_s = -3 #range start
    r_f = 3.1 #range final
    d = 0.1
    input_l = ["x", "y", "z", "xy", "xz", "yz", "xyz"]
    #identidy the idx of input_l
    if sum(input_d) == 1:
        idx = [input_d[k]*(k) for k in range(len(input_d))] 
    else:
        idx = [input_d[k]*(k+1) for k in range(len(input_d))]
    
    strain_idx = str(input_l[sum(idx)])
    input_x, input_y, input_z = np.arange(r_s, r_f, d), np.arange(r_s, r_f, d), np.arange(r_s, r_f, d)
    input_tmp = [input_x, input_y, input_z]
    input_s = []
    for i in range(len(input_d)):
        input_s.append([round(input_d[i]*s, 3) for s in input_tmp[i]])
    print("input_s is:", input_s)
    model_infer = Fit_net(N1, N2, N1, layers)
    print("load model is ok!")
    model_infer.load_state_dict(torch.load(save_path + "/model_v{}_{}.pth".format(vth, epoch), map_location=device))
    print("load infer model is ok!")
    model_infer.to(device)

    nums = len(input_x)
    print("the number of datasets is:", nums)
    save_results(save_infer_path) 
    
    save_text_file = save_infer_path + "/infer_v{0}_{1}_epoch_{2}_{3}_{4}_{s_idx}.dat".format(vth, epoch, round(r_s*0.01, 3), round((r_f-0.1)*0.01, 3), int((r_f-r_s)/d), s_idx=strain_idx)
    #write the inference results into files
    with open(save_text_file, "w") as f1:
        print("Inference results by NN model", file = f1)
        print("strain parameters and last term is v{}".format(vth), file = f1)
        print("strain_x" + "    " + "strain_y" + "     " + "strain_z" + "     " +  "v{}".format(vth), file = f1)
        for i in range(nums):
            input = np.array([input_s[0][i], input_s[1][i], input_s[2][i]]).reshape(-1, 3)
            print("input is:", input[0]*0.01)
            input_tensor = torch.tensor(input, dtype = torch.float32)
            out = model_infer(input_tensor.to(device, torch.float32))
            print("infer result is:", out.data.cpu().numpy()[0])

            y_pred = out.data.cpu().numpy()[0][0]
            #print(str(round(xx[i]*0.01, 6))+"   "+str(round(yy[i]*0.01, 6))+"   "+str(round(zz[i]*0.01,6))+"  "+str(y_pred), file=f1)
            print(str(round(input_s[0][i]*0.01, 6))+"     "+ str(round(input_s[1][i]*0.01, 6)) + "     " + str(round(input_s[2][i]*0.01, 6)) + "     " +str(y_pred), file=f1)
    
    f1.close()

def infer_main():

    strain_d = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]

    for idx in strain_d:
        inference_results_uniform(idx)

if __name__ == "__main__":
    inference_results_mesh()
