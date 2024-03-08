import os
import numpy as np
from config import *
from model import Fit_net

def inference_results_mesh():
    print("test now")
    r_s = -3 #range start
    r_f = 3 #range final
    d = 0.1
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
    with open(save_path + "/infer_v{}_{}_epoch_{}_{}_{}_test.dat".format(vth, epoch, r_s*0.01, r_f*0.01, int((r_f-r_s)/d)), "w") as f1:
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

def inference_results_uniform():
    print("test now")
    r_s = -3 #range start
    r_f = 3.1 #range final
    d = 0.1
    input_d = [1, 0, 0]
    input_x, input_y, input_z = np.arange(r_s, r_f, d), np.arange(r_s, r_f, d), np.arange(r_s, r_f, d)
    input_tmp = [input_x, input_y, input_z]
    input_s = []
    for i in range(len(input_d)):
        input_s.append([round(input_d[i]*s, 3) for s in input_tmp[i]])
    print("input_s is:", input_s)
    model_infer = Fit_net(N1, N2, N1, layers)
    print("load model is ok!")
    model_infer.load_state_dict(torch.load(save_path + "/model_v{}.pth".format(vth), map_location=device))
    print("load infer model is ok!")
    model_infer.to(device)

    nums = len(input_x)
    print("the number of datasets is:", nums)

    #write the inference results into files
    with open(save_path + "/infer_v{}_{}_epoch_{}_{}_{}_x.dat".format(vth, epoch, r_s*0.01, r_f*0.01, int((r_f-r_s)/d)), "w") as f1:
        print("Inference results by NN model", file = f1)
        print("1st is strain parameter along x, 2nd is y, 3rd is z and last term is v{}".format(vth), file = f1)
        print("strin_x"+ "    " + "strain_y" + "    " + "strain_z" + "    " + "v{}".format(vth), file = f1)
        for i in range(nums):
            input = np.array([input_s[0][i], input_s[1][i], input_s[2][i]]).reshape(-1, 3)
            print("input is:", input)
            input_tensor = torch.tensor(input, dtype = torch.float32)
            print("trans input is:", input_tensor)
            out = model_infer(input_tensor.to(device, torch.float32))
            print("infer result is:", out.data.cpu().numpy()[0])

            y_pred = out.data.cpu().numpy()[0][0]
            #print(str(round(xx[i]*0.01, 6))+"   "+str(round(yy[i]*0.01, 6))+"   "+str(round(zz[i]*0.01,6))+"  "+str(y_pred), file=f1)
            print(str(round(input_s[0][0]*0.01, 6))+"   "+str(round(input_s[1][1]*0.01, 6))+"   "+str(round(input_s[2][2]*0.01, 6))+"  "+str(y_pred), file=f1)
    f1.close()


if __name__ == "__main__":
    inference_results_uniform()