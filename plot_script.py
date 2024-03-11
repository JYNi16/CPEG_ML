import os
import numpy as np
import matplotlib.pyplot as plt
from config import * 
from train import save_results

def svdata_xyz_dft(s_idx):
    """
    vth represent the nth of the v-data that we want to fit by ML 
    vth ~ 1,2,3,4,5,6
    
    return: training and validation datasets
    """
     
    path = ".//datasets//s_v_data//strain_{name}.txt".format(name=s_idx)
    
    with open(path, "r") as f:
        a = f.readlines()
        #print("len(a) is:", len(a))
        v = [[], [], [], [], [], []]
        data_x = [float(s) for s in a[0].split()] #read the strain parameters
        for i in range(1, 7):
            #print("ith is:", i)
            v[i-1] = [float(s) for s in a[i].split()]
            #print(v[i-1])
    
    return data_x, v 


def svdata_xyz_ml(input_d):

    input_l = ["x", "y", "z", "xy", "xz", "yz", "xyz"]
    #identidy the idx of input_l
    if sum(input_d) == 1:
        idx = [input_d[k]*(k) for k in range(len(input_d))] 
    else:
        idx = [input_d[k]*(k+1) for k in range(len(input_d))]
    
    strain_idx = str(input_l[sum(idx)])
    path = ".//datasets//s_v_data//strain_{name}.txt".format(name=strain_idx)

    v = [[], [], [], [], [], []]
    x = [[], [], [], [], [], []]
    for k in range(1,7):
        infer_text = save_infer_path + "/infer_v{}_{}_epoch_-0.03_0.03_60_{idx}.dat".format(int(k), epoch, idx=strain_idx)
        with open(infer_text, "r") as f:
            a = f.readlines()
            for i in range(3, len(a)):
                #print("a_i is:", i, "len(a) is:", len(a))
                data = [float(s) for s in a[i].split()]
                #print("data is:", data)
                x[k-1].append(sum([data[w]*input_d[w] for w in [0,1,2]])/sum(input_d))
                v[k-1].append(data[-1]) 
                
    return x, v


def plot():
    
    input_d = [1, 1, 0]
    input_l = ["x", "y", "z", "xy", "xz", "yz", "xyz"]
    font = {'family': "Times New Roman", "weight":"normal", "size":26,}
    font2 = {'family': "Times New Roman", "weight":"normal", "size":64,}
    #identidy the idx of input_l
    if sum(input_d) == 1:
        idx = [input_d[k]*(k) for k in range(len(input_d))] 
    else:
        idx = [input_d[k]*(k+1) for k in range(len(input_d))]
    
    
    strain_idx = str(input_l[sum(idx)])
    
    print("strain_idx is:", strain_idx)
    strain_x_dft, y_dft = svdata_xyz_dft(strain_idx)
    strain_x_ml, y_ml = svdata_xyz_ml(input_d)

    plt.figure(6, figsize=(24,12.5))
    
    for i in range(6):
        plt.subplot(2,3,int(i+1))
        plt.scatter(strain_x_dft, y_dft[i], s=50, color = "seagreen", label = "DFT")
        #plt.plot(strain_x_dft, y_dft[i], linewidth = 4,  color = "seagreen", label = "DFT")
        plt.scatter(strain_x_ml[i], y_ml[i],s=50, color = "red",  label = "Machine Learning")
        #plt.plot(strain_x_ml[i], y_ml[i], linewidth = 4, color = "red",  label = "Machine Learning")
        
        plt.xticks(fontproperties='Times New Roman', fontsize = 18)
        plt.yticks(fontproperties='Times New Roman', fontsize = 18)
        
        plt.ylabel(r"$v_{}$".format(int(i+1)), fontdict = font)
    
    #title = "v with strain along {}".format(strain_idx)
    plt.legend(loc = "upper right",prop = {'family': "Times New Roman", "weight":"normal", "size":26,}, frameon=False)
    #plt.title(title,loc = "center",fontdict={"size":"xx-large","color":"black", "family":"Times New Roman"})
    plt.suptitle("v under strain along {}".format(strain_idx), fontsize = 32, y =0.93)
    
    save_results(save_infer_path) 
    plt.savefig(save_infer_path + "/{}_epoch_infer_{name}.png".format(epoch, name=strain_idx), dpi=500)
    
    

if __name__=="__main__":
    plot() 