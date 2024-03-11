# CPEG_ML
Based on the artifical Neural Network to fit the optical parameters

Before starting this project, you should install following libraries

1. Pytorch cpu or cuda version are both fine.
2. Numpy; matplotlib

You can modify the training details/saving path in config.py 

You can start to train model by following command: python train.py ...

The function of each part is defined as follows: 

1. config.py
   vth： the nth velocity with range of v1~v6

   epoch： total step of training process  

   N1/N2 mean that the nums of nerons in each layer and layesr is the nums of hidden layers

   save_path is the training saving path, which include the training results, saving model 

   save_infer_path include the inference results

3. inference.py




