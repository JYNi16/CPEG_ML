# CPEG_ML
Based on the artifical Neural Network to fit the optical parameters

Before starting this project, you should install following libraries: 

1. Pytorch cpu or cuda version are both fine.
2. Numpy; matplotlib

You can modify the training details/saving path in config.py.\
You can start to train model by following command: python train.py. \

The function of each py.files are defined as follows: \

1. config.py\
   vth： the nth velocity with range of v1~v6\
   epoch： total step of training process.\  
   N1/N2 mean that the nums of nerons in each layer and layesr is the nums of hidden layers.\
   save_path is the training saving path, which include the training results.\
   save_infer_path include the inference results.

2. inference.py\
   
   (1). inference_results_mesh().\
   In this function, you can infer the vth by feeding the mesh inputs into the well training model.\ 
   The process of this calculation function is:
   
   for x in (-0.05, 0.05, 0.01):

        for y in (-0.05, 0.05, 0.01):

             for z in (-0.05, 0.05, 0.01):

                 vth = model([x, y, z])
   
   (2). inference_results_uniform().
   In this function, you can infer the vth by feeding the uniform inputs into the well training model. 
   The process of this calculation function is:

   for x in (-0.05, 0.05, 0.01):

        y, z = x

        vth = model([x, y, z])
   
   Additionally, you can change the strain direction by modifying the input_d vector. For example, input_d = [1, 0, 0] corresponds to strain along the x-direction, [1, 1, 0] corresponds xy direction.

   
