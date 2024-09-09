import imp
from termios import XTABS
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

from tvm.runtime.ndarray import array
import os
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

xt = np.random.randint(-50,50,size = (256,128),dtype=np.int32).astype(np.float32)
ht_1 = np.random.randint(-50,50,size = (256,128),dtype=np.int32).astype(np.float32)

# 拼接 ht-1 和 xt
xt_ht_1 = np.concatenate((ht_1,xt),axis=1)

weight_r = np.random.randint(-50,50,size = (256,128),dtype=np.int32).astype(np.float32)
weight_z = np.random.randint(-50,50,size = (256,128),dtype=np.int32).astype(np.float32)
bias_r = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32)
bias_z = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32)

bias_zr = np.concatenate((bias_z,bias_r),axis=0)
scale_zr = 0.0009765625

array1 = np.concatenate((weight_z,weight_r),axis=1)

# wx
wr_mul_xt_ht_1 = np.matmul(xt_ht_1,weight_r) 
wz_mul_xt_ht_1 = np.matmul(xt_ht_1,weight_z) 

# 计算 rt zt 
rt = sigmoid((wr_mul_xt_ht_1 + bias_r) * scale_zr)
zt = sigmoid((wz_mul_xt_ht_1 + bias_z) * scale_zr)
ztrt = np.concatenate((zt,rt),axis=1)

# 计算 ht-1' 
rt_ht_1 = rt * ht_1

# 拼接 ht-1' 和 xt 
xt_ht_1_ = np.concatenate((rt_ht_1,xt),axis=1)

weight_ht = np.random.randint(-50,50,size = (256,128),dtype=np.int32).astype(np.float32)
bias_ht = np.random.randint(-50,50,size = (128,),dtype=np.int32).astype(np.float32)
scale_ht = 0.0009765625

array2 = weight_ht
# wx
ht_wave_mul = np.matmul(xt_ht_1_,weight_ht)

# 对上一步拼接后的数据通过tanh函数激活，得到h'
ht_wave = np.tanh((ht_wave_mul + bias_ht) * scale_ht)

# ht = (1 - zt) * ht-1 + zt * h'
ht = (1 - zt) * ht_1 + zt * ht_wave

# save_params
params_path = "gru_params"
if not os.path.exists(params_path):
        os.mkdir(params_path)


np.savetxt("%s/array1.txt"%(params_path),array1)
np.savetxt("%s/array2.txt"%(params_path),array2)
np.savetxt("%s/bias_zr.txt"%(params_path),bias_zr)
np.savetxt("%s/bias_ht.txt"%(params_path),bias_ht)


np.savetxt("%s/xt.txt"%(params_path),xt)
np.savetxt("%s/ht_1.txt"%(params_path),ht_1)
np.savetxt("%s/rt.txt"%(params_path),rt)
np.savetxt("%s/zt.txt"%(params_path),zt)
np.savetxt("%s/rt_ht_1.txt"%(params_path),rt_ht_1)
np.savetxt("%s/ht_wave_mul.txt"%(params_path),ht_wave_mul)
np.savetxt("%s/htwave.txt"%(params_path),ht_wave)
np.savetxt("%s/ht.txt"%(params_path),ht)
