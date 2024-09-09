import torch
import torch.nn as nn
import torch.nn.functional as F
import witin
from witin import *
import numpy as np


class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()
        # self.in_planes = 24
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3,
                               stride=2, padding=1, bias=True)

        self.conv2 = nn.Conv2d(12, 12, kernel_size=3,
                        stride=2, padding=1, bias=True)

        self.conv3 = nn.Conv2d(12, 16, kernel_size=3,
                        stride=2, padding=1, bias=True)

        self.conv4 = nn.Conv2d(16, 18, kernel_size=3,
                        stride=2, padding=1, bias=True)

        self.conv5 = nn.Conv2d(18, 24, kernel_size=3,
                        stride=2, padding=1, bias=True)

        self.conv6 = nn.Conv2d(24, 32, kernel_size=3,
                        stride=2, padding=1, bias=True)

        self.linear = nn.Linear(32, num_classes)

	#0.000244140625 4096
	#0.00048828125 2048
	#0.0009765625 1024
	#0.001953125 512
	#0.00390625 256

    def forward(self, x):
        out = self.conv1(x)
        out = out * 0.001953125 #512
        #out = torch.mul(out,2)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = out * 0.0009765625
        #out = torch.mul(out,2)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = out * 0.001953125
        #out = torch.mul(out,2)
        out = F.relu(out)

        out = self.conv4(out)
        out = out * 0.0009765625
        #out = torch.mul(out,2)
        out = F.relu(out)

        out = self.conv5(out)
        out = out * 0.0009765625
        #out = torch.mul(out,2)
        out = F.relu(out)

        out = self.conv6(out)
        out = out * 0.0009765625
        #out = torch.mul(out,2)
        out = F.relu(out)

        out = out.view(out.size(0), -1)   # view reshape
        #out = self.linear(out)
        return out


chip = "BB04P1" # NB01PP,BB04P1
model_path = "./model/cnn_model/ningshiDetG.pt"
input_name = 'input'
input_shape = [1,1,64,64]
shape_list = [(input_name, input_shape)]

# model = Net()
model = ResNet()
# model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

model.load_state_dict(torch.load('./model/cnn_model/ningshiDetG.pt', map_location=lambda storage, loc:storage))

model.cpu()
model.eval()

# model_dict =  model.state_dict()
# # state_dict = {k:v for k,v in model_state_dict.items() if k in model_dict.keys()}
# state_dict = {}
# for k,v in model_state_dict.items():
#     if k in model_dict.keys():
#         v = torch.round(v*1024)
#         # if "bias" in k:
#         #     state_dict[k] = torch.round(v*1024)
#         # else:
#         #     state_dict[k] = torch.clamp(v,-128,127)
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)

data = torch.rand(input_shape)
scripted_model = torch.jit.trace(model, data).eval()

mod, params = witin_frontend.frontend.from_pytorch(scripted_model, shape_list)

# compile the model
target = 'npu'
target_host = 'npu'
input_dt = {}
torch.manual_seed(1)

# duys comment 这个地方需要输入真实的参考数据 爱飞建议最少100个样本 
# data = torch.rand([100,28,28,1])*255
# must be float32
data = np.loadtxt(os.path.join("/home/ubuntu/duys/simulation", "ningshiDetExpected_in0.txt")).astype("float32")
#data = np.loadtxt('/home/ubuntu/duys/simulation/ningshiDetExpected_in0.txt').astype("float32")
# N H W C
# data = data.reshape((data.shape[0], 64, 64, 1))
data = data.reshape((100, 64, 64, 1))

input_dt['input_data'] = witin.nd.array(data)
#generate config files in build/output/ false
# pdb.set_trace()
with witin.transform.PassContext(opt_level=3):
    _, _, _, npu_graph = witin_frontend.build_module.build(mod, target=target,
                        target_host=target_host, params=params, 
                        input_data=input_dt,
                        chip = chip
                        )
