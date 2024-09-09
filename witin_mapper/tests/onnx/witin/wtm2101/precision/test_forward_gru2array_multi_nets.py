import numpy as np
import onnx
import os
import datetime
import witin
from witin import *

np.random.seed(0)
root_dir = "./model/pipeline/gruarray_multi_nets/"

optimize_config = root_dir + 'base.protobuf'

def get_data_vp1(root_dir):
    data = np.load(root_dir +'/layer_debug/conv1_in.npy').astype('float32')
    data = np.transpose(data, (0, 3, 2, 1)) # B T D C
    data = data.reshape(data.shape[0] * data.shape[1], 1, data.shape[2], data.shape[3]) # NW,1,H,C

    # merge the adjacent 2 frame data, [301, 1, 128, 4] -> [300, 2, 128, 4]
    new_data = np.zeros([data.shape[0], data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype)
    new_data[0, 1, :, :] = data[0, :, :, :]
    for n in range(data.shape[0] - 1):
        new_data[n+1, 0, :, :] = data[n, :, :, :]
        new_data[n+1, 1, :, :] = data[n + 1, :, :, :]
    print(new_data.shape)  # NW,KT,H,C
    
    return new_data

def get_data_vp2(root_dir):
    data = np.load(root_dir +'/layer_debug/conv2_in.npy').astype('float32')
    data = np.transpose(data, (0, 3, 2, 1)) # B T D C
    data = data.reshape(data.shape[0] * data.shape[1], 1, data.shape[2], data.shape[3]) # NW,1,H,C
    
    # merge the adjacent 2 frame data, [301, 1, 128, 4] -> [300, 2, 128, 4]
    new_data = np.zeros([data.shape[0], data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype)
    new_data[0, 1, :, :] = data[0, :, :, :]
    for n in range(data.shape[0] - 1):
        new_data[n+1, 0, :, :] = data[n, :, :, :]
        new_data[n+1, 1, :, :] = data[n + 1, :, :, :]
    print(new_data.shape)  # NW,KT,H,C
    
    return new_data
    
def get_data_vp3(root_dir):
    data = np.loadtxt(root_dir + '/layer_debug/input0.txt').astype('float32')
    return data

def test_forward_gru2array_dnn_multi_net():
    mod_list = []
    param_list = []
    input_list = []   
    
    # net 1
    shape_dict = {}
    shape_dict['conv1_in'] = [1, 2, 192, 515]
    model_vp1 = onnx.load(root_dir+"enc2101_0301_net1.onnx")
    mod, params = witin_frontend.frontend.from_onnx(model_vp1, shape_dict)
    new_data = get_data_vp1(root_dir)
    
    input_dt = {}
    input_dt['conv1_in'] = witin.nd.array(new_data)
    
    # mod_list.append(mod)
    # param_list.append(params)
    # input_list.append(input_dt)

    # net 2
    shape_dict_2 = {}
    shape_dict_2['conv2_in'] = [1, 8, 48, 515]
    model_vp2 = onnx.load(root_dir+"enc2101_0301_net2.onnx")
    mod_2, params_2 = witin_frontend.frontend.from_onnx(model_vp2, shape_dict_2)
    new_data_2 = get_data_vp2(root_dir)
    print(new_data_2.shape)
    input_dt_2 = {}
    input_dt_2['conv2_in'] = witin.nd.array(new_data_2)
    
    # mod_list.append(mod_2)
    # param_list.append(params_2)
    # input_list.append(input_dt_2)
    
    # net 3
    shape_dict_3 = {}
    shape_dict_3['gru1_in'] = [1, 224]
    model_vp3 = onnx.load(root_dir+"enc2101_0301_net3.onnx")
    mod_3, params_3 = witin_frontend.frontend.from_onnx(model_vp3, shape_dict_3)
    new_data_3 = get_data_vp3(root_dir)
    print(new_data_3.shape)
    input_dt_3 = {}
    input_dt_3['gru1_in'] = witin.nd.array(new_data_3)
    
    mod_list.append(mod_3)
    param_list.append(params_3)
    input_list.append(input_dt_3)
    
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    chip_type = "BB04P1"
    with witin.transform.PassContext(opt_level=3):
          _, _, _, npu_graph = witin_frontend.build_module.build(mod_list,
                                                                 target='npu',
                                                                 target_host='npu',
                                                                 params=param_list,
                                                                 input_data=input_list,
                                                                 chip = chip_type,
                                                                 output_dir=build_dir,
                                                                 optimize_method_config=optimize_config)
    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, chip_type, output_dir=build_dir)
    # m.set_input(0, witin.nd.array(new_data))    
    # m.set_input(1, witin.nd.array(new_data_2))
    m.set_input(0, witin.nd.array(new_data_3))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)

if __name__ == "__main__":
    test_forward_gru2array_dnn_multi_net()
