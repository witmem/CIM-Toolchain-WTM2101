import onnx
import numpy as np
import witin
from witin import *
import os
import google.protobuf
import google.protobuf.text_format

import sys
sys.path.append("tests/onnx/witin/wtm2101/")
from engine.auto_optimizer import Optimizer

def npu_graph_build(mods, params, inputs, optimize_config):
    target = 'npu'
    target_host = 'npu'
    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestamp
    optimize_path = ""
    if optimize_config and optimize_config.ByteSize() > 0:
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        optimize_path = os.path.join(build_dir, "optimize_dccrn.protobuf")
        with open(optimize_path, "w") as f:
            txt_opti = google.protobuf.text_format.MessageToString(optimize_config)
            f.write(txt_opti)

    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(
            mods, target=target, target_host=target_host, params=params,
            input_data=inputs, optimize_method_config=optimize_path,
            output_dir=build_dir, chip="BB04P1")
    assert npu_graph is not None
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    for input_dict in inputs:
        for key, data in input_dict.items():
            m.set_input(key, data)

    return m



def get_fuse_output(graph_def,root_dir, protobuf_dir, input_mul):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'

    net_mods = []
    net_pms = []
    net_input_dts = []

    data = np.load(root_dir +'/layer_debug/conv1_input.npy').astype('float32')
    data_shape = data.shape
    print('ddddddddd',data.shape)
    data = data.astype('float32')  # [1, 2, 128, 301]
    # data = np.concatenate((data, data * (-1)), axis=1)*input_mul
    # data[data < 0] = 0 # [1, 2, 128, 301]
    data = np.transpose(data, (0, 3, 2, 1)) # [1, 301, 128, 4]
    data = data.reshape(data.shape[0] * data.shape[1], 1, data.shape[2], data.shape[3]) # [301, 1, 128, 4]
    # data = np.clip(data, 0, 255)

    # merge the adjacent 2 frame data, [301, 1, 128, 4] -> [300, 2, 128, 4]
    new_data = np.zeros([data.shape[0] - 1, data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype)
    for n in range(data.shape[0] - 1):
      new_data[n, 0, :, :] = data[n, :, :, :]
      new_data[n, 1, :, :] = data[n + 1, :, :, :]

    shape_dict_conv = {}
    shape_dict_conv['in'] = data_shape  # NCHW
    mod, param = witin_frontend.frontend.from_onnx(graph_def[0], shape_dict_conv)
    net_mods.append(mod)
    net_pms.append(param)

    input_dt_dccrn = {}
    input_dt_dccrn['in'] = witin.nd.array(new_data)
    net_input_dts.append(input_dt_dccrn)

    shape_dict = {}
    shape_dict['tdnn_in'] = (1, 128)
    mod, param = witin_frontend.frontend.from_onnx(graph_def[1], shape_dict)
    net_mods.append(mod)
    net_pms.append(param)

    #feats = np.loadtxt('../data/howl/feats_int8_yuexi.txt').astype("float32")
    #feats = np.loadtxt('../data/howl/data.txt').astype("float32")
    #data1 = feats[:][:]
    ##print(len(data1), len(data1[0]))
    ##data2 = data1
    #data2 = []
    #tempdata = np.zeros(128*3)
    #for i in range(len(data1)):
    #    tempdata = np.concatenate([tempdata[128:], data1[i]])
    #    data2.append(tempdata)
    #data2 = np.array(data2)
    #print(len(data2), len(data2[0]))
    #data2 = data2.astype(np.float32)

    data1 = np.loadtxt('./model/pipeline/auto_optimizer_model/dccrn_and_tdnn/howl/data.txt', delimiter=" ").astype("float32")
    print(data1.shape)
    tempdata = np.zeros(128*3)
    data2 = []
    for i in range(len(data1)):
        tempdata = np.concatenate([tempdata[128:], data1[i]])
        data2.append(tempdata)
    data2 = np.array(data2)
    print(len(data2), len(data2[0]))
    data2 = data2.astype(np.float32)
    #data2 = data1#data1[:]

    input_dt = {}
    input_dt['tdnn_in'] = witin.nd.array(data2)
    net_input_dts.append(input_dt)

    opt_config = ['./model/pipeline/auto_optimizer_model/dccrn_and_tdnn/protobuf/base.protobuf']

    for manual_opt_conf in opt_config:
        onnx_models = graph_def
        shape_dicts = [shape_dict_conv, shape_dict]
        inputs = net_input_dts
        mods = net_mods
        params = net_pms
        optimizer = Optimizer(onnx_models, shape_dicts, inputs, manual_opt_conf)

        for conf in optimizer.get_optimize_configs():
            if conf.ByteSize() > 0:
                graphs_inputs = optimizer.update_graphs_inputs(conf, inputs)
                runtime_model = npu_graph_build(mods, params, graphs_inputs, conf)
                assert runtime_model is not None
                runtime_model.run()
    # dateTimeObj = datetime.datetime.now()
    # timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    # #build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    # build_dir = root_dir+'net_configs/'+protobuf_name
    # #build_dir = os.path.join("./build/net_configs", protobuf_name)

    # with witin.transform.PassContext(opt_level=3):
    #     _, _, _, npu_graphs = witin_frontend.build_module.build(mods,
    #                                                             target=target,
    #                                                             target_host=target_host,
    #                                                             params=pms,
    #                                                             input_data=input_dts,
    #                                                             chip="BB04P1",
    #                                                             output_dir=build_dir,
    #                                                             optimize_method_config=opt_config)


def test_forward_multinet():
    root_dir = './model/pipeline/auto_optimizer_model/dccrn_and_tdnn/'
    protobuf_dir = './model/pipeline/auto_optimizer_model/dccrn_and_tdnn/protobuf/'

    howl_onnx = './model/pipeline/auto_optimizer_model/dccrn_and_tdnn/howl/howl_net_int8_0322.onnx'
    onnx_model = onnx.load(howl_onnx)

    ns_onnx = './model/pipeline/auto_optimizer_model/dccrn_and_tdnn/dccrn_denoise_net_0313_table1024.onnx'


    dccrn_onnx_model = onnx.load(ns_onnx)
    models = []
    models.append(dccrn_onnx_model)
    models.append(onnx_model)

    get_fuse_output(models,root_dir, protobuf_dir, 1)


test_forward_multinet()
