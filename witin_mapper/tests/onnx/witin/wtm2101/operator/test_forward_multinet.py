import numpy as np
import onnx
import os
import datetime
import witin
from witin import *

root_dir = './model/huawei_model/'


def get_fuse_output(graph_def):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'

    mods = []
    pms = []
    input_dts = []

    shape_dict = {}
    shape_dict['fingerprint_input'] = (1, 39)
    mod, params = witin_frontend.frontend.from_onnx(graph_def[0], shape_dict)
    mods.append(mod)
    pms.append(params)
    np.random.seed(10)
    data1 = np.round(np.random.rand(100, 195) * 255).astype("float32")

    input_dt = {}
    input_dt['fingerprint_input'] = witin.nd.array(data1)
    input_dts.append(input_dt)

    shape_dict_vad = {}
    shape_dict_vad['fingerprint_input'] = (1, 514)
    mod_vad, params_vad = witin_frontend.frontend.from_onnx(graph_def[1], shape_dict_vad)
    mods.append(mod_vad)
    pms.append(params_vad)

    data_vad = np.round(np.random.rand(100, 514) * 255).astype("float32")
    input_dt_vad = {}
    input_dt_vad['fingerprint_input'] = witin.nd.array(data_vad)
    input_dts.append(input_dt_vad)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graphs = witin_frontend.build_module.build(mods,
                                                                target=target,
                                                                target_host=target_host,
                                                                params=pms,
                                                                input_data=input_dts,
                                                                chip="BB04P1",
                                                                output_dir=build_dir,
                                                                optimize_method_config=root_dir+'optimize_config_multinet.protobuf',
                                                                )

    ######################################################################
    # multi net runtime create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime

    m = npu_graph_runtime.create(npu_graphs, "BB04P1", output_dir=build_dir)
    # 两种方法设置输入
    # 1.已知输入节点的名称
    m.set_input('fingerprint_input', witin.nd.array(data1))
    m.set_input('fingerprint_input', witin.nd.array(data_vad))
    # 2.不知道节点名称，按照Graph的顺序
    # m.set_input(0, witin.nd.array(data1))
    # m.set_input(1, witin.nd.array(data_vad))

    m.run()

    # m.get_num_outputs() 获得输出节点的个数
    # m.get_output(i) 获得第i个输出节点的所有帧输出
    output = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    # print(output[0][1:100])


def test_forward_multinet():
    np.random.seed(10)
    kws_onnx = root_dir + 'huawei_hz_e2e_3layer_1residual_v2.onnx'
    nr_onnx = root_dir + 'huawei_hz_e2e_decoder_1_relu.onnx'
    onnx_model = onnx.load(kws_onnx)
    vad_onnx_model = onnx.load(nr_onnx)
    models = []
    models.append(onnx_model)
    models.append(vad_onnx_model)
    tvm_out = get_fuse_output(models)

test_forward_multinet()
