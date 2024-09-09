import onnx
import numpy as np
import witin
from witin import *
import os
import datetime
import google.protobuf
import google.protobuf.text_format
import sys
sys.path.append("tests/onnx/witin/wtm2101/")
from engine.auto_optimizer import Optimizer
np.random.seed(0)


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
        optimize_path = os.path.join(build_dir, "optimize_dccrn_hw.protobuf")
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


def run_model():
    onnx_model = onnx.load("./model/pipeline/auto_optimizer_model/auto_optimizer_dccrn_connect/test_dccrn_11-7.onnx")

    data = np.load('./model/pipeline/auto_optimizer_model/auto_optimizer_dccrn_connect/input1.npy').astype('float32')
    data = data.astype('float32') # [1, 2, 256, 101]
    
    data = np.transpose(data, (0, 3, 2, 1)) # [1, 101, 256, 4]
    data = data.reshape(data.shape[0] * data.shape[1], 1, data.shape[2], data.shape[3]) # [101, 1, 256, 4]

    # merge the adjacent 2 frame data, [101, 1, 256, 4] -> [100, 2, 256, 4]
    new_data = np.zeros([data.shape[0] - 1, data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype)
    for n in range(data.shape[0] - 1):
        new_data[n, 0, :, :] = data[n, :, :, :]
        new_data[n, 1, :, :] = data[n + 1, :, :, :]

    shape_dict = {}
    shape_dict['in'] = (1, 2, 256, 101)  # NCHW
    mod, param = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    input_dict = {}
    input_dict['in'] = witin.nd.array(new_data)

    manual_opt_conf = './model/pipeline/auto_optimizer_model/auto_optimizer_dccrn_connect/dccrn_opti.protobuf'
    onnx_models = [onnx_model]
    shape_dicts = [shape_dict]
    inputs = [input_dict]
    mods = [mod]
    params = [param]
    optimizer = Optimizer(onnx_models, shape_dicts, inputs, manual_opt_conf)
    

    for conf in optimizer.get_optimize_configs():
        if conf.ByteSize() > 0:
            graphs_inputs = optimizer.update_graphs_inputs(conf, inputs)
            runtime_model = npu_graph_build(mods, params, graphs_inputs, conf)
            assert runtime_model is not None
            runtime_model.run()


def test_dccrn_hw_auto_optimizer():
    run_model()


if __name__ == "__main__":
    test_dccrn_hw_auto_optimizer()
