# Copyright(c) 2022 Witin Tech Corporation. All Rights Reserved.
# Author: shihang.fu
# Email: shihang.fu@witintech.com
# Last modified: 2022-06-08
# Filename: test_multi_thread.py
# Description:

import threading
import numpy as np
import onnx
from onnx import helper, TensorProto, mapping
import witin
from witin import *


# 为线程定义一个函数
def build_run_model(model_file,
                    input_shape,
                    input_data,
                    optimize_method_config="",
                    output_dir="./output"):
    print("do........", model_file)
    onnx_model = onnx.load(model_file)
    shape_dict = {}
    shape_dict['fingerprint_input'] = input_shape
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    input_dt = {}
    input_data = input_data.astype('float32')
    input_dt['input_data'] = witin.nd.array(input_data)
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(
            mod,
            target='npu',
            target_host='npu',
            params=params,
            input_data=input_dt,
            chip="BB04P1",
            optimize_method_config=optimize_method_config,
            output_dir=output_dir)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=output_dir)
    m.set_input('fingerprint_input', witin.nd.array(input_data))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(1)]


def test_multi_thread():
    model_file1 = "./model/huawei_model/huawei_hz_e2e_2layer.onnx"
    input_shape1 = (1, 39)
    input_data1 = np.random.rand(150, 195) * 255  # 5-frame tdnn
    optimize_method_config1 = ""
    output_dir1 = "./ouptut/build_" + os.path.basename(__file__).split(".")[0] + "_1",

    model_file2 = "./model/huawei_model/huawei_hz_e2e_3layer_1residual.onnx"
    input_shape2 = (1, 39)
    input_data2 = np.random.rand(150, 195) * 255
    optimize_method_config2 = ""
    output_dir2 = "./output/build_" + os.path.basename(__file__).split(".")[0] + "_2",

    # 创建两个线程
    thread1 = threading.Thread(target=build_run_model,
                               args=(model_file1, input_shape1, input_data1,
                                     optimize_method_config1, output_dir1))
    thread2 = threading.Thread(target=build_run_model,
                               args=(model_file2, input_shape2, input_data2,
                                     optimize_method_config2, output_dir2))
    #启动线程
    thread1.start()
    thread2.start()


if __name__ == "__main__":
    test_multi_thread()
