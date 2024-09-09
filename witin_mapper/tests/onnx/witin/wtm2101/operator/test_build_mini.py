# Copyright(c) 2022 Witin Tech Corporation. All Rights Reserved.
# Author: shihang.fu
# Email: shihang.fu@witintech.com
# Last modified: 2022-06-10
# Filename: test_build_mini.py
# Description:

import time
import numpy as np

start = time.time()
import onnx

end = time.time()
print("导入onnx库时间:%.2f秒" % (end - start))

start = time.time()
import witin
from witin import *

end = time.time()
print("导入witin库时间:%.2f秒" % (end - start))

from test_forward_hua_shibie_all import generate_tdnn_model


# 为线程定义一个函数
def build_run_model(model_file,
                    input_shape,
                    input_data,
                    optimize_method_config="",
                    output_dir="./output",
                    array_distribute=""):
    onnx_model = onnx.load(model_file)
    shape_dict = {}
    shape_dict['in'] = input_shape
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
            output_dir=output_dir,
            array_distribute=array_distribute)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    # from tvm.contrib import npu_graph_runtime
    # m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=output_dir)
    # m.set_input('fingerprint_input', witin.nd.array(input_data))
    # # execute
    # m.run()
    # output = [m.get_output(i).asnumpy() for i in range(1)]


def test_build_mini():
    model_file1 = "./model/tdnn_bb04p1_hua_shibie_all.onnx"
    input_shape1 = (1, 514)
    input_data1 = np.random.rand(150, 514) * 255  # 5-frame tdnn
    optimize_method_config1 = ""

    output_dir1 = "./output"
    array_distribute = ""
    start = time.time()
    build_run_model(model_file1, input_shape1, input_data1, optimize_method_config1, output_dir1)
    end = time.time()
    print("运行时间:%.2f秒" % (end - start))  #2.32

    # 运行精简版build，需要输入完整版生成的权重布局文件
    output_dir1 = "./output/output_mini"
    array_distribute = "./output/map/layers.txt"
    start = time.time()
    build_run_model(model_file1, input_shape1, input_data1, optimize_method_config1, output_dir1,
                    array_distribute)
    end = time.time()
    print("运行时间:%.2f秒" % (end - start))  #0.57


if __name__ == "__main__":
    np.random.seed(10)
    generate_tdnn_model()
    test_build_mini()