# =============================================================================
# Copyright(c) 2021 Witin Tech Corporation. All Rights Reserved.
# Author: shuaikai.liu
#
# Email: shuaikai.liu@witintech.com
#
# Last modified: 2023-06-07 14:27
#
# Filename: auto_optimizer_npu_build.py
# Description:
#
# =============================================================================
import numpy as np
import onnx
import os
import sys
import datetime
import witin
import google.protobuf
import google.protobuf.text_format
import logging
from onnx import helper, TensorProto as tp
from witin import *

from engine import optiConfig_pb2
from engine.auto_optimizer import Optimizer
import tvm.relay as witin_frontend
logging.getLogger().setLevel(logging.INFO)
np.random.seed(100)


def auto_optimizer_npu_graph_build(mods, params, inputs, optimize_config):
    target = 'npu'
    target_host = 'npu'
    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestamp
    optimize_path = ""
    if optimize_config and optimize_config.ByteSize() > 0:
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        optimize_path = os.path.join(build_dir, "auto_optimizer.protobuf")
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


def get_auto_optimizer_output(onnx_models, shape_dicts, mods, params, input_datas, manual_opt_conf):
    
    optimizer = Optimizer(onnx_models, shape_dicts, input_datas, manual_opt_conf)

    for conf in optimizer.get_optimize_configs():
        if conf.ByteSize() > 0:
            graphs_inputs = optimizer.update_graphs_inputs(conf, input_datas)
            runtime_model = auto_optimizer_npu_graph_build(mods, params, graphs_inputs, conf)
            assert runtime_model is not None
            runtime_model.run()



def build_run_model(mods, params, input_datas, manual_opt_conf, build_dir):
     
    #build
    with witin.transform.PassContext(opt_level=3):
        _,_,_,npu_graph = witin_frontend.build_module.build(mods,
                                                              target='npu',
                                                              target_host='npu',
                                                              params=params,
                                                              input_data=input_datas,
                                                              chip='BB04P1',
                                                              output_dir=build_dir,
                                                              optimize_method_config=manual_opt_conf)
          
    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    count = 0
    for i in range(len(input_datas)):
        for value in input_datas[i].values():
            m.set_input(count, witin.nd.array(value))
            count = count + 1
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)

