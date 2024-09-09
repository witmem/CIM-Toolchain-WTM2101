# =============================================================================
# Copyright(c) 2021 Witin Tech Corporation. All Rights Reserved.
# Author: jianchao.hu
#
# Email: jianchao.hu@witintech.com
#
# Last modified: 2022-09-07 13:58
#
# Filename: test_forward_op_output_repeat.py
#
# Description:
#
# =============================================================================*/
import datetime
import logging
import numpy as np
import onnx
import os
from onnx import helper, TensorProto as tp
import witin
from witin import *
import tvm.relay as witin_frontend
logging.getLogger().setLevel(logging.INFO)


def npu_graph_build(mods, params, inputs, optimize_config):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestamp

    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(
            mods,
            target=target,
            target_host=target_host,
            params=params,
            input_data=inputs,
            optimize_method_config=optimize_config,
            output_dir=build_dir,
            chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    for input_dict in inputs:
        for key, data in input_dict.items():
            m.set_input(key, data)
    return m


def get_tvm_output(model_path):
    """ Generic function to execute and get tvm output"""
    inputs = []
    mods = []
    params = []
    onnx_models = []

    onnx_model = onnx.load(model_path)
    shape_dict = {}
    shape_dict['in'] = (1, 40)
    mod, param = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    feats = np.loadtxt('./model/730/calibration_data/Rec_010.txt', skiprows=70, max_rows=30).astype("float32")
    data1 = feats[0:30][:]
    input_dt = {}
    input_dt['in'] = witin.nd.array(data1)
    mods.append(mod)
    params.append(param)
    inputs.append(input_dt)
    onnx_models.append(onnx_model)

    optimize_config = "./model/optimize_layer/op_output_repeat.protobuf"
    runtime_model = npu_graph_build(mods, params, inputs, optimize_config)
    runtime_model.run()


def generate_tdnn_model():
    weight0 = np.random.randint(-44, 44, size=(440, 448), dtype=np.int32).astype(np.float32)
    weight1 = np.random.randint(-44, 44, size=(448, 448), dtype=np.int32).astype(np.float32)
    weight2 = np.random.randint(-44, 44, size=(448, 448), dtype=np.int32).astype(np.float32)
    weight3 = np.random.randint(-44, 44, size=(448, 448), dtype=np.int32).astype(np.float32)
    weight4 = np.random.randint(-44, 44, size=(448, 128), dtype=np.int32).astype(np.float32)

    bias0 = 128 * np.random.randint(-128, 127, size=(448,), dtype=np.int32).astype(np.float32)
    bias1 = 128 * np.random.randint(-128, 127, size=(448,), dtype=np.int32).astype(np.float32)
    bias2 = 128 * np.random.randint(-128, 127, size=(448,), dtype=np.int32).astype(np.float32)
    bias3 = 128 * np.random.randint(-128, 127, size=(448,), dtype=np.int32).astype(np.float32)
    bias4 = 128 * np.random.randint(-128, 127, size=(128,), dtype=np.int32).astype(np.float32)
    offsets_op1 = helper.make_tensor(
        "op1_offset", data_type=tp.FLOAT, dims=(11,),
        vals=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    tdnn_wt1 = helper.make_tensor(
        "op1_wt", data_type=tp.FLOAT, dims=(440, 448),
        vals=weight0.flatten())
    dnn_wt2 = helper.make_tensor(
        "op2_wt", data_type=tp.FLOAT, dims=(448, 448),
        vals=weight1.flatten())
    dnn_wt3 = helper.make_tensor(
        "op3_wt", data_type=tp.FLOAT, dims=(448, 448),
        vals=weight2.flatten())
    dnn_wt4 = helper.make_tensor(
        "op4_wt", data_type=tp.FLOAT, dims=(448, 448),
        vals=weight3.flatten())
    dnn_wt5 = helper.make_tensor(
        "op5_wt", data_type=tp.FLOAT, dims=(448, 128),
        vals=weight4.flatten())

    tdnn_bias1 = helper.make_tensor(
        "op1_bias", data_type=tp.FLOAT, dims=(448,), vals=bias0.flatten())
    dnn_bias2 = helper.make_tensor(
        "op2_bias", data_type=tp.FLOAT, dims=(448,), vals=bias1.flatten())
    dnn_bias3 = helper.make_tensor(
        "op3_bias", data_type=tp.FLOAT, dims=(448,), vals=bias2.flatten())
    dnn_bias4 = helper.make_tensor(
        "op4_bias", data_type=tp.FLOAT, dims=(448,), vals=bias3.flatten())
    dnn_bias5 = helper.make_tensor(
        "op5_bias", data_type=tp.FLOAT, dims=(128,), vals=bias4.flatten())

    op1 = helper.make_node(
        'Tdnn', inputs=['in', 'op1_wt'], outputs=['tdnn1'],
        time_offsets=offsets_op1, bias_params=tdnn_bias1,
        scale_params=512, name="tdnn_op1")

    relu_op1 = helper.make_node('Relu', inputs=['tdnn1'],
                                outputs=['relu_out1'])

    op2 = helper.make_node('Gemm', inputs=['relu_out1', 'op2_wt', 'op2_bias'],
                           outputs=['tdnn2'], name="dense_op1")
    mul_op1 = helper.make_node('Scale', ['tdnn2'], ['mul_op_out1'], scale=0.001953125)
    relu_op2 = helper.make_node('Relu', ['mul_op_out1'], ['relu_out2'])

    op3 = helper.make_node('Gemm', inputs=['relu_out2', 'op3_wt', 'op3_bias'],
                           outputs=['tdnn3'], name="dense_op2")
    mul_op2 = helper.make_node('Scale', ['tdnn3'], ['mul_op_out2'], scale=0.001953125)
    relu_op3 = helper.make_node('Relu', ['mul_op_out2'], ['relu_out3'])

    op4 = helper.make_node('Gemm', inputs=['relu_out3', 'op4_wt', 'op4_bias'],
                           outputs=['tdnn4'], name="dense_op3")
    mul_op3 = helper.make_node('Scale', ['tdnn4'], ['scale_out3'], scale=0.001953125)
    relu_op4 = helper.make_node('Relu', ['scale_out3'], ['relu_out4'])

    op5 = helper.make_node('Gemm', inputs=['relu_out4', 'op5_wt', 'op5_bias'],
                           outputs=['dense_out'], name="dense_op4")
    mul_op4 = helper.make_node('Scale', ['dense_out'], ['out'], scale=0.001953125)

    in_shape = (1, 40)
    out_shape = (1, 128)
    initializer = [offsets_op1, tdnn_wt1, dnn_wt2, dnn_wt3,
                   dnn_wt4, dnn_wt5, tdnn_bias1, dnn_bias2,
                   dnn_bias3, dnn_bias4, dnn_bias5]

    graph = helper.make_graph(
        [op1, relu_op1, op2, mul_op1, relu_op2, op3, mul_op2, relu_op3, op4,
         mul_op3, relu_op4, op5, mul_op4],
        "auto_optimizer",
        inputs=[helper.make_tensor_value_info("in", tp.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", tp.FLOAT, list(out_shape))],
        initializer=initializer)

    model = helper.make_model(graph, producer_name='auto_optimizer')
    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/output_repeat_tdnn_dnn_" + timestamp + ".onnx"
    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())
    logging.info("generate optimizer.onnx sucessfully!")
    return model_path


def test_model(model_path):
    get_tvm_output(model_path)


def test_forward_output_repeat_multi_tdnn_dnn():
    np.random.seed(10)
    model_path = generate_tdnn_model()
    test_model(model_path)


if __name__ == '__main__':
    test_forward_output_repeat_multi_tdnn_dnn()
