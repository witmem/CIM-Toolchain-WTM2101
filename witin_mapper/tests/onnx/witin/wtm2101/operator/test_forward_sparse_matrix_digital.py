# Copyright(c) 2022 Witin Tech Corporation. All Rights Reserved.
# Author: shihang.fu
# Email: shihang.fu@witintech.com
# Last modified: 2022-04-20
# Filename: test_forward_sparse_matrix_di.py
# Description:
import numpy as np
import onnx
import os
import datetime
import witin
from witin import *

np.random.seed(100)

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def tanh_q(x, qx, qy):
    x = x / (2**qx)
    x = np.tanh(x)
    x = (x * (2**qy)).round().clip(-128, 127)
    return x


def sigmoid_q(x, qx, qy):
    x = x / (2**qx)
    x = sigmoid(x)
    x = (x * (2**qy)).round().clip(-128, 127)
    return x

def generate_lstm_activate_table():
    act_input256 = [i for i in range(128)] + [i - 128 for i in range(128)]
    act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
    act_input256 = np.array(act_input256)
    act_input1024 = np.array(act_input1024)

    sigmoid_table_list = sigmoid_q(act_input1024, 6, 7)
    tanh1_table_list = tanh_q(act_input1024, 8, 7)
    tanh2_table_list = tanh_q(act_input256, 7, 7)

    act_table1 = onnx.helper.make_tensor("act_table1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(1, 1024),
                                         vals=sigmoid_table_list.flatten())
    act_table2 = onnx.helper.make_tensor("act_table2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(1, 1024),
                                         vals=tanh1_table_list.flatten())
    act_table3 = onnx.helper.make_tensor("act_table3",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(1, 256),
                                         vals=tanh2_table_list.flatten())
    return act_table1, act_table2, act_table3

def get_tvm_output(graph_def):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 80)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    data1 = np.round(np.random.rand(40, 80) * 255).astype("float32")
    input_dt = {}
    input_dt['in'] = witin.nd.array(data1)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(
            mod,
            target=target,
            target_host=target_host,
            params=params,
            input_data=input_dt,
            optimize_method_config='./model/optimize_layer/optimize_config_sparse_matrix_digital.protobuf',
            output_dir=build_dir,
            chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data1))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]


def generate_onnx_model():
    gemm_params_1 = np.random.randint(-128, 127, size=(12, 80), dtype=np.int32).astype(np.float32)
    gemm_bias_1 = 128 * np.random.randint(-128, 127, size=(12,), dtype=np.int32).astype(np.float32)
    print(gemm_bias_1)

    gemm_params1 = onnx.helper.make_tensor("gemm_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(80, 12),
                                           vals=gemm_params_1.flatten())

    gemm_bias1 = onnx.helper.make_tensor("gemm_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(12,),
                                         vals=gemm_bias_1.flatten())

    gemm_node1 = onnx.helper.make_node('Gemm',
                                       inputs=['in', 'gemm_params1', 'gemm_bias1'],
                                       outputs=['gemm_out1'],
                                       name="gemm_node1")
    gemm_mul_node1 = onnx.helper.make_node('Scale', ['gemm_out1'], ['gemm_mul_node_out1'],
                                           scale=0.0009765625)

    gemm_relu_node1 = onnx.helper.make_node('Relu', ['gemm_mul_node_out1'], ['out'])

    in_shape = (1, 80)
    out_shape = (1, 12)
    initializer = [gemm_params1, gemm_bias1]

    graph = onnx.helper.make_graph(
        [gemm_node1, gemm_mul_node1, gemm_relu_node1],
        "sparse_matrix",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph, producer_name='sparse_matrix')
    with open('./model/sparse_matrix.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate sparse_matrix.onnx sucessfully!")


def test_model():
    file_name = './model/sparse_matrix.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    tvm_out = get_tvm_output(onnx_model)


def test_forward_sparse_matrix_digital():
    generate_onnx_model()
    test_model()

if __name__ == '__main__':
    generate_onnx_model()
    test_model()
