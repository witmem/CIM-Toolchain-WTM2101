#p. Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import onnx
from onnx import helper
import pytest
import os
import datetime
import witin
from witin import *

np.random.seed(100)


def get_tvm_output(graph_def, input_shape, data, opt_config, threshold=0.99):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, input_shape)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target=target,
                                                               target_host=target_host,
                                                               params=params,
                                                               input_data=input_dt,
                                                               optimize_method_config=opt_config,
                                                               output_dir=build_dir,
                                                               chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    # execute
    m.run(threshold)
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list)


def generate_dnn_model(min, max, size):
    gemm_params_1 = np.random.randint(min, max, size=size, dtype=np.int32).astype(np.float32)
    gemm_bias_1 = 128 * np.random.randint(-128, 127, size=(size[1],), dtype=np.int32).astype(
        np.float32)

    gemm_params1 = onnx.helper.make_tensor("gemm_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=size,
                                           vals=gemm_params_1.flatten())

    gemm_bias1 = onnx.helper.make_tensor("gemm_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(size[1],),
                                         vals=gemm_bias_1.flatten())

    gemm_node1 = onnx.helper.make_node('Gemm',
                                       inputs=['in', 'gemm_params1', 'gemm_bias1'],
                                       outputs=['gemm_out1'],
                                       name="gemm_node1")
    gemm_mul_node1 = onnx.helper.make_node('Scale', ['gemm_out1'], ['gemm_mul_node_out1'],
                                           scale=0.0009765625)

    gemm_relu_node1 = onnx.helper.make_node('Relu', ['gemm_mul_node_out1'], ['out'])

    in_shape = (1, size[0])
    out_shape = (1, size[1])
    initializer = [gemm_params1, gemm_bias1]

    graph = onnx.helper.make_graph(
        [gemm_node1, gemm_mul_node1, gemm_relu_node1],
        "sparse_matrix",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='sparse_matrix')
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"
    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate sparse_matrix.onnx sucessfully!")
    return model_path


def generate_tdnn_model(min, max, size, dims, offset_vals):
    tdnn_weight_params = np.random.randint(min, max, size=size, dtype=np.int32).astype(np.float32)
    tdnn_bias_params = np.random.randint(-128, 127, size=(size[1],),
                                         dtype=np.int32).astype(np.float32) * 128

    offset_layer1 = helper.make_tensor("offset_layer1",
                                       data_type=onnx.TensorProto.FLOAT,
                                       dims=(dims,),
                                       vals=offset_vals)
    tdnn_weight = helper.make_tensor("tdnn_weight",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=size,
                                     vals=tdnn_weight_params.flatten())
    tdnn_bias = helper.make_tensor("tdnn_bias",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(size[1],),
                                   vals=tdnn_bias_params.flatten())

    tdnn_node = helper.make_node("Tdnn",
                                 inputs=['in', 'tdnn_weight', 'tdnn_bias'],
                                 outputs=['tdnn_out'],
                                 time_offsets=offset_layer1,
                                 bias_params=tdnn_bias,
                                 scale_params=1024,
                                 name="tdnn_node")
    relu_node = helper.make_node("Relu", ['tdnn_out'], ['out'])

    in_shape = (1, int(size[0] / dims))
    out_shape = (1, size[1])
    initializer = [tdnn_weight, offset_layer1, tdnn_bias]
    graph = helper.make_graph(
        [tdnn_node, relu_node],
        "dnn_tdnn_sparse_matrix",
        inputs=[helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
        initializer=initializer)

    model = helper.make_model(graph=graph, producer_name='tdnn_sparse_matrix')
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"

    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate %s sucessfully!"%model_path)
    return model_path


def generate_gru_model(array1_min_max, array2_min_max, array1_shape, array2_shape):
    gru_weight1_params = np.random.randint(array1_min_max[0],
                                           array1_min_max[1],
                                           size=array1_shape,
                                           dtype=np.int32).astype(np.float32)
    gru_weight2_params = np.random.randint(array2_min_max[0],
                                           array2_min_max[1],
                                           size=array2_shape,
                                           dtype=np.int32).astype(np.float32)

    gru_bias1_params = 128 * np.random.randint(-128, 127, size=(array1_shape[1],),
                                               dtype=np.int32).astype(np.float32)
    gru_bias2_params = 128 * np.random.randint(-128, 127, size=(array2_shape[1],),
                                               dtype=np.int32).astype(np.float32)

    gru_weight1 = helper.make_tensor("gru_weight1",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=array1_shape,
                                     vals=gru_weight1_params.flatten())
    gru_weight2 = helper.make_tensor("gru_weight2",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=array2_shape,
                                     vals=gru_weight2_params.flatten())

    gru_bias1 = helper.make_tensor("gru_bias1",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(array1_shape[1],),
                                   vals=gru_bias1_params.flatten())
    gru_bias2 = helper.make_tensor("gru_bias2",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(array2_shape[1],),
                                   vals=gru_bias2_params.flatten())

    act_table_list_1 = [i for i in range(128)]  #0~127
    act_table_list_2 = [i - 128 for i in range(128)]  #-128~-1
    act_table_list = act_table_list_1 + act_table_list_2
    act_table_list = act_table_list + act_table_list
    act_table = np.array(act_table_list)
    act_table = np.reshape(act_table, (2, 256)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 256),
                                        vals=act_table.flatten())

    gru_node = helper.make_node(
        "Gru",
        inputs=['in', 'gru_weight1', 'gru_weight2', 'gru_bias1', 'gru_bias2'],
        outputs=['out'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[-7, -7],
        name='gru_node')

    in_shape = (1, array2_shape[1])
    out_shape = (1, array2_shape[1])
    initializer = [gru_weight1, gru_weight2, gru_bias1, gru_bias2, act_table]

    graph = helper.make_graph(
        [gru_node],
        "gru_sparse_matrix",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)

    model = helper.make_model(graph, producer_name='gru_sparse_matrix')
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"

    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate gru_sparse_matrix.onnx sucessfully!")
    return model_path


def generate_lstm_model():
    lstm_weight_1 = np.random.randint(-200, 200, size=(8, 12), dtype=np.int32).astype(np.float32)
    lstm_weight_2 = np.random.randint(-200, 200, size=(8, 4), dtype=np.int32).astype(np.float32)

    lstm_bias_1 = 128 * np.random.randint(-128, 127, size=(12,), dtype=np.int32).astype(np.float32)
    lstm_bias_2 = 218 * np.random.randint(-128, 127, size=(4,), dtype=np.int32).astype(np.float32)

    lstm_weight1 = onnx.helper.make_tensor("lstm_weight1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(8, 12),
                                           vals=lstm_weight_1.flatten())
    lstm_weight2 = onnx.helper.make_tensor("lstm_weight2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(8, 4),
                                           vals=lstm_weight_2.flatten())
    lstm_bias1 = onnx.helper.make_tensor("lstm_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(12,),
                                         vals=lstm_bias_1.flatten())
    lstm_bias2 = onnx.helper.make_tensor("lstm_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(4,),
                                         vals=lstm_bias_2.flatten())
    act_table = np.random.randint(-50, 50, size=(3, 256), dtype=np.int32).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(3, 256),
                                        vals=act_table.flatten())
    lstm_node = helper.make_node(
        "Lstm",
        inputs=['in', 'lstm_weight1', 'lstm_weight2', 'lstm_bias1', 'lstm_bias2'],
        scale_ioft=1024,
        scale_ct=1024,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=act_table,
        outputs=['out'],
        name='lstm_node')

    in_shape = (1, 4)
    out_shape = (1, 4)
    initializer = [lstm_weight1, lstm_weight2, lstm_bias1, lstm_bias2, act_table]
    graph = helper.make_graph(
        [lstm_node],
        "lstm_sparse_matrix",
        inputs=[helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='lstm_sparse_matrix')
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"

    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate lstm_sparse_matrix.onnx sucessfully!")
    return model_path


def generate_dnn_dnn_model(dnn_w1_min_max, dnn_w2_min_max, dnn_w1_shape, dnn_w2_shape):
    dnn1_weight_params = np.random.randint(dnn_w1_min_max[0],
                                           dnn_w1_min_max[1],
                                           size=dnn_w1_shape,
                                           dtype=np.int32).astype(np.float32)
    dnn2_weight_params = np.random.randint(dnn_w2_min_max[0],
                                           dnn_w2_min_max[1],
                                           size=dnn_w2_shape,
                                           dtype=np.int32).astype(np.float32)
    dnn1_bias_params = 128 * np.random.randint(-128, 127, size=(dnn_w1_shape[1],),
                                               dtype=np.int32).astype(np.float32)
    dnn2_bias_params = 128 * np.random.randint(-128, 127, size=(dnn_w2_shape[1],),
                                               dtype=np.int32).astype(np.float32)

    dnn1_weight = helper.make_tensor("dnn1_weight",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=dnn_w1_shape,
                                     vals=dnn1_weight_params.flatten())
    dnn1_bias = helper.make_tensor("dnn1_bias",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(dnn_w1_shape[1],),
                                   vals=dnn1_bias_params.flatten())
    dnn2_weight = helper.make_tensor("dnn2_weight",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=dnn_w2_shape,
                                     vals=dnn2_weight_params.flatten())
    dnn2_bais = helper.make_tensor("dnn2_bias",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(dnn_w2_shape[1],),
                                   vals=dnn2_bias_params.flatten())

    dnn_node1 = helper.make_node("Gemm",
                                 inputs=['in', 'dnn1_weight', 'dnn1_bias'],
                                 outputs=['dnn_node1_out'],
                                 name="dnn_node1")
    dnn1_mul_node = helper.make_node("Scale", ['dnn_node1_out'], ['dnn1_mul_node_out'],
                                     scale=0.0009765625)
    dnn1_relu_node = helper.make_node("Relu", ['dnn1_mul_node_out'], ['dnn1_relu_node_out'])

    dnn_node2 = helper.make_node("Gemm",
                                 inputs=['dnn1_relu_node_out', 'dnn2_weight', 'dnn2_bias'],
                                 outputs=['dnn_node2_out'],
                                 name='dnn_node2')
    dnn2_mul_node = helper.make_node("Scale", ['dnn_node2_out'], ['dnn2_mul_node_out'],
                                     scale=0.0009765625)
    dnn2_relu_node = helper.make_node("Relu", ['dnn2_mul_node_out'], ['out'])

    in_shape = (1, dnn_w1_shape[0])
    out_shape = (1, dnn_w2_shape[1])
    initializer = [dnn1_weight, dnn2_weight, dnn1_bias, dnn2_bais]
    graph = helper.make_graph(
        [dnn_node1, dnn1_mul_node, dnn1_relu_node, dnn_node2, dnn2_mul_node, dnn2_relu_node],
        "dnn_dnn_sparse_matrix",
        inputs=[helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
        initializer=initializer)

    model = helper.make_model(graph=graph, producer_name="dnn_dnn_sparse_matrix")
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"

    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate dnn_dnn_sparse_matrix.onnx sucessfully!")
    return model_path


def generate_dnn_gru_model(dnn_min_max, arr1_min_max, arr2_min_max, dnn_shape, arr1_shape,
                           arr2_shape):
    dnn_weight_params = np.random.randint(dnn_min_max[0],
                                          dnn_min_max[1],
                                          size=dnn_shape,
                                          dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128, 127, size=(dnn_shape[1],),
                                              dtype=np.int32).astype(np.float32)

    gru_weight1_params = np.random.randint(arr1_min_max[0],
                                           arr1_min_max[1],
                                           size=arr1_shape,
                                           dtype=np.int32).astype(np.float32)
    gru_weight2_params = np.random.randint(arr2_min_max[0],
                                           arr2_min_max[1],
                                           size=arr2_shape,
                                           dtype=np.int32).astype(np.float32)
    gru_bias1_params = 128 * np.random.randint(-128, 127, size=(arr1_shape[1],),
                                               dtype=np.int32).astype(np.float32)
    gru_bias2_params = 128 * np.random.randint(-128, 127, size=(arr2_shape[1],),
                                               dtype=np.int32).astype(np.float32)

    dnn_weight = helper.make_tensor("dnn_weight",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=dnn_shape,
                                    vals=dnn_weight_params.flatten())
    dnn_bias = helper.make_tensor("dnn_bias",
                                  data_type=onnx.TensorProto.FLOAT,
                                  dims=(dnn_shape[1],),
                                  vals=dnn_bias_params.flatten())
    gru_weight1 = helper.make_tensor("gru_weight1",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=arr1_shape,
                                     vals=gru_weight1_params.flatten())
    gru_weight2 = helper.make_tensor("gru_weight2",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=arr2_shape,
                                     vals=gru_weight2_params.flatten())
    gru_bias1 = helper.make_tensor("gru_bias1",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(arr1_shape[1],),
                                   vals=gru_bias1_params.flatten())
    gru_bias2 = helper.make_tensor("gru_bias2",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(arr2_shape[1],),
                                   vals=gru_bias2_params.flatten())

    act_table_list_1 = [i for i in range(128)]  #0~127
    act_table_list_2 = [i - 128 for i in range(128)]  #-128~-1
    act_table_list = act_table_list_1 + act_table_list_2
    act_table_list = act_table_list + act_table_list
    act_table = np.array(act_table_list)
    act_table = np.reshape(act_table, (2, 256)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 256),
                                        vals=act_table.flatten())

    dnn_node = helper.make_node("Gemm",
                                inputs=['in', 'dnn_weight', 'dnn_bias'],
                                outputs=['dnn_out'],
                                name='dnn_node')

    gru_node = helper.make_node(
        "Gru",
        inputs=['dnn_out', 'gru_weight1', 'gru_weight2', 'gru_bias1', 'gru_bias2'],
        outputs=['out'],
        scale_ht=1024,
        scale_zr=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[-7, -7],
        name="gru_node")

    in_shape = (1, dnn_shape[0])
    out_shape = (1, arr2_shape[1])
    initializer = [dnn_weight, dnn_bias, gru_weight1, gru_weight2, gru_bias1, gru_bias2, act_table]

    graph = helper.make_graph(
        nodes=[dnn_node, gru_node],
        name="dnn_gru_sparse_matrix",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)

    model = helper.make_model(graph, producer_name='dnn_gru_sparse_matrix')
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"

    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate dnn_gru_sparse_matrix.onnx sucessfully!")
    return model_path


def generate_gru_dnn_model(arr1_min_max, arr2_min_max, dnn_min_max, arr1_shape, arr2_shape,
                           dnn_shape):
    dnn_weight_params = np.random.randint(dnn_min_max[0],
                                          dnn_min_max[1],
                                          size=dnn_shape,
                                          dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128, 127, size=(dnn_shape[1],),
                                              dtype=np.int32).astype(np.float32)
    gru_weight1_params = np.random.randint(arr1_min_max[0],
                                           arr1_min_max[1],
                                           size=arr1_shape,
                                           dtype=np.int32).astype(np.float32)
    gru_weight2_params = np.random.randint(arr2_min_max[0],
                                           arr2_min_max[1],
                                           size=arr2_shape,
                                           dtype=np.int32).astype(np.float32)
    gru_bias1_params = 128 * np.random.randint(-128, 127, size=(arr1_shape[1],),
                                               dtype=np.int32).astype(np.float32)
    gru_bias2_params = 128 * np.random.randint(-128, 127, size=(arr2_shape[1],),
                                               dtype=np.int32).astype(np.float32)

    gru_weight1 = helper.make_tensor("gru_weight1",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=arr1_shape,
                                     vals=gru_weight1_params.flatten())
    gru_weight2 = helper.make_tensor("gru_weight2",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=arr2_shape,
                                     vals=gru_weight2_params.flatten())
    gru_bias1 = helper.make_tensor("gru_bias1",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(arr1_shape[1],),
                                   vals=gru_bias1_params.flatten())
    gru_bias2 = helper.make_tensor("gru_bias2",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(arr2_shape[1],),
                                   vals=gru_bias2_params.flatten())
    dnn_weight = helper.make_tensor("dnn_weight",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=dnn_shape,
                                    vals=dnn_weight_params.flatten())
    dnn_bias = helper.make_tensor("dnn_bias",
                                  data_type=onnx.TensorProto.FLOAT,
                                  dims=(dnn_shape[1],),
                                  vals=dnn_bias_params.flatten())

    act_table_list_1 = [i for i in range(128)]  #0~127
    act_table_list_2 = [i - 128 for i in range(128)]  #-128~-1
    act_table_list = act_table_list_1 + act_table_list_2
    act_table_list = act_table_list + act_table_list
    act_table = np.array(act_table_list)
    act_table = np.reshape(act_table, (2, 256)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 256),
                                        vals=act_table.flatten())

    gru_node = helper.make_node(
        "Gru",
        inputs=['in', 'gru_weight1', 'gru_weight2', 'gru_bias1', 'gru_bias2'],
        outputs=['gru_out'],
        scale_ht=1024,
        scale_zr=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[0, -4],
        name="gru_node")

    dnn_node = helper.make_node("Gemm",
                                inputs=['gru_out', 'dnn_weight', 'dnn_bias'],
                                outputs=['out'],
                                name='dnn_node')

    in_shape = (1, arr2_shape[1])
    out_shape = (1, dnn_shape[1])
    initializer = [dnn_weight, dnn_bias, gru_weight1, gru_weight2, gru_bias1, gru_bias2, act_table]

    graph = helper.make_graph(
        nodes=[gru_node, dnn_node],
        name='gru_dnn_sparse_matrix',
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)

    model = helper.make_model(graph, producer_name='gru_dnn_sparse_matrix')
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"

    with open(model_path, 'wb') as of:
        of.write(model.SerializeToString())

    print("generate gru_dnn_sparse_matrix.onnx sucessfully!")
    return model_path


def generate_dnn_dnn_dnn_model(w1_min_max, w2_min_max, w3_min_max, w1_shape, w2_shape, w3_shape):
    dnn1_weight_params = np.random.randint(w1_min_max[0],
                                           w1_min_max[1],
                                           size=w1_shape,
                                           dtype=np.int32).astype(np.float32)
    dnn2_weight_params = np.random.randint(w2_min_max[0],
                                           w2_min_max[1],
                                           size=w2_shape,
                                           dtype=np.int32).astype(np.float32)
    dnn3_weight_params = np.random.randint(w3_min_max[0],
                                           w3_min_max[1],
                                           size=w3_shape,
                                           dtype=np.int32).astype(np.float32)
    dnn1_bias_params = 128 * np.random.randint(-128, 127, size=(w1_shape[1],),
                                               dtype=np.int32).astype(np.float32)
    dnn2_bias_params = 128 * np.random.randint(-128, 127, size=(w2_shape[1],),
                                               dtype=np.int32).astype(np.float32)
    dnn3_bias_params = 128 * np.random.randint(-128, 127, size=(w3_shape[1],),
                                               dtype=np.int32).astype(np.float32)

    dnn1_weight = helper.make_tensor("dnn1_weight",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=w1_shape,
                                     vals=dnn1_weight_params.flatten())
    dnn1_bias = helper.make_tensor("dnn1_bias",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(w1_shape[1],),
                                   vals=dnn1_bias_params.flatten())
    dnn2_weight = helper.make_tensor("dnn2_weight",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=w2_shape,
                                     vals=dnn2_weight_params.flatten())
    dnn2_bias = helper.make_tensor("dnn2_bias",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(w2_shape[1],),
                                   vals=dnn2_bias_params.flatten())
    dnn3_weight = helper.make_tensor("dnn3_weight",
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=w3_shape,
                                     vals=dnn3_weight_params.flatten())
    dnn3_bias = helper.make_tensor("dnn3_bias",
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=(w3_shape[1],),
                                   vals=dnn3_bias_params.flatten())

    dnn_node1 = helper.make_node("Gemm",
                                 inputs=['in', 'dnn1_weight', 'dnn1_bias'],
                                 outputs=['dnn_node1_out'],
                                 name="dnn_node1")
    dnn1_relu_node = helper.make_node("Relu", ['dnn_node1_out'], ['dnn1_relu_node_out'])

    dnn_node2 = helper.make_node("Gemm",
                                 inputs=['dnn1_relu_node_out', 'dnn2_weight', 'dnn2_bias'],
                                 outputs=['dnn_node2_out'],
                                 name='dnn_node2')
    dnn2_relu_node = helper.make_node("Relu", ['dnn_node2_out'], ['dnn2_relu_node_out'])

    dnn_node3 = helper.make_node("Gemm",
                                 inputs=['dnn2_relu_node_out', 'dnn3_weight', 'dnn3_bias'],
                                 outputs=['dnn_node3_out'],
                                 name='dnn_node3')
    dnn3_relu_node = helper.make_node("Relu", ['dnn_node3_out'], ['out'])

    in_shape = (1, w1_shape[0])
    out_shape = (1, w3_shape[1])
    initializer = [dnn1_weight, dnn2_weight, dnn3_weight, dnn3_bias, dnn1_bias, dnn2_bias]
    graph = helper.make_graph(
        [dnn_node1, dnn1_relu_node, dnn_node2, dnn2_relu_node, dnn_node3, dnn3_relu_node],
        "dnn_dnn_dnn_sparse_matrix",
        inputs=[helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
        initializer=initializer)

    model = helper.make_model(graph=graph, producer_name="dnn_dnn_dnn_sparse_matrix")
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix" + timestampStr + ".onnx"

    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate dnn_dnn_dnn_sparse_matrix.onnx sucessfully!")
    return model_path


dnn_spaese_matrix_params = [
    # 2bit
    (-300, 300, (16, 16)),
    (-200, 200, (64, 16)),
    (-200, 200, (48, 48)),
    (-400, 400, (32, 16)),
    (-500, 380, (32, 16)),
    # 4bit
    (-900, 900, (64, 16)),
    (-1100, 1100, (64, 16)),
    (-1300, 1300, (16, 16)),
    (-1500, 1500, (32, 16)),
    (-1700, 1700, (32, 32)),
    (-1900, 1900, (16, 16)),
    # 8 bit
    (-5000, 5000, (32, 16)),
    (-10000, 10000, (32, 16)),
    (-11000, 11000, (16, 16)),
    (-12000, 12000, (32, 16)),
    (-13000, 13000, (16, 8)),
    (-14000, 14000, (16, 16)),
    (-14000, 14000, (16, 16)),
    (-15000, 15000, (16, 16)),
    (-20000, 20000, (16, 8)),
    (-30000, 30000, (16, 8)),
    (-30000, 30000, (16, 16)),
]


# test dnn sparse_martix optimize, all tests OK!
@pytest.mark.parametrize("min,max,size", dnn_spaese_matrix_params)
def test_dnn_model(min, max, size):
    np.random.seed(10)
    # 2 、4 、8bit
    file_name = generate_dnn_model(min, max, size)
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, size[0]) * 255).astype("float32")
    opt_config = './model/optimize_layer/optimize_config_sparse_matrix.protobuf'
    tvm_out = get_tvm_output(onnx_model, size[0], data, opt_config)


tdnn_sparse_matrix_params = [
    # 2bit
    (-200, 200, (80, 40), 5, [-2, -1, 0, 1, 2]),
    (-200, 200, (90, 40), 3, [-1, 0, 1]),
    (-400, 400, (42, 8), 7, [-3, -2, -1, 0, 1, 2, 3]),
    (-500, 380, (12, 4), 3, [-1, 0, 1]),
    # 4bit
    (-900, 900, (20, 4), 5, [-2, -1, 0, 1, 2]),
    (-1300, 1300, (70, 10), 7, [-3, -2, -1, 0, 1, 2, 3]),
    (-1500, 1500, (40, 8), 5, [-2, -1, 0, 1, 2]),
    (-1900, 1900, (12, 4), 3, [-1, 0, 1]),
    # 8bit
    (-5000, 5000, (20, 4), 5, [-2, -1, 0, 1, 2]),
    (-10000, 10000, (70, 10), 7, [-3, -2, -1, 0, 1, 2, 3]),
    (-20000, 20000, (40, 8), 5, [-2, -1, 0, 1, 2]),
    (-30000, 30000, (12, 4), 3, [-1, 0, 1]),
]


# test tdnn optimize sparse_matrix, all tests OK!
@pytest.mark.parametrize("min,max,size,dims,offset_vals", tdnn_sparse_matrix_params)
def test_tdnn_model(min, max, size, dims, offset_vals):
    np.random.seed(10)
    file_name = generate_tdnn_model(min, max, size, dims, offset_vals)
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(20, int(size[0])) * 255).astype("float32")
    opt_config = './model/optimize_layer/optimize_config_sparse_matrix_tdnn.protobuf'
    tvm_out = get_tvm_output(onnx_model, int(size[0] / dims), data, opt_config)


gru_sparse_matrix_params = [
    ((-200, 200), (-200, 200), (32, 32), (32, 16)),
    ((-350, 350), (-400, 400), (32, 32), (32, 16)),
    ((-500, 500), (-500, 500), (32, 32), (32, 16)),
    ((-1300, 1300), (-1350, 1350), (16, 16), (16, 8)),
    ((-1500, 1500), (-1500, 1500), (8, 8), (8, 4)),
    ((-5000, 5000), (-5000, 5000), (8, 8), (8, 4)),
    ((-10000, 10000), (-10000, 10000), (8, 8), (8, 4)),
    ((-25000, 25000), (-30000, 30000), (8, 8), (8, 4)),
]


# test gru optimize sparse_matrix, all tests pass!!!!!!!!! But the shape of array cann't to big if arrays's weight is to big
@pytest.mark.parametrize("array1_min_max,array2_min_max,array1_shape,array2_shape",
                         gru_sparse_matrix_params)
def test_gru_model(array1_min_max, array2_min_max, array1_shape, array2_shape):
    np.random.seed(10)
    file_name = generate_gru_model(array1_min_max, array2_min_max, array1_shape, array2_shape)
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, array2_shape[1]) * 255).astype("float32")
    opt_config = './model/optimize_layer/optimize_config_sparse_matrix_gru.protobuf'
    '''
    For case test_gru_model((-5000, 5000), (-5000, 5000), (8, 8), (8, 4)), the similarity is low than 0.99
    reason: when sparseMatrixOpti field is MIX_MODE, the weight/bias will be splitted into high and low bits and do the calculation dependently, so there exist the loss of precision for the div operation, the inference formula is as below:
      y=(w*x+b)/G => 
      y=((w1+w2)*x+(b1+b2))/G =>
      y=((w1*x+b1)+(w2*x+b2))/G =>
      y=(w1*x+b1)/G+(w2*x+b2)/G
    '''
    threshold = 0.57
    tvm_out = get_tvm_output(onnx_model, array2_shape[1], data, opt_config, threshold)

# test lstm optimize sparse_matrix, have some problem, simulator haven't support lstm compare
# def test_lstm_model():
#     generate_lstm_model()
#     file_name = './model/lstm_sparse_matrix.onnx'
#     onnx_model = onnx.load(file_name)
#     data=np.round(np.random.rand(11, 4)*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_lstm.protobuf'
#     tvm_out = get_tvm_output(onnx_model,4,data,opt_config)

dnn_dnn_sparse_matrix_params = [
    # 2bit
    ((-200, 200), (-200, 200), (32, 16), (16, 4)),
    ((-300, 300), (-300, 300), (16, 8), (8, 4)),
    ((-500, 500), (-500, 500), (32, 8), (8, 4)),
    # 4bit
    ((-900, 900), (-900, 900), (36, 32), (32, 4)),
    ((-1500, 1500), (-1500, 1500), (36, 32), (32, 16)),
    ((-1900, 1900), (-1900, 1900), (36, 32), (32, 16)),
    # 8bit
    ((-5000, 5000), (-5000, 5000), (32, 16), (16, 4)),
    ((-10000, 10000), (-10000, 10000), (16, 8), (8, 4)),
    ((-30000, 30000), (-30000, 30000), (32, 8), (8, 4)),
]


# test dnn+dnn optimize sparse_matrix, all test pass!!!
@pytest.mark.parametrize("dnn_w1_min_max,dnn_w2_min_max,dnn_w1_shape,dnn_w2_shape",
                         dnn_dnn_sparse_matrix_params)
def test_dnn_dnn_model(dnn_w1_min_max, dnn_w2_min_max, dnn_w1_shape, dnn_w2_shape):
    np.random.seed(10)
    file_name = generate_dnn_dnn_model(dnn_w1_min_max, dnn_w2_min_max, dnn_w1_shape, dnn_w2_shape)
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, dnn_w1_shape[0]) * 255).astype("float32")
    opt_config = './model/optimize_layer/optimize_config_sparse_matrix_dnn_dnn.protobuf'
    tvm_out = get_tvm_output(onnx_model, dnn_w1_shape[0], data, opt_config)


dnn_gru_sparse_matrix_params = [
    ((-300, 300), (-300, 300), (-300, 300), (32, 16), (32, 32), (32, 16), "optimize_config_sparse_matrix_dnn_gru_1.protobuf"),
    # ((-500, 500), (-450, 450), (-500, 500), (32, 16), (32, 32), (32, 16)), // Invalid sparse matrix size: 4792 exceed the HW limit 4096
    # ((-1000, 1000), (-900, 900), (-1100, 1100), (32, 16), (32, 32), (32, 16)),
    ((-1500, 1500), (-1500, 1450), (-1400, 1350), (16, 4), (8, 8), (8, 4), "optimize_config_sparse_matrix_dnn_gru_2.protobuf"),
    # ((-10000, 10000), (-10000, 10000), (-10000, 10000), (8, 4), (8, 8), (8, 4)),
    ((-15000, 15000), (-20000, 20000), (-30000, 30000), (8, 4), (8, 8), (8, 4), "optimize_config_sparse_matrix_dnn_gru_3.protobuf"),
]


# all dnn + gru tests pass
@pytest.mark.parametrize("dnn_min_max, arr1_min_max, arr2_min_max, dnn_shape, arr1_shape, arr2_shape, opt_config",
                         dnn_gru_sparse_matrix_params)
def test_dnn_gru_model(dnn_min_max, arr1_min_max, arr2_min_max, dnn_shape, arr1_shape, arr2_shape, opt_config):
    np.random.seed(10)
    file_name = generate_dnn_gru_model(dnn_min_max, arr1_min_max, arr2_min_max, dnn_shape, arr1_shape, arr2_shape)
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, dnn_shape[0]) * 255).astype("float32")
    opt_config_path = './model/optimize_layer/' + opt_config
    tvm_out = get_tvm_output(onnx_model, dnn_shape[0], data, opt_config_path)


gru_dnn_sparse_matrix_params = [
    # ((-300, 300), (-300, 300), (-300, 300), (32, 32), (32, 16), (16, 8)),
    # ((-500, 500), (-450, 450), (-500, 500), (32, 32), (32, 16), (16, 8)),
    # ((-1000, 1000), (-900, 900), (-1100, 1100), (16, 16), (16, 8), (8, 8)),
    ((-1500, 1500), (-1500, 1450), (-1450, 1300), (8, 8), (8, 4), (4, 8)),
    ((-10000, 10000), (-10000, 10000), (-10000, 10000), (8, 8), (8, 4), (4, 4)),
    # ((-15000, 15000), (-20000, 20000), (-30000, 30000), (8, 8), (8, 4), (4, 4)),
]


# all gru + dnn tests pass !!!
@pytest.mark.parametrize("arr1_min_max,arr2_min_max,dnn_min_max,arr1_shape,arr2_shape,dnn_shape",
                         gru_dnn_sparse_matrix_params)
def test_gru_dnn_model(arr1_min_max, arr2_min_max, dnn_min_max, arr1_shape, arr2_shape, dnn_shape):
    np.random.seed(10)
    file_name = generate_gru_dnn_model(arr1_min_max, arr2_min_max, dnn_min_max, arr1_shape, arr2_shape,
                           dnn_shape)
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, arr2_shape[1]) * 255).astype("float32")
    opt_config = './model/optimize_layer/optimize_config_sparse_matrix_gru_dnn.protobuf'
    tvm_out = get_tvm_output(onnx_model, arr2_shape[1], data, opt_config)


dnn_x3_sparse_matrix_params = [
    # 2bit
    ((-200, 200), (-200, 200), (-200, 200), (64, 32), (32, 16), (16, 8)),
    ((-300, 300), (-300, 300), (-350, 350), (16, 8), (8, 4), (4, 8)),
    ((-500, 500), (-450, 450), (-400, 400), (32, 8), (8, 16), (16, 8)),
    # 4bit
    ((-890, 900), (-1000, 1000), (-1000, 1100), (32, 8), (8, 16), (16, 8)),
    ((-1500, 1500), (-1450, 1450), (-1500, 1600), (8, 8), (8, 16), (16, 8)),
    ((-1900, 1800), (-1700, 1800), (-1650, 1560), (32, 16), (16, 16), (16, 8)),
    # 8bit
    ((-5000, 5000), (-4350, 4350), (-3300, 3300), (8, 8), (8, 16), (16, 8)),
    ((-10000, 10000), (-12000, 11000), (-11000, 9800), (4, 8), (8, 16), (16, 8)),
    ((-30000, 30000), (-30000, 30000), (-20000, 20000), (4, 8), (8, 8), (8, 8)),
]


# test 3 x dnn optimize sparse_matrix, all tests pass!!!!!!!!
@pytest.mark.parametrize("w1_min_max,w2_min_max,w3_min_max,w1_shape,w2_shape,w3_shape",
                         dnn_x3_sparse_matrix_params)
def test_dnn_dnn_dnn_model(w1_min_max, w2_min_max, w3_min_max, w1_shape, w2_shape, w3_shape):
    np.random.seed(10)
    file_name = generate_dnn_dnn_dnn_model(w1_min_max, w2_min_max, w3_min_max, w1_shape, w2_shape, w3_shape)
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(20, w1_shape[0]) * 255).astype("float32")
    opt_config = './model/optimize_layer/optimize_config_sparse_matrix_dnn_dnn_dnn.protobuf'
    tvm_out = get_tvm_output(onnx_model, w1_shape[0], data, opt_config)


if __name__ == '__main__':
    np.random.seed(10)
    # test_tdnn_model(-30000, 30000, (12, 4), 3, [-1, 0, 1])
    # test_dnn_model(-900, 900, (64, 16))
    # test_gru_model()
    # test_lstm_model()
    # test_dnn_gru_model((-500, 500), (-450, 450), (-500, 500), (32, 16), (32, 32), (32, 16))
    # test_gru_dnn_model()
    # test_dnn_dnn_model()
    # test_dnn_dnn_dnn_model((-300, 300), (-300, 300), (-350, 350), (16, 8), (8, 4), (4, 8))
    test_dnn_model(-300, 300, (16, 16))
