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
from onnx import helper, TensorProto
import datetime
import os
import pytest
import witin
from witin import *
np.random.seed(10)

def generate_tdnn_model():
    '''
    NOTE: Some layer's previous node not have relu,so the bias may too big.

    In(514*1)
        |
    [0]Gemm(514*128) Scale
        |
    [1]Tdnn(384*128) Relu
    |           |
    |        [2]Gemm(128*64) Scale
    |            |
    |        [3]Tdnn(192*64) Relu
    |        |           |
    |        |        [4]Gemm(64*32) Scale
    |        |            |
    |        |        [5]Tdnn(96*32) Relu
    |        |        |           |
    |        |        |        [6]Gemm(32*16) Scale
    |        |        |            |
    |        |        |        [7]Tdnn(48*16) Relu
    |        |        |            |
    |        |        |        [8]Tdnn(48*32)
    |        |        |____________|
    |        |            |
    |        |            Add
    |        |            Relu
    |        |            |(32)
    |        |            [9]Tdnn(96*64)
    |        |____________|
    |            |
    |            Add
    |            Relu
    |            |(64)
    |            [10]Tdnn(192*128)
    |____________|
        |(128)
        Add
        Relu
        |
        [11]Tdnn(384*256) Relu
        |
        [12]Gemm(256*514) Scale
        |
        out(514)
    '''
    gemm_params_0 = np.random.randint(-50, 50, size=(514, 128),dtype=np.int32).astype(np.float32)
    tdnn_params_1 = np.random.randint(-50, 50, size=(384, 128),dtype=np.int32).astype(np.float32)
    gemm_params_2 = np.random.randint(-50, 50, size=(128, 64),dtype=np.int32).astype(np.float32)
    tdnn_params_3 = np.random.randint(-50, 50, size=(192, 64),dtype=np.int32).astype(np.float32)
    gemm_params_4 = np.random.randint(-128, 127, size=(64, 32),dtype=np.int32).astype(np.float32)
    tdnn_params_5 = np.random.randint(-50, 50, size=(96, 32),dtype=np.int32).astype(np.float32)
    gemm_params_6 = np.random.randint(-50, 50, size=(32, 16),dtype=np.int32).astype(np.float32)
    tdnn_params_7 = np.random.randint(-50, 50, size=(48, 16),dtype=np.int32).astype(np.float32)
    tdnn_params_8 = np.random.randint(-50, 50, size=(48, 32),dtype=np.int32).astype(np.float32)
    tdnn_params_9 = np.random.randint(-128, 127, size=(96, 64),dtype=np.int32).astype(np.float32)
    tdnn_params_10 = np.random.randint(-50, 50, size=(192, 128),dtype=np.int32).astype(np.float32)
    tdnn_params_11 = np.random.randint(-50, 50, size=(384, 256),dtype=np.int32).astype(np.float32)
    gemm_params_12 = np.random.randint(-50, 50, size=(256, 514),dtype=np.int32).astype(np.float32)

    gemm_bias_0 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_1 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    gemm_bias_2 = np.random.randint(-50, 50, size=(64,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_3 = np.random.randint(-50, 50, size=(64,),dtype=np.int32).astype(np.float32) * 128
    gemm_bias_4 = np.random.randint(-128, 127, size=(32,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_5 = np.random.randint(-50, 50, size=(32,),dtype=np.int32).astype(np.float32) * 128
    gemm_bias_6 = np.random.randint(-50, 50, size=(16,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_7 = np.random.randint(-50, 50, size=(16,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_8 = np.random.randint(-50, 50, size=(32,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_9 = np.random.randint(-128, 127, size=(64,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_10 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_11 = np.random.randint(-50, 50, size=(256,),dtype=np.int32).astype(np.float32) * 128
    gemm_bias_12 = np.random.randint(-50, 50, size=(514,),dtype=np.int32).astype(np.float32) * 128

    out_dim = gemm_params_12.shape[1]

    offsets_layer1 = onnx.helper.make_tensor("layer1_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer3 = onnx.helper.make_tensor("layer3_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer5 = onnx.helper.make_tensor("layer5_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer7 = onnx.helper.make_tensor("layer7_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer8 = onnx.helper.make_tensor("layer8_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer9 = onnx.helper.make_tensor("layer9_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer10 = onnx.helper.make_tensor("layer10_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer11 = onnx.helper.make_tensor("layer11_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])

    params0 = onnx.helper.make_tensor("params0", data_type=onnx.TensorProto.FLOAT, dims=(514, 128), vals=gemm_params_0.flatten())
    params1 = onnx.helper.make_tensor("params1", data_type=onnx.TensorProto.FLOAT, dims=(384, 128), vals=tdnn_params_1.flatten())
    params2 = onnx.helper.make_tensor("params2", data_type=onnx.TensorProto.FLOAT, dims=(128, 64), vals=gemm_params_2.flatten())
    params3 = onnx.helper.make_tensor("params3", data_type=onnx.TensorProto.FLOAT, dims=(192, 64), vals=tdnn_params_3.flatten())
    params4 = onnx.helper.make_tensor("params4", data_type=onnx.TensorProto.FLOAT, dims=(64, 32), vals=gemm_params_4.flatten())
    params5 = onnx.helper.make_tensor("params5", data_type=onnx.TensorProto.FLOAT, dims=(96, 32), vals=tdnn_params_5.flatten())
    params6 = onnx.helper.make_tensor("params6", data_type=onnx.TensorProto.FLOAT, dims=(32, 16), vals=gemm_params_6.flatten())
    params7 = onnx.helper.make_tensor("params7", data_type=onnx.TensorProto.FLOAT, dims=(48, 16), vals=tdnn_params_7.flatten())
    params8 = onnx.helper.make_tensor("params8", data_type=onnx.TensorProto.FLOAT, dims=(48, 32), vals=tdnn_params_8.flatten())
    params9 = onnx.helper.make_tensor("params9", data_type=onnx.TensorProto.FLOAT, dims=(96, 64), vals=tdnn_params_9.flatten())
    params10 = onnx.helper.make_tensor("params10", data_type=onnx.TensorProto.FLOAT, dims=(192, 128), vals=tdnn_params_10.flatten())
    params11 = onnx.helper.make_tensor("params11", data_type=onnx.TensorProto.FLOAT, dims=(384, 256), vals=tdnn_params_11.flatten())
    params12 = onnx.helper.make_tensor("params12", data_type=onnx.TensorProto.FLOAT, dims=(256, 514), vals=gemm_params_12.flatten())

    bias0 = onnx.helper.make_tensor("bias0", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=gemm_bias_0.flatten())
    bias1 = onnx.helper.make_tensor("bias1", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_1.flatten())
    bias2 = onnx.helper.make_tensor("bias2", data_type=onnx.TensorProto.FLOAT, dims=(64,), vals=gemm_bias_2.flatten())
    bias3 = onnx.helper.make_tensor("bias3", data_type=onnx.TensorProto.FLOAT, dims=(64,), vals=tdnn_bias_3.flatten())
    bias4 = onnx.helper.make_tensor("bias4", data_type=onnx.TensorProto.FLOAT, dims=(32,), vals=gemm_bias_4.flatten())
    bias5 = onnx.helper.make_tensor("bias5", data_type=onnx.TensorProto.FLOAT, dims=(32,), vals=tdnn_bias_5.flatten())
    bias6 = onnx.helper.make_tensor("bias6", data_type=onnx.TensorProto.FLOAT, dims=(16,), vals=gemm_bias_6.flatten())
    bias7 = onnx.helper.make_tensor("bias7", data_type=onnx.TensorProto.FLOAT, dims=(16,), vals=tdnn_bias_7.flatten())
    bias8 = onnx.helper.make_tensor("bias8", data_type=onnx.TensorProto.FLOAT, dims=(32,), vals=tdnn_bias_8.flatten())
    bias9 = onnx.helper.make_tensor("bias9", data_type=onnx.TensorProto.FLOAT, dims=(64,), vals=tdnn_bias_9.flatten())
    bias10 = onnx.helper.make_tensor("bias10", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_10.flatten())
    bias11 = onnx.helper.make_tensor("bias11", data_type=onnx.TensorProto.FLOAT, dims=(256,), vals=tdnn_bias_11.flatten())
    bias12 = onnx.helper.make_tensor("bias12", data_type=onnx.TensorProto.FLOAT, dims=(514,), vals=gemm_bias_12.flatten())


    node0 = onnx.helper.make_node('Gemm', inputs = ['in', 'params0', 'bias0'], outputs = ['node0'], name = 'dnn1')
    node0_mul = onnx.helper.make_node('Scale', ['node0'], ['node0_mul'], scale = 0.0009765625)
    node1 = onnx.helper.make_node('Tdnn', inputs = ['node0_mul', 'params1'], outputs = ['node1'],
                                    time_offsets = offsets_layer1, bias_params = bias1, scale_params=1024, name = 'tdnn1')
    node1_relu = onnx.helper.make_node('Relu', ['node1'], ['node1_relu'])

    node2 = onnx.helper.make_node('Gemm', inputs = ['node1_relu', 'params2', 'bias2'], outputs = ['node2'], name = 'dnn2')
    node2_mul = onnx.helper.make_node('Scale', ['node2'], ['node2_mul'], scale = 0.0009765625)
    node3 = onnx.helper.make_node('Tdnn', inputs = ['node2_mul', 'params3'], outputs = ['node3'],
                                    time_offsets = offsets_layer3, bias_params = bias3, scale_params=1024, name = 'tdnn2')
    node3_relu = onnx.helper.make_node('Relu', ['node3'], ['node3_relu'])

    node4 = onnx.helper.make_node('Gemm', inputs = ['node3_relu', 'params4', 'bias4'], outputs = ['node4'], name = 'dnn3')
    node4_mul = onnx.helper.make_node('Scale', ['node4'], ['node4_mul'], scale = 0.0009765625)
    node5 = onnx.helper.make_node('Tdnn', inputs = ['node4_mul', 'params5'], outputs = ['node5'],
                                    time_offsets = offsets_layer5, bias_params = bias5, scale_params=1024, name = 'tdnn3')
    node5_relu = onnx.helper.make_node('Relu', ['node5'], ['node5_relu'])

    node6 = onnx.helper.make_node('Gemm', inputs = ['node5_relu', 'params6', 'bias6'], outputs = ['node6'], name = 'dnn4')
    node6_mul = onnx.helper.make_node('Scale', ['node6'], ['node6_mul'], scale = 0.0009765625)
    node7 = onnx.helper.make_node('Tdnn', inputs = ['node6_mul', 'params7'], outputs = ['node7'],
                                    time_offsets = offsets_layer7, bias_params = bias7, scale_params=1024, name = 'tdnn4')
    node7_relu = onnx.helper.make_node('Relu', ['node7'], ['node7_relu'])

    node8 = onnx.helper.make_node('Tdnn', inputs = ['node7_relu', 'params8'], outputs = ['node8'],
                                    time_offsets = offsets_layer8, bias_params = bias8, scale_params=1024, name = 'tdnn5')
    node_add_0 = onnx.helper.make_node('Add', ['node5_relu', 'node8'], ['node_add_0'])
    node_add_0_relu = onnx.helper.make_node('Relu', ['node_add_0'], ['node_add_0_relu'])

    node9 = onnx.helper.make_node('Tdnn', inputs = ['node_add_0_relu', 'params9'], outputs = ['node9'],
                                    time_offsets = offsets_layer9, bias_params = bias9, scale_params=1024, name = 'tdnn6')
    node_add_1 = onnx.helper.make_node('Add', ['node3_relu', 'node9'], ['node_add_1'])
    node_add_1_relu = onnx.helper.make_node('Relu', ['node_add_1'], ['node_add_1_relu'])

    node10 = onnx.helper.make_node('Tdnn', inputs = ['node_add_1_relu', 'params10'], outputs = ['node10'],
                                    time_offsets = offsets_layer10, bias_params = bias10, scale_params=1024, name = 'tdnn7')
    node_add_2 = onnx.helper.make_node('Add', ['node1_relu', 'node10'], ['node_add_2'])
    node_add_2_relu = onnx.helper.make_node('Relu', ['node_add_2'], ['node_add_2_relu'])

    node11 = onnx.helper.make_node('Tdnn', inputs = ['node_add_2_relu', 'params11'], outputs = ['node11'],
                                    time_offsets = offsets_layer11, bias_params = bias11, scale_params=1024, name = 'tdnn8')
    node11_relu = onnx.helper.make_node('Relu', ['node11'], ['node11_relu'])

    node12 = onnx.helper.make_node('Gemm', inputs = ['node11_relu', 'params12', 'bias12'], outputs = ['node12'], name = 'dnn5')
    node12_mul = onnx.helper.make_node('Scale', ['node12'], ['out'], scale = 0.0009765625)


    in_shape = (1, 514)
    out_shape = (1, 514)
    initializer=[offsets_layer1, offsets_layer3, offsets_layer5, offsets_layer7, offsets_layer8, offsets_layer9, offsets_layer10, offsets_layer11,
                params0, params1, params2, params3, params4, params5, params6, params7, params8, params9, params10, params11, params12,
                bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10, bias11, bias12]

    graph = onnx.helper.make_graph([node0, node0_mul, node1, node1_relu, node2, node2_mul, node3, node3_relu,
                                    node4, node4_mul, node5, node5_relu, node6, node6_mul, node7, node7_relu,
                                    node8, node_add_0, node_add_0_relu, node9, node_add_1, node_add_1_relu,
                                    node10, node_add_2, node_add_2_relu, node11, node11_relu, node12, node12_mul
                                    ], "tdnn_bb04p1_hua_shibie_all",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name='tdnn_bb04p1_hua_shibie_all')
    with open('./model/tdnn_bb04p1_hua_shibie_all.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate tdnn_bb04p1_hua_shibie_all.onnx sucessfully!")


def distance(a, b):
    v1 = np.sqrt(np.sum((np.int32(b) - np.int32(a))**2))
    v2 = np.sqrt(np.sum(1e-5 + np.int32(b)**2))
    v3 = v1 / v2
    ret = np.sum(v3)
    # print("compare distance is:%.4f"%(ret))
    return ret


def build_run_model(mod, params, optimize_method_config, data, base_out):
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod, target='npu',
                        target_host='npu', params=params,
                        input_data=input_dt,
                        chip = "BB04P1",
                        output_dir=build_dir,
                        optimize_method_config = optimize_method_config
                        )
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(1)]
    # print(output[0])
    if base_out is not None:
       ret = distance(base_out, output[0])
       if 1 - ret < 0.95:
            print(base_out)
            print(output[0])
            print(base_out - output[0])
            # raise ValueError("similarity is  '%f' " % (ret))
    return output[0]


optimize_method_config = [
    ("./model/pipeline/SH_CBG_shibie_optimize/2_optimize_config_weightOpt.protobuf"),
    ("./model/pipeline/SH_CBG_shibie_optimize/3_optimize_config_doubleweight.protobuf"),
    ("./model/pipeline/SH_CBG_shibie_optimize/6_optimize_config_weightOpt_doubleweight.protobuf"),
]

optimize_method_config_frist_layer_inputOpt = [
    ("./model/pipeline/SH_CBG_shibie_optimize/1_optimize_config_inputOpt.protobuf"),
    ("./model/huawei_pipelinemodel/SH_CBG_shibie_optimize/4_optimize_config_inputOpt_weightOpt.protobuf"),
    ("./model/pipeline/SH_CBG_shibie_optimize/5_optimize_config_inputOpt_doubleweight.protobuf"),
    ("./model/pipeline/SH_CBG_shibie_optimize/7_optimize_config_inputOpt_weightOpt_doubleweight.protobuf"),
]


def gen_hua_shibie_all_model_data():
    np.random.seed(10)
    generate_tdnn_model()
    file_name = './model/tdnn_bb04p1_hua_shibie_all.onnx'
    onnx_model = onnx.load(file_name)
    shape_dict = {}
    shape_dict['in'] = (1, 514)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data = np.round(np.random.rand(150, 514)*110).astype("float32")
    return mod, params, data


mod, params, data = gen_hua_shibie_all_model_data()
base_out = build_run_model(mod, params, "", data, None)

@pytest.mark.parametrize("config", optimize_method_config)
def hua_shibie_all_optimize(config):
    build_run_model(mod, params, config, data, base_out)


@pytest.mark.parametrize("config", optimize_method_config_frist_layer_inputOpt)
def hua_shibie_all_first_layer_optimize(config):
    build_run_model(mod, params, config, data*2, base_out)



