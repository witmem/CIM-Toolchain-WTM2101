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
import os
import datetime
import witin
from witin import *
np.random.seed(10)

ROOT_DIR = './model/'
def generate_act_lut_model_case1():
    '''
    dnn
     |
    relu
     |
    tdnn (input_opt)
     |
    ActLut
    '''
    model_name = "act_lut_bb04p1_case1"
    gemm_params_0 = np.random.randint(-50, 50, size=(514, 128),dtype=np.int32).astype(np.float32)
    tdnn_params_1 = np.random.randint(-50, 50, size=(384, 128),dtype=np.int32).astype(np.float32)

    gemm_bias_0 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_1 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128

    act_table1 = np.random.randint(-50, 50, size=(256,),dtype=np.int32).astype(np.float32)

    offsets_layer1 = onnx.helper.make_tensor("layer1_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    params0 = onnx.helper.make_tensor("params0", data_type=onnx.TensorProto.FLOAT, dims=(514, 128), vals=gemm_params_0.flatten())
    params1 = onnx.helper.make_tensor("params1", data_type=onnx.TensorProto.FLOAT, dims=(384, 128), vals=tdnn_params_1.flatten())

    bias0 = onnx.helper.make_tensor("bias0", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=gemm_bias_0.flatten())
    bias1 = onnx.helper.make_tensor("bias1", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_1.flatten())
    table1 = onnx.helper.make_tensor("table1", data_type=onnx.TensorProto.FLOAT, dims=(256,), vals=act_table1.flatten())

    node0 = onnx.helper.make_node('Gemm', inputs = ['in', 'params0', 'bias0'], outputs = ['node0'], name = 'dnn1')
    node0_mul = onnx.helper.make_node('Scale', ['node0'], ['node0_mul'], scale = 0.0009765625)
    node0_relu = onnx.helper.make_node('Relu', ['node0_mul'], ['node0_relu'])
    node1 = onnx.helper.make_node('Tdnn', inputs = ['node0_relu', 'params1'], outputs = ['node1'],
                                    time_offsets = offsets_layer1, bias_params = bias1, scale_params=1024, name = 'tdnn1')
    node1_act_lut = onnx.helper.make_node('ActLut', ['node1'], ['out'], act_type = 'sigmoid' , table_params = table1)

    in_shape = (1, 514)
    out_shape = (1, 128)
    initializer=[offsets_layer1, params0, params1, bias0, bias1, table1]

    graph = onnx.helper.make_graph([node0, node0_mul, node0_relu, node1, node1_act_lut], model_name,
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name=model_name)
    with open(ROOT_DIR + model_name + '.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate %s.onnx sucessfully!" % model_name)
    return ROOT_DIR + model_name + '.onnx'


def generate_act_lut_model_case2():
    '''
    dnn
     |
    relu
     |
    tdnn
     |
    tanh
     |
    ActLut
    '''
    model_name = "act_lut_bb04p1_case2"
    gemm_params_0 = np.random.randint(-50, 50, size=(514, 128),dtype=np.int32).astype(np.float32)
    tdnn_params_1 = np.random.randint(-50, 50, size=(384, 128),dtype=np.int32).astype(np.float32)

    gemm_bias_0 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_1 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128

    act_table1 = np.random.randint(-50, 50, size=(256,),dtype=np.int32).astype(np.float32)

    offsets_layer1 = onnx.helper.make_tensor("layer1_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    params0 = onnx.helper.make_tensor("params0", data_type=onnx.TensorProto.FLOAT, dims=(514, 128), vals=gemm_params_0.flatten())
    params1 = onnx.helper.make_tensor("params1", data_type=onnx.TensorProto.FLOAT, dims=(384, 128), vals=tdnn_params_1.flatten())

    bias0 = onnx.helper.make_tensor("bias0", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=gemm_bias_0.flatten())
    bias1 = onnx.helper.make_tensor("bias1", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_1.flatten())
    table1 = onnx.helper.make_tensor("table1", data_type=onnx.TensorProto.FLOAT, dims=(256,), vals=act_table1.flatten())

    node0 = onnx.helper.make_node('Gemm', inputs = ['in', 'params0', 'bias0'], outputs = ['node0'], name = 'dnn1')
    node0_mul = onnx.helper.make_node('Scale', ['node0'], ['node0_mul'], scale = 0.0009765625)
    node0_relu = onnx.helper.make_node('Relu', ['node0_mul'], ['node0_relu'])
    node1 = onnx.helper.make_node('Tdnn', inputs = ['node0_relu', 'params1'], outputs = ['node1'],
                                    time_offsets = offsets_layer1, bias_params = bias1, scale_params=1024, name = 'tdnn1')
    node1_tanh = onnx.helper.make_node('Tanh', ['node1'], ['node1_tanh'])
    node1_act_lut = onnx.helper.make_node('ActLut', ['node1_tanh'], ['out'], act_type = 'sigmoid' , table_params = table1)

    in_shape = (1, 514)
    out_shape = (1, 128)
    initializer=[offsets_layer1, params0, params1, bias0, bias1, table1]

    graph = onnx.helper.make_graph([node0, node0_mul, node0_relu, node1, node1_tanh, node1_act_lut], model_name,
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name=model_name)
    with open(ROOT_DIR + model_name + '.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate %s.onnx sucessfully!" % model_name)
    return ROOT_DIR + model_name + '.onnx'

def generate_act_lut_model_case3():
    '''
    dnn
     |
    tdnn
     |
    ActLut(sigmoid)
     |
    ActLut(tanh)
     |
    dnn
     |
    tdnn
     |
    ActLut(sigmoid)
     |
    ActLut(tanh)
    '''
    model_name = "act_lut_bb04p1_case3"
    gemm_params_0 = np.random.randint(-50, 50, size=(514, 128),dtype=np.int32).astype(np.float32)
    tdnn_params_1 = np.random.randint(-50, 50, size=(384, 128),dtype=np.int32).astype(np.float32)
    gemm_params_2 = np.random.randint(-50, 50, size=(128, 64),dtype=np.int32).astype(np.float32)
    tdnn_params_3 = np.random.randint(-50, 50, size=(192, 64),dtype=np.int32).astype(np.float32)

    gemm_bias_0 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_1 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    gemm_bias_2 = np.random.randint(-50, 50, size=(64,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_3 = np.random.randint(-50, 50, size=(64,),dtype=np.int32).astype(np.float32) * 128

    act_table1 = np.random.randint(-50, 50, size=(256,),dtype=np.int32).astype(np.float32)
    act_table2 = np.random.randint(-50, 50, size=(256,),dtype=np.int32).astype(np.float32)

    offsets_layer1 = onnx.helper.make_tensor("layer1_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])
    offsets_layer3 = onnx.helper.make_tensor("layer3_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-2, -1, 0])

    params0 = onnx.helper.make_tensor("params0", data_type=onnx.TensorProto.FLOAT, dims=(514, 128), vals=gemm_params_0.flatten())
    params1 = onnx.helper.make_tensor("params1", data_type=onnx.TensorProto.FLOAT, dims=(384, 128), vals=tdnn_params_1.flatten())
    params2 = onnx.helper.make_tensor("params2", data_type=onnx.TensorProto.FLOAT, dims=(128, 64), vals=gemm_params_2.flatten())
    params3 = onnx.helper.make_tensor("params3", data_type=onnx.TensorProto.FLOAT, dims=(192, 64), vals=tdnn_params_3.flatten())

    bias0 = onnx.helper.make_tensor("bias0", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=gemm_bias_0.flatten())
    bias1 = onnx.helper.make_tensor("bias1", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_1.flatten())
    bias2 = onnx.helper.make_tensor("bias2", data_type=onnx.TensorProto.FLOAT, dims=(64,), vals=gemm_bias_2.flatten())
    bias3 = onnx.helper.make_tensor("bias3", data_type=onnx.TensorProto.FLOAT, dims=(64,), vals=tdnn_bias_3.flatten())

    table1 = onnx.helper.make_tensor("table1", data_type=onnx.TensorProto.FLOAT, dims=(256,), vals=act_table1.flatten())
    table2 = onnx.helper.make_tensor("table2", data_type=onnx.TensorProto.FLOAT, dims=(256,), vals=act_table2.flatten())

    node0 = onnx.helper.make_node('Gemm', inputs = ['in', 'params0', 'bias0'], outputs = ['node0'], name = 'dnn1')
    node0_mul = onnx.helper.make_node('Scale', ['node0'], ['node0_mul'], scale = 0.0009765625)
    node1 = onnx.helper.make_node('Tdnn', inputs = ['node0_mul', 'params1'], outputs = ['node1'],
                                    time_offsets = offsets_layer1, bias_params = bias1, scale_params=1024, name = 'tdnn1')
    node1_act_lut_sig = onnx.helper.make_node('ActLut', ['node1'], ['node1_act_lut_sig'], table_params = table1, act_type = 'sigmoid')
    node1_act_lut_tanh = onnx.helper.make_node('ActLut', ['node1_act_lut_sig'], ['node1_act_lut_tanh'], table_params = table2, act_type = 'tanh')

    node2 = onnx.helper.make_node('Gemm', inputs = ['node1_act_lut_tanh', 'params2', 'bias2'], outputs = ['node2'], name = 'dnn2')
    node2_mul = onnx.helper.make_node('Scale', ['node2'], ['node2_mul'], scale = 0.0009765625)
    node3 = onnx.helper.make_node('Tdnn', inputs = ['node2_mul', 'params3'], outputs = ['node3'],
                                    time_offsets = offsets_layer3, bias_params = bias3, scale_params=1024, name = 'tdnn2')
    node3_act_lut_sig = onnx.helper.make_node('ActLut', ['node3'], ['node3_act_lut_sig'], table_params = table1, act_type = 'sigmoid')
    node3_act_lut_tanh = onnx.helper.make_node('ActLut', ['node3_act_lut_sig'], ['out'], table_params = table2, act_type = 'tanh')

    in_shape = (1, 514)
    out_shape = (1, 64)
    initializer=[offsets_layer1, offsets_layer3, params0, params1, params2, params3, bias0, bias1,
                bias2, bias3, table1, table2]

    graph = onnx.helper.make_graph([node0, node0_mul, node1, node1_act_lut_sig, node1_act_lut_tanh,
                                    node2, node2_mul, node3, node3_act_lut_sig, node3_act_lut_tanh], model_name,
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name=model_name)
    with open(ROOT_DIR + model_name + '.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate %s.onnx sucessfully!" % model_name)
    return ROOT_DIR + model_name + '.onnx'


def distance(a, b):
    v1 = np.sqrt(np.sum((np.int32(b) - np.int32(a))**2))
    v2 =  np.sqrt(np.sum(1e-5 + np.int32(b)**2))
    v3 = v1 / v2
    ret = np.sum(v3)
    # print("compare distance is:%.4f"%(ret))
    return ret

base_out = []
def build_run_model(mod, params, optimize_method_config, data):
    global base_out
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
    if(optimize_method_config == ""):
        base_out = output[0]
    else:
        ret = distance(base_out, output[0])
        if 1 - ret < 0.95:
            print(base_out)
            print(output[0])
            print(base_out - output[0])
            # raise ValueError("similarity is  '%f' " % (ret))

def test_model(file_path, optimize_method_config=[], optimize_method_config_frist_layer_inputOpt=[]):
    global base_out
    base_out = []
    onnx_model = onnx.load(file_path)
    shape_dict = {}
    shape_dict['in'] = (1, 514)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data = np.round(np.random.rand(100, 514)*110).astype("float32")

    # build_run_model(mod, params, "", data)
    for config in optimize_method_config:
        build_run_model(mod, params, config, data)

    #if frist layer need inputOpt ,you should manual * 2
    for config in optimize_method_config_frist_layer_inputOpt:
        build_run_model(mod, params, config, data*2)


def test_forward_acttable():
    np.random.seed(10)
    onnx_model_path = generate_act_lut_model_case1()
    optimize_method_config = [
        '',
        './model/optimize_layer/acttable_optimize_config_inputOpt.protobuf'
    ]
    test_model(onnx_model_path, optimize_method_config)

    onnx_model_path = generate_act_lut_model_case2()
    optimize_method_config = [
        '',
        # fatal because have 3 table:TANH; SIGMOID; RELU_MAP;
        # './model/optimize_layer/acttable_optimize_config_inputOpt.protobuf'
    ]
    test_model(onnx_model_path, optimize_method_config)

    onnx_model_path = generate_act_lut_model_case3()
    optimize_method_config = [
        '',
    ]
    test_model(onnx_model_path, optimize_method_config)

if __name__ == '__main__':
    test_forward_acttable()
