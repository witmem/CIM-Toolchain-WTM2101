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
from onnx import helper
import onnx
import os
import datetime
import witin
from witin import *


def get_tvm_output(graph_def):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 40)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    data1 = np.round(np.random.rand(70, 40)*255).astype("float32")
    input_dt = {}
    input_dt['in'] = witin.nd.array(data1)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        output_dir=build_dir,
                                        chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime

    data2 = np.round(np.random.rand(150, 120)*255).astype("float32")
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data2))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]


def generate_tdnn_model():
    tdnn_weight_0 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_1 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_2 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_3 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_4 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_6 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_7 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    gemm_params_1 = np.random.randint(-128, 127, size=(40, 40), dtype=np.int32).astype(np.float32)
    gemm_params_2 = np.random.randint(-128, 127, size=(40, 40), dtype=np.int32).astype(np.float32)
    gemm_params_0 = np.random.randint(-128, 127, size=(40, 40), dtype=np.int32).astype(np.float32)

    tdnn_bias_0 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_1 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_2 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_3 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_5 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_7 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    gemm_bias_1 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    gemm_bias_2 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    gemm_bias_0 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)

    out_dim = tdnn_weight_7.shape[1]
    offsets_layer1 = onnx.helper.make_tensor("layer1_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer2 = onnx.helper.make_tensor("layer2_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer3 = onnx.helper.make_tensor("layer3_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer4 = onnx.helper.make_tensor("layer4_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer5 = onnx.helper.make_tensor("layer5_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer6 = onnx.helper.make_tensor("layer6_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer7 = onnx.helper.make_tensor("layer7_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer8 = onnx.helper.make_tensor("layer8_offset", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])



    linear_params1 = onnx.helper.make_tensor("layer1_params", data_type=onnx.TensorProto.FLOAT, dims=(120, 40), vals=tdnn_weight_0.flatten())
    linear_params2 = onnx.helper.make_tensor("layer2_params", data_type=onnx.TensorProto.FLOAT, dims=(120, 40), vals=tdnn_weight_1.flatten())
    linear_params3 = onnx.helper.make_tensor("layer3_params", data_type=onnx.TensorProto.FLOAT, dims=(120, 40), vals=tdnn_weight_2.flatten())
    linear_params4 = onnx.helper.make_tensor("layer4_params", data_type=onnx.TensorProto.FLOAT, dims=(120, 40), vals=tdnn_weight_3.flatten())
    linear_params8 = onnx.helper.make_tensor("layer8_params", data_type=onnx.TensorProto.FLOAT, dims=(120, 40), vals=tdnn_weight_7.flatten())
    gemm_params1 = onnx.helper.make_tensor("gemm_params1",
                    data_type=onnx.TensorProto.FLOAT, dims=(40, 40), vals=gemm_params_1.flatten())
    gemm_params2 = onnx.helper.make_tensor("gemm_params2",
                    data_type=onnx.TensorProto.FLOAT, dims=(40, 40), vals=gemm_params_2.flatten())

    gemm_params = onnx.helper.make_tensor("gemm_params",
                    data_type=onnx.TensorProto.FLOAT, dims=(40, 40), vals=gemm_params_0.flatten())



    linear_bias1 = onnx.helper.make_tensor("layer1_bias", data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=tdnn_bias_0.flatten())
    linear_bias2 = onnx.helper.make_tensor("layer2_bias", data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=tdnn_bias_1.flatten())
    linear_bias3 = onnx.helper.make_tensor("layer3_bias", data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=tdnn_bias_2.flatten())
    linear_bias4 = onnx.helper.make_tensor("layer4_bias", data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=tdnn_bias_3.flatten())
    linear_bias8 = onnx.helper.make_tensor("layer8_bias", data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=tdnn_bias_7.flatten())
    gemm_bias1 = onnx.helper.make_tensor("gemm_bias1",
                    data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=gemm_bias_1.flatten())

    gemm_bias2 = onnx.helper.make_tensor("gemm_bias2",
                    data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=gemm_bias_2.flatten())

    gemm_bias = onnx.helper.make_tensor("gemm_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=gemm_bias_0.flatten())

    node1=onnx.helper.make_node('Tdnn', inputs=['in', 'layer1_params'],
                                outputs=['tdnn1'],
                                time_offsets=offsets_layer1,
                                bias_params=linear_bias1,
                                scale_params=1024
                                )

    relu_node1 = onnx.helper.make_node('Relu', inputs=['tdnn1'],
                                                outputs=['relu_out1'])

    node2=onnx.helper.make_node('Tdnn', inputs=['relu_out1', 'layer2_params'],
                                outputs=['tdnn2'],
                                time_offsets=offsets_layer2,
                                bias_params=linear_bias2,
                                scale_params=1024
                                )

    # relu_node3 = onnx.helper.make_node('Relu', ['tdnn2'], ['relu_out3'])

    # gemm_node1=onnx.helper.make_node('Gemm', inputs=['relu_out3', 'gemm_params1', 'gemm_bias1'],outputs=['gemm_out1'])
    gemm_node1=onnx.helper.make_node('Gemm', inputs=['tdnn2', 'gemm_params1', 'gemm_bias1'],outputs=['gemm_out1'])
    gemm_mul_node1 = onnx.helper.make_node('Scale', ['gemm_out1'], ['gemm_mul_node_out1'], scale=0.0009765625)

    gemm_relu_node1 = onnx.helper.make_node('Relu', ['gemm_mul_node_out1'], ['gemm_relu_out1'])


    node4=onnx.helper.make_node('Tdnn', inputs=['gemm_relu_out1', 'layer4_params'],
                                outputs=['tdnn4'],
                                time_offsets=offsets_layer4,
                                bias_params=linear_bias4,
                                scale_params=1024
                                )

    relu_node4 = onnx.helper.make_node('Relu', ['tdnn4'], ['relu_out4'])

    gemm_node2=onnx.helper.make_node('Gemm', inputs=['relu_out4', 'gemm_params2', 'gemm_bias2'],
                                outputs=['gemm_out2']
                                )
    gemm_mul_node2 = onnx.helper.make_node('Scale', ['gemm_out2'], ['gemm_mul_node_out2'], scale=0.0009765625)

    gemm_relu_node2 = onnx.helper.make_node('Relu', ['gemm_mul_node_out2'], ['gemm_relu_out2'])

    # scale_node1 = onnx.helper.make_node('Scale', ['relu_out3'], ['scale_node_out1'], scale=0.5)
    scale_node1 = onnx.helper.make_node('Scale', ['tdnn2'], ['scale_node_out1'], scale=0.5)

    add_node = onnx.helper.make_node('Add', ['scale_node_out1', 'gemm_relu_out2'], ['add_out'])

    y = np.random.randint(-128, 127, size=(40,),dtype=np.int32).astype(np.float32)
    y_t = onnx.helper.make_tensor("y_t_bias", data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=y.flatten())

    mul_node = onnx.helper.make_node('Mul',['add_out', 'y_t_bias' ], ['mul_node'])

    relu_node7 = onnx.helper.make_node('Relu', ['mul_node'], ['relu_out7'])

    gemm_node=onnx.helper.make_node('Gemm', inputs=['relu_out7', 'gemm_params', 'gemm_bias'],
                                outputs=['gemm_out']
                                )
    gemm_mul_node = onnx.helper.make_node('Scale', ['gemm_out'], ['gemm_mul_node_out'], scale=0.0009765625)

    gemm_relu_node = onnx.helper.make_node('Relu', ['gemm_mul_node_out'], ['gemm_relu_out'])

    node8=onnx.helper.make_node('Tdnn', inputs=['gemm_relu_out', 'layer8_params'],
                                outputs=['out'],
                                time_offsets=offsets_layer8,
                                bias_params=linear_bias8,
                                scale_params=1024
                                )


    in_shape = (1, 40)
    out_shape = (1, 40)
    initializer=[offsets_layer1, offsets_layer2, offsets_layer3, offsets_layer4, offsets_layer8,
            linear_params1, linear_params2, linear_params3, linear_params4, linear_params8,
            linear_bias1, linear_bias2, linear_bias3, linear_bias4, linear_bias8, y_t, gemm_params1, gemm_params2, gemm_params, gemm_bias1, gemm_bias2, gemm_bias]

    graph = onnx.helper.make_graph([node1, relu_node1, node2, gemm_node1, gemm_mul_node1, gemm_relu_node1, node4, relu_node4, gemm_node2,
                                 gemm_mul_node2, gemm_relu_node2, scale_node1, add_node, mul_node, relu_node7,gemm_node, gemm_mul_node, gemm_relu_node, node8], "skip_scale",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name='skip_scale')
    with open('./model/skip_scale.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate skip_scale.onnx sucessfully!")

def test_model():
    file_name = './model/skip_scale.onnx'
    onnx_model = onnx.load(file_name)
    target="npu"
    ctx="npu"
    tvm_out = get_tvm_output(onnx_model)

def test_forward_skip_scale():
    np.random.seed(10)
    generate_tdnn_model()
    test_model()

if __name__ == '__main__':
    test_forward_skip_scale()

