# Licensed to the Apache Software Foundation (ASF) under one
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
import witin
from witin import *
import numpy as np
import math
import onnx
from onnx import helper, TensorProto, mapping
import scipy
import sys

def get_tvm_output(graph_def):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 40)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    feats = np.loadtxt('./model/CHML_TDNN20210701_TDNN_kws/calibration_data.txt', skiprows=70, max_rows=140).astype("float32")
    data1=feats[0:140][:]

    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data1)
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt)

def generate_tdnn_model(modelpath):
    tdnn_weight_0 = np.loadtxt("%s/w_0.txt"%(modelpath), dtype=np.float32).transpose()
    tdnn_weight_1 =  np.loadtxt("%s/w_1.txt"%(modelpath), dtype=np.float32).transpose()
    tdnn_weight_2 =  np.loadtxt("%s/w_2.txt"%(modelpath), dtype=np.float32).transpose()
    tdnn_weight_3 =  np.loadtxt("%s/w_3.txt"%(modelpath), dtype=np.float32).transpose()
    tdnn_weight_4 =  np.loadtxt("%s/w_4_trim.txt"%(modelpath), dtype=np.float32).transpose()

    tdnn_bias_0 = np.loadtxt("%s/b_0.txt"%(modelpath), dtype=np.float32)
    tdnn_bias_1 = np.loadtxt("%s/b_1.txt"%(modelpath), dtype=np.float32)
    tdnn_bias_2 = np.loadtxt("%s/b_2.txt"%(modelpath), dtype=np.float32)
    tdnn_bias_3 = np.loadtxt("%s/b_3.txt"%(modelpath), dtype=np.float32)
    tdnn_bias_4 = np.loadtxt("%s/b_4_trim.txt"%(modelpath), dtype=np.float32)
    
    out_dim = tdnn_weight_4.shape[1]
    offsets_layer1 = onnx.helper.make_tensor("layer1_offset", data_type=onnx.TensorProto.FLOAT, dims=(11,), vals=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    offsets_layer2 = onnx.helper.make_tensor("layer2_offset", data_type=onnx.TensorProto.FLOAT, dims=(7,), vals=[-3,-2,-1,0,1,2,3])
    offsets_layer3 = onnx.helper.make_tensor("layer3_offset", data_type=onnx.TensorProto.FLOAT, dims=(7,), vals=[-3,-2,-1,0,1,2,3])
    offsets_layer4 = onnx.helper.make_tensor("layer4_offset", data_type=onnx.TensorProto.FLOAT, dims=(7,), vals=[-3,-2,-1,0,1,2,3])
    offsets_layer5 = onnx.helper.make_tensor("layer5_offset", data_type=onnx.TensorProto.FLOAT, dims=(7,), vals=[-3,-2,-1,0,1,2,3])

    linear_params1 = onnx.helper.make_tensor("layer1_params", data_type=onnx.TensorProto.FLOAT, dims=(440, 128), vals=tdnn_weight_0.flatten())
    linear_params2 = onnx.helper.make_tensor("layer2_params", data_type=onnx.TensorProto.FLOAT, dims=(896, 128), vals=tdnn_weight_1.flatten())
    linear_params3 = onnx.helper.make_tensor("layer3_params", data_type=onnx.TensorProto.FLOAT, dims=(896, 128), vals=tdnn_weight_2.flatten())
    linear_params4 = onnx.helper.make_tensor("layer4_params", data_type=onnx.TensorProto.FLOAT, dims=(896, 128), vals=tdnn_weight_3.flatten())
    linear_params5 = onnx.helper.make_tensor("layer5_params", data_type=onnx.TensorProto.FLOAT, dims=(896, out_dim), vals=tdnn_weight_4.flatten())

    linear_bias1 = onnx.helper.make_tensor("layer1_bias", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_0.flatten())
    linear_bias2 = onnx.helper.make_tensor("layer2_bias", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_1.flatten())
    linear_bias3 = onnx.helper.make_tensor("layer3_bias", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_2.flatten())
    linear_bias4 = onnx.helper.make_tensor("layer4_bias", data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_3.flatten())
    linear_bias5 = onnx.helper.make_tensor("layer5_bias", data_type=onnx.TensorProto.FLOAT, dims=(out_dim,), vals=tdnn_bias_4.flatten())

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

    relu_node2 = onnx.helper.make_node('Relu', ['tdnn2'], ['relu_out2'])

    node3=onnx.helper.make_node('Tdnn', inputs=['relu_out2', 'layer3_params'],
                                outputs=['tdnn3'],
                                time_offsets=offsets_layer3,
                                bias_params=linear_bias3,
                                scale_params=1024
                                )

    relu_node3 = onnx.helper.make_node('Relu', ['tdnn3'], ['relu_out3'])

    node4=onnx.helper.make_node('Tdnn', inputs=['relu_out3', 'layer4_params'],
                                outputs=['tdnn4'],
                                time_offsets=offsets_layer4,
                                bias_params=linear_bias4,
                                scale_params=1024
                                )

    relu_node4 = onnx.helper.make_node('Relu', ['tdnn4'], ['relu_out4'])

    node5=onnx.helper.make_node('Tdnn', inputs=['relu_out4', 'layer5_params'],
                                outputs=['out'],
                                time_offsets=offsets_layer5,
                                bias_params=linear_bias5,
                                scale_params=1024
                                )
    in_shape = (1, 40)
    out_shape = (1, out_dim)
    initializer=[offsets_layer1, offsets_layer2, offsets_layer3, offsets_layer4, offsets_layer5
            ,linear_params1, linear_params2, linear_params3, linear_params4, linear_params5,
            linear_bias1, linear_bias2, linear_bias3, linear_bias4, linear_bias5]

    graph = onnx.helper.make_graph([node1, relu_node1, node2, relu_node2, node3, relu_node3
                            , node4, relu_node4, node5], "tdnn_bb04p",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )

    model=helper.make_model(graph, producer_name='tdnn_bb04p')
    with open('./model/tdnn_bb04p.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate tdnn_bb04p.onnx sucessfully!")

def test_model():
    file_name = './model/tdnn_bb04p.onnx'
    onnx_model = onnx.load(file_name)
    target="npu"
    ctx="npu"
    tvm_out = get_tvm_output(onnx_model)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        printf("*.py  <modelpath>")
    else:
        generate_tdnn_model(sys.argv[1])
        test_model()
