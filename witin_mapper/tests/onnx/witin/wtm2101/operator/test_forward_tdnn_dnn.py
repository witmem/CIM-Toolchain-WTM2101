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
import numpy as np
import onnx
import os
import datetime
import witin
from witin import *

def convert_to_list(x):
    if not isinstance(x, list):
        x=[x]
    return x

def get_tvm_output(graph_def):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 40)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    #feats = np.loadtxt('./model/tdnn_model/calibration_data.txt', skiprows=70, max_rows=140).astype("float32")
    feats = np.loadtxt('./model/730/calibration_data/Rec_010.txt', skiprows=70, max_rows=140).astype("float32")
    data1 = feats[0:140][:]
    data=feats[0:1][:]
    data=convert_to_list(data)
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
                                        chip = "BB04P1"
                                        )
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    # from tvm.contrib import npu_graph_runtime
    # m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    # m.set_input('in1', witin.nd.array(data1))
    # # execute
    # m.run()
    # output = [m.get_output(i).asnumpy() for i in range(1)]


def generate_tdnn_model():
    tdnn_weight_0 =  np.loadtxt("./model/BB04P_DNN20200730M_retrain/w_0.txt", dtype=np.float32)
    tdnn_weight_1 =  np.loadtxt("./model/BB04P_DNN20200730M_retrain/w_1.txt", dtype=np.float32)
    tdnn_weight_2 =  np.loadtxt("./model/BB04P_DNN20200730M_retrain/w_2.txt", dtype=np.float32)
    tdnn_weight_3 =  np.loadtxt("./model/BB04P_DNN20200730M_retrain/w_3.txt", dtype=np.float32)
    tdnn_weight_4 =  np.loadtxt("./model/BB04P_DNN20200730M_retrain/w_4_trim.txt", dtype=np.float32)

    tdnn_bias_0 = np.loadtxt("./model/BB04P_DNN20200730M_retrain/b_0.txt", dtype=np.float32)
    tdnn_bias_1 = np.loadtxt("./model/BB04P_DNN20200730M_retrain/b_1.txt", dtype=np.float32)
    tdnn_bias_2 = np.loadtxt("./model/BB04P_DNN20200730M_retrain/b_2.txt", dtype=np.float32)
    tdnn_bias_3 = np.loadtxt("./model/BB04P_DNN20200730M_retrain/b_3.txt", dtype=np.float32)
    tdnn_bias_4 = np.loadtxt("./model/BB04P_DNN20200730M_retrain/b_4_trim.txt", dtype=np.float32)
    offsets_layer1 = onnx.helper.make_tensor("layer1_offset",
                    data_type=onnx.TensorProto.FLOAT, dims=(11,), vals=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    linear_params1 = onnx.helper.make_tensor("layer1_params",
                    data_type=onnx.TensorProto.FLOAT, dims=(440, 448), vals=tdnn_weight_0.flatten())
    linear_params2 = onnx.helper.make_tensor("layer2_params",
                    data_type=onnx.TensorProto.FLOAT, dims=(448, 448), vals=tdnn_weight_1.flatten())
    linear_params3 = onnx.helper.make_tensor("layer3_params",
                    data_type=onnx.TensorProto.FLOAT, dims=(448, 448), vals=tdnn_weight_2.flatten())
    linear_params4 = onnx.helper.make_tensor("layer4_params",
                    data_type=onnx.TensorProto.FLOAT, dims=(448, 448), vals=tdnn_weight_3.flatten())
    linear_params5 = onnx.helper.make_tensor("layer5_params",
                    data_type=onnx.TensorProto.FLOAT, dims=(448, 916), vals=tdnn_weight_4.flatten())

    linear_bias1 = onnx.helper.make_tensor("layer1_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(448,), vals=tdnn_bias_0.flatten())
    linear_bias2 = onnx.helper.make_tensor("layer2_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(448,), vals=tdnn_bias_1.flatten())
    linear_bias3 = onnx.helper.make_tensor("layer3_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(448,), vals=tdnn_bias_2.flatten())
    linear_bias4 = onnx.helper.make_tensor("layer4_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(448,), vals=tdnn_bias_3.flatten())
    linear_bias5 = onnx.helper.make_tensor("layer5_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(916,), vals=tdnn_bias_4.flatten())

    node1=onnx.helper.make_node('Tdnn', inputs=['in', 'layer1_params'],
                                outputs=['tdnn1'],
                                time_offsets=offsets_layer1,
                                bias_params=linear_bias1,
                                scale_params=512
                                )

    relu_node1 = onnx.helper.make_node('Relu', inputs=['tdnn1'],
                                                outputs=['relu_out1'])

    node2=onnx.helper.make_node('Gemm', inputs=['relu_out1', 'layer2_params', 'layer2_bias'],
                                outputs=['tdnn2']
                                )
    mul_node1 = onnx.helper.make_node('Scale', ['tdnn2'], ['mul_node_out1'], scale=0.001953125)
    relu_node2 = onnx.helper.make_node('Relu', ['mul_node_out1'], ['relu_out2'])

    node3=onnx.helper.make_node('Gemm', inputs=['relu_out2', 'layer3_params', 'layer3_bias'],
                                outputs=['tdnn3']
                                )
    mul_node2 = onnx.helper.make_node('Scale', ['tdnn3'], ['mul_node_out2'], scale=0.001953125)
    relu_node3 = onnx.helper.make_node('Relu', ['mul_node_out2'], ['relu_out3'])

    node4=onnx.helper.make_node('Gemm', inputs=['relu_out3', 'layer4_params', 'layer4_bias'],
                                outputs=['tdnn4']
                                )
    mul_node3 = onnx.helper.make_node('Scale', ['tdnn4'], ['mul_node_out3'], scale=0.001953125)
    relu_node4 = onnx.helper.make_node('Relu', ['mul_node_out3'], ['relu_out4'])

    node5=onnx.helper.make_node('Gemm', inputs=['relu_out4', 'layer5_params', 'layer5_bias'],
                                outputs=['out']
                                )
    mul_node4 = onnx.helper.make_node('Scale', ['out'], ['mul_node_out4'], scale=0.001953125)

    in_shape = (1, 40)
    out_shape = (1, 916)
    initializer=[offsets_layer1,linear_params1, linear_params2, linear_params3, linear_params4, linear_params5,
            linear_bias1, linear_bias2, linear_bias3, linear_bias4, linear_bias5]

    graph = onnx.helper.make_graph([node1, relu_node1, node2, mul_node1, relu_node2, node3,mul_node2, relu_node3
                            , node4, mul_node3, relu_node4, node5, mul_node4], "tdnn_bb04p_tdnn_dnn",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("mul_node_out4", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )

    model = onnx.helper.make_model(graph, producer_name='tdnn_bb04p_tdnn_dnn')
    with open('./model/tdnn_bb04p_tdnn_dnn.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate tdnn_bb04p_tdnn_dnn.onnx sucessfully!")

def test_model():
    file_name = './model/tdnn_bb04p_tdnn_dnn.onnx'
    onnx_model = onnx.load(file_name)
    target="npu"
    ctx="npu"
    tvm_out = get_tvm_output(onnx_model)

def test_forward_tdnn_dnn():
    np.random.seed(10)
    generate_tdnn_model()
    test_model()

if __name__ == '__main__':
    test_forward_tdnn_dnn()
