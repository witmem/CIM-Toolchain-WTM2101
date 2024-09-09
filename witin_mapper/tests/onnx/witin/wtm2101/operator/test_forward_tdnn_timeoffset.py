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
import witin
from witin import *
np.random.seed(10)

def generate_tdnn_timeoffset_model():

    gemm_params_0 = np.random.randint(-50, 50, size=(40, 40),dtype=np.int32).astype(np.float32)

    tdnn_weight_0 =  np.random.randint(-50, 50, size=(80, 128),dtype=np.int32).astype(np.float32)
    tdnn_weight_1 =  np.random.randint(-50, 50, size=(256, 128),dtype=np.int32).astype(np.float32)
    tdnn_weight_2 =  np.random.randint(-50, 50, size=(256, 128),dtype=np.int32).astype(np.float32)
    tdnn_weight_3 =  np.random.randint(-50, 50, size=(256, 128),dtype=np.int32).astype(np.float32)
    tdnn_weight_4 =  np.random.randint(-50, 50, size=(256, 916),dtype=np.int32).astype(np.float32)

    gemm_bias_0 = np.random.randint(-50, 50, size=(40,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_0 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_1 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_2 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_3 = np.random.randint(-50, 50, size=(128,),dtype=np.int32).astype(np.float32) * 128
    tdnn_bias_4 = np.random.randint(-50, 50, size=(916,),dtype=np.int32).astype(np.float32) * 128

    offsets_layer1 = onnx.helper.make_tensor("layer1_offset",data_type=onnx.TensorProto.FLOAT, dims=(2,), vals=[-5,5])
    offsets_layer2 = onnx.helper.make_tensor("layer2_offset",data_type=onnx.TensorProto.FLOAT, dims=(2,), vals=[-3,3])
    offsets_layer3 = onnx.helper.make_tensor("layer3_offset",data_type=onnx.TensorProto.FLOAT, dims=(2,), vals=[-3,3])
    offsets_layer4 = onnx.helper.make_tensor("layer4_offset",data_type=onnx.TensorProto.FLOAT, dims=(2,), vals=[-3,3])
    offsets_layer5 = onnx.helper.make_tensor("layer5_offset",data_type=onnx.TensorProto.FLOAT, dims=(2,), vals=[-3,3])

    params0 = onnx.helper.make_tensor("params0", data_type=onnx.TensorProto.FLOAT, dims=(40, 40), vals=gemm_params_0.flatten())
    linear_params1 = onnx.helper.make_tensor("layer1_params",data_type=onnx.TensorProto.FLOAT, dims=(80, 128), vals=tdnn_weight_0.flatten())
    linear_params2 = onnx.helper.make_tensor("layer2_params",data_type=onnx.TensorProto.FLOAT, dims=(256, 128), vals=tdnn_weight_1.flatten())
    linear_params3 = onnx.helper.make_tensor("layer3_params",data_type=onnx.TensorProto.FLOAT, dims=(256, 128), vals=tdnn_weight_2.flatten())
    linear_params4 = onnx.helper.make_tensor("layer4_params",data_type=onnx.TensorProto.FLOAT, dims=(256, 128), vals=tdnn_weight_3.flatten())
    linear_params5 = onnx.helper.make_tensor("layer5_params",data_type=onnx.TensorProto.FLOAT, dims=(256, 916), vals=tdnn_weight_4.flatten())

    bias0 = onnx.helper.make_tensor("bias0", data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=gemm_bias_0.flatten())
    linear_bias1 = onnx.helper.make_tensor("layer1_bias",data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_0.flatten())
    linear_bias2 = onnx.helper.make_tensor("layer2_bias",data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_1.flatten())
    linear_bias3 = onnx.helper.make_tensor("layer3_bias",data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_2.flatten())
    linear_bias4 = onnx.helper.make_tensor("layer4_bias",data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=tdnn_bias_3.flatten())
    linear_bias5 = onnx.helper.make_tensor("layer5_bias",data_type=onnx.TensorProto.FLOAT, dims=(916,), vals=tdnn_bias_4.flatten())

    node0 = onnx.helper.make_node('Gemm', inputs = ['in', 'params0', 'bias0'], outputs = ['node0'], name = 'dnn1')
    node0_mul = onnx.helper.make_node('Scale', ['node0'], ['node0_mul'], scale = 0.0009765625)

    node1=onnx.helper.make_node('Tdnn', inputs=['node0_mul', 'layer1_params'],
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
                                scale_params=512
                                )

    relu_node2 = onnx.helper.make_node('Relu', ['tdnn2'], ['relu_out2'])

    node3=onnx.helper.make_node('Tdnn', inputs=['relu_out2', 'layer3_params'],
                                outputs=['tdnn3'],
                                time_offsets=offsets_layer3,
                                bias_params=linear_bias3,
                                scale_params=512
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
    out_shape = (1, 916)
    initializer=[params0, bias0, offsets_layer1,offsets_layer2, offsets_layer3, offsets_layer4, offsets_layer5
            ,linear_params1, linear_params2, linear_params3, linear_params4, linear_params5,
            linear_bias1, linear_bias2, linear_bias3, linear_bias4, linear_bias5]

    graph = onnx.helper.make_graph([node0, node0_mul, node1, relu_node1, node2, relu_node2, node3, relu_node3
                            , node4, relu_node4, node5], "tdnn_bb04p_tdnn_timeoffset",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )

    model=helper.make_model(graph, producer_name='tdnn_bb04p_tdnn_timeoffset')
    with open('./model/tdnn_bb04p_tdnn_timeoffset.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate tdnn_bb04p_tdnn_timeoffset.onnx sucessfully!")



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

def test_model():
    file_name = './model/tdnn_bb04p_tdnn_timeoffset.onnx'
    onnx_model = onnx.load(file_name)
    shape_dict = {}
    shape_dict['in'] = (1, 40)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data = np.round(np.random.rand(50, 40)*100).astype("float32")

    optimize_method_config = [
        "",
    ]

    for config in optimize_method_config:
        build_run_model(mod, params, config, data)


def test_forward_tdnn_timeoffset():
    generate_tdnn_timeoffset_model()
    test_model()

if __name__ == '__main__':
    test_forward_tdnn_timeoffset()
