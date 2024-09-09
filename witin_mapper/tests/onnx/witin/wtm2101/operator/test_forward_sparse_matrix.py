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
np.random.seed(100)


def get_tvm_output(graph_def):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 128)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    data1=np.round(np.random.rand(40, 128)*255).astype("float32")
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
                                        optimize_method_config = './model/optimize_layer/optimize_config_sparse_matrix.protobuf',
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
    witin_output_list = [m.get_output(i).asnumpy()
                            for i in range(1)]

def generate_onnx_model():
    gemm_params_1 = np.random.randint(-144, 144, size=(16, 128),dtype=np.int32).astype(np.float32)

    gemm_bias_1 = 128 * np.random.randint(-128, 127, size=(16,),dtype=np.int32).astype(np.float32)

    gemm_params1 = onnx.helper.make_tensor("gemm_params1",
                    data_type=onnx.TensorProto.FLOAT, dims=(128, 16), vals=gemm_params_1.flatten())

    gemm_bias1 = onnx.helper.make_tensor("gemm_bias1",
                    data_type=onnx.TensorProto.FLOAT, dims=(16,), vals=gemm_bias_1.flatten())


    gemm_node1=onnx.helper.make_node('Gemm', inputs=['in', 'gemm_params1', 'gemm_bias1'],
                    outputs=['gemm_out1'], name="gemm_node1")
    gemm_mul_node1 = onnx.helper.make_node('Scale', ['gemm_out1'], ['gemm_mul_node_out1'], scale=0.0009765625)

    gemm_relu_node1 = onnx.helper.make_node('Relu', ['gemm_mul_node_out1'], ['out'])

    in_shape = (1, 128)
    out_shape = (1, 16)
    initializer=[gemm_params1, gemm_bias1]

    graph = onnx.helper.make_graph([ gemm_node1, gemm_mul_node1, gemm_relu_node1], "sparse_matrix",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name='sparse_matrix')
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path = "./model/sparse_matrix_" + timestampStr + ".onnx"
    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())
    print("generate sparse_matrix.onnx sucessfully!")
    return model_path


def test_model(model_path):
    onnx_model = onnx.load(model_path)
    target = "npu"
    ctx = "npu"
    tvm_out = get_tvm_output(onnx_model)


def test_forward_sparse_matrix():
    model_path = generate_onnx_model()
    test_model(model_path)


if __name__ == '__main__':
    test_forward_sparse_matrix()
