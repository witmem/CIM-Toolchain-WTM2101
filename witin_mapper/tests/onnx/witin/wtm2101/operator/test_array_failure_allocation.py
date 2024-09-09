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
import witin
import datetime
import pytest
from witin import *
np.random.seed(100)
weight_row_size = 235
weight_col_size = 123
graphs_num = 48


def get_tvm_output():
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    graphs = []
    params = []
    input_datas = []
    for index in range(graphs_num):
        suffix = str(index)
        shape_dict = {}
        shape_dict['in' + suffix] = (1, weight_row_size)
        model = onnx.load("./model/failure_alloc" + suffix + ".onnx")
        graph, param = witin_frontend.frontend.from_onnx(model, shape_dict)
        graphs.append(graph)
        params.append(param)
        data1 = np.round(np.random.rand(40, weight_row_size)*255).astype("float32")
        input_dt = {}
        input_dt['in' + suffix] = witin.nd.array(data1)
        input_datas.append(input_dt)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(graphs,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_datas,
                                        output_dir=build_dir,
                                        chip="BB04P1")

def generate_onnx_model(index):
    suffix = str(index)
    gemm_params_1 = np.random.randint(-144, 144, size=(weight_col_size, weight_row_size), dtype=np.int32).astype(np.float32)

    gemm_bias_1 = 128 * np.random.randint(-128, 127, size=(weight_col_size,), dtype=np.int32).astype(np.float32)

    gemm_params1 = onnx.helper.make_tensor("gemm_params" + suffix,
                    data_type=onnx.TensorProto.FLOAT, dims=(weight_row_size, weight_col_size), vals=gemm_params_1.flatten())

    gemm_bias1 = onnx.helper.make_tensor("gemm_bias" + suffix,
                    data_type=onnx.TensorProto.FLOAT, dims=(weight_col_size,), vals=gemm_bias_1.flatten())


    gemm_node1=onnx.helper.make_node('Gemm', inputs=['in' + suffix, 'gemm_params' + suffix, 'gemm_bias' + suffix],
                    outputs=['gemm_out' + suffix], name="gemm_node" + suffix)
    gemm_mul_node1 = onnx.helper.make_node('Scale', ['gemm_out' + suffix], ['gemm_mul_node_out' + suffix], scale=0.0009765625)

    gemm_relu_node1 = onnx.helper.make_node('Relu', ['gemm_mul_node_out' + suffix], ['out' + suffix])

    in_shape = (1, weight_row_size)
    out_shape = (1, weight_col_size)
    initializer = [gemm_params1, gemm_bias1]

    graph = onnx.helper.make_graph([ gemm_node1, gemm_mul_node1, gemm_relu_node1], "failure_alloc",
            inputs=[onnx.helper.make_tensor_value_info("in" + suffix, onnx.TensorProto.FLOAT, list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out" + suffix, onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model = helper.make_model(graph, producer_name='failure_alloc')
    with open('./model/failure_alloc' + suffix + '.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate failure_alloc.onnx sucessfully!")


@pytest.mark.xfail
def test_failure_alloc():
    for index in range(graphs_num):
        generate_onnx_model(index)
    get_tvm_output()

if __name__ == '__main__':
    test_failure_alloc()
