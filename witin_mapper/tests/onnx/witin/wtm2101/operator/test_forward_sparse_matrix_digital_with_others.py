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
from fileinput import filename
from importlib.machinery import OPTIMIZED_BYTECODE_SUFFIXES
from pydoc import Helper
from cv2 import FileNode_MAP, setIdentity
import numpy as np
import math
import onnx
from onnx import helper, TensorProto, mapping
import pytest
import scipy
import sys

from scipy.fftpack import shift
import witin
from witin import *
np.random.seed(100)
def get_tvm_output(graph_def,input_shape,data,opt_config):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, input_shape)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        optimize_method_config = opt_config,
                                        chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1")
    m.set_input('in', witin.nd.array(data))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy()
                            for i in range(1)]
    print(witin_output_list)

def generate_cnn_model():
    cnn_weight_params = np.random.randint(-128,127,size=(3,3,5,5),dtype=np.int32).astype(np.float32)
    cnn_bias_params = 128 * np.random.randint(-128,127,size=(3,),dtype=np.int32).astype(np.float32)

    cnn_weight = helper.make_tensor("cnn_weight", data_type=onnx.TensorProto.FLOAT, dims=(3,3,5,5), vals=cnn_weight_params.flatten())
    cnn_bias = helper.make_tensor("cnn_bias", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=cnn_bias_params.flatten())

    cnn_node = helper.make_node("Conv",
                                inputs = ['in','cnn_weight','cnn_bias'],
                                outputs = ['out'],
                                strides = (1,1),
                                kernel_shape = (5,5),
                                padding = (0,0,0,0),
                                name = 'cnn_node')

    in_shape = (1,3,26,26)
    # o_shape = ((self.cnn_input[2] - self.cnn_weight[2]) + 2 * self.padding[0]) / self.stride[1] + 1
    out_shape = (1,3,22,22)

    nodes = [cnn_node]
    name = "cnn_case"
    inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [cnn_weight,cnn_bias]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name='cnn_sparse_matrix_digital')


    with open('./model/cnn_sparse_matrix_digital.onnx',"wb") as of:
        of.write(model.SerializeToString())

def generate_dnn_model(size):
    gemm_params_1 = np.random.randint(-128, 127, size=size, dtype=np.int32).astype(np.float32)
    gemm_bias_1 = 128 * np.random.randint(-128, 127, size=(size[1],),dtype=np.int32).astype(np.float32)

    gemm_params1 = onnx.helper.make_tensor("gemm_params1",
                    data_type=onnx.TensorProto.FLOAT, dims=size, vals=gemm_params_1.flatten())

    gemm_bias1 = onnx.helper.make_tensor("gemm_bias1",
                    data_type=onnx.TensorProto.FLOAT, dims=(size[1],), vals=gemm_bias_1.flatten())


    gemm_node1=onnx.helper.make_node('Gemm', inputs=['in', 'gemm_params1', 'gemm_bias1'],
                    outputs=['gemm_out1'], name="gemm_node1")
    gemm_mul_node1 = onnx.helper.make_node('Scale', ['gemm_out1'], ['gemm_mul_node_out1'], scale=0.0009765625)

    gemm_relu_node1 = onnx.helper.make_node('Relu', ['gemm_mul_node_out1'], ['out'])

    in_shape = (1, size[0])
    out_shape = (1, size[1])
    initializer=[gemm_params1, gemm_bias1]

    graph = onnx.helper.make_graph([ gemm_node1, gemm_mul_node1, gemm_relu_node1], "sparse_matrix",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name='dnn_sparse_matrix_digital')
    with open('./model/dnn_sparse_matrix_digital.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate dnn_sparse_matrix_digital.onnx sucessfully!")

def generate_tdnn_model(size,dims,offset_vals):
    tdnn_weight_params = np.random.randint(-128,127, size=size, dtype=np.int32).astype(np.float32)
    tdnn_bias_params = 128 * np.random.randint(-128,127, size=(size[1],), dtype=np.int32).astype(np.float32)

    offset_layer1 = helper.make_tensor("offset_layer1", data_type=onnx.TensorProto.FLOAT, dims=(dims,), vals=offset_vals)
    tdnn_weight = helper.make_tensor("tdnn_weight", data_type=onnx.TensorProto.FLOAT, dims=size, vals=tdnn_weight_params.flatten())
    tdnn_bias = helper.make_tensor("tdnn_bias", data_type=onnx.TensorProto.FLOAT, dims=(size[1],), vals=tdnn_bias_params.flatten())

    tdnn_node = helper.make_node("Tdnn", inputs=['in', 'tdnn_weight', 'tdnn_bias'],
                                outputs=['tdnn_out'],
                                time_offsets=offset_layer1,
                                bias_params=tdnn_bias,
                                scale_params=1024,
                                name="tdnn_node")
    relu_node = helper.make_node("Relu", ['tdnn_out'], ['out'])

    in_shape = (1,int(size[0]/dims))
    out_shape = (1,size[1])
    initializer=[tdnn_weight,offset_layer1,tdnn_bias]
    graph = helper.make_graph([tdnn_node,relu_node],"tdnn_sparse_matrix_digital",
                            inputs=[helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))],
                            outputs=[helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))],
                            initializer=initializer)

    model = helper.make_model(graph=graph, producer_name='tdnn_sparse_matrix_digital')
    with open('./model/tdnn_sparse_matrix_digital.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate tdnn_sparse_matrix_digital.onnx sucessfully!")

def generate_gru_model(array1_shape,array2_shape):
    gru_weight1_params = np.random.randint(-128,127,size=array1_shape,dtype=np.int32).astype(np.float32)
    gru_weight2_params = np.random.randint(-128,127,size=array2_shape,dtype=np.int32).astype(np.float32)

    gru_bias1_params = 128 * np.random.randint(-128,127,size=(array1_shape[1],),dtype=np.int32).astype(np.float32)
    gru_bias2_params = 128 * np.random.randint(-128,127,size=(array2_shape[1],),dtype=np.int32).astype(np.float32)

    gru_weight1 = helper.make_tensor("gru_weight1",data_type=onnx.TensorProto.FLOAT,dims=array1_shape,vals=gru_weight1_params.flatten())
    gru_weight2 = helper.make_tensor("gru_weight2",data_type=onnx.TensorProto.FLOAT,dims=array2_shape,vals=gru_weight2_params.flatten())

    gru_bias1 = helper.make_tensor("gru_bias1",data_type=onnx.TensorProto.FLOAT,dims=(array1_shape[1],),vals=gru_bias1_params.flatten())
    gru_bias2 = helper.make_tensor("gru_bias2",data_type=onnx.TensorProto.FLOAT,dims=(array2_shape[1],),vals=gru_bias2_params.flatten())

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

    gru_node = helper.make_node("Gru",inputs=['in','gru_weight1','gru_weight2','gru_bias1','gru_bias2'],
                                outputs=['out'],scale_zr=1024,scale_ht=1024,
                                activate_type=['sigmoid','tanh'],
                                activate_table=act_table,shift_bit=[0,4],name='gru_node')
    
    in_shape = (1,array2_shape[1])
    out_shape = (1,array2_shape[1])
    initializer = [gru_weight1,gru_weight2,gru_bias1,gru_bias2,act_table]

    graph = helper.make_graph([gru_node],"gru_sparse_matrix_digital",
                            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
                            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
                            initializer=initializer)
    
    model=helper.make_model(graph, producer_name='gru_sparse_matrix_digital')
    with open('./model/gru_sparse_matrix_digital.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate gru_sparse_matrix_digital.onnx sucessfully!")

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
    lstm_node = helper.make_node("Lstm", inputs=['in','lstm_weight1','lstm_weight2','lstm_bias1','lstm_bias2'],
                                scale_ioft=1024, scale_ct=1024,
                                activate_type=['sigmoid','tanh','tanh'],
                                activate_table=act_table,
                                outputs=['out'],
                                name='lstm_node')
    
    in_shape = (1,4)
    out_shape = (1,4)
    initializer = [lstm_weight1,lstm_weight2,lstm_bias1,lstm_bias2,act_table]
    graph = helper.make_graph([lstm_node],"lstm_sparse_matrix",
                            inputs=[helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))],
                            outputs=[helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))],
                            initializer=initializer)
    model = helper.make_model(graph, producer_name='lstm_sparse_matrix')
    with open('./model/lstm_sparse_matrix.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate lstm_sparse_matrix.onnx sucessfully!")

def generate_dnn_dnn_model(dnn_w1_shape,dnn_w2_shape):
    dnn1_weight_params = np.random.randint(-128,127, size=dnn_w1_shape, dtype=np.int32).astype(np.float32)
    dnn2_weight_params = np.random.randint(-128,127, size=dnn_w2_shape, dtype=np.int32).astype(np.float32)
    dnn1_bias_params = 128 * np.random.randint(-128,127, size=(dnn_w1_shape[1],), dtype=np.int32).astype(np.float32)
    dnn2_bias_params = 128 * np.random.randint(-128,127, size=(dnn_w2_shape[1],), dtype=np.int32).astype(np.float32)

    dnn1_weight = helper.make_tensor("dnn1_weight", data_type=onnx.TensorProto.FLOAT, dims=dnn_w1_shape, vals=dnn1_weight_params.flatten())
    dnn1_bias = helper.make_tensor("dnn1_bias", data_type=onnx.TensorProto.FLOAT, dims=(dnn_w1_shape[1],), vals=dnn1_bias_params.flatten())
    dnn2_weight = helper.make_tensor("dnn2_weight", data_type=onnx.TensorProto.FLOAT, dims=dnn_w2_shape, vals=dnn2_weight_params.flatten())
    dnn2_bais = helper.make_tensor("dnn2_bias", data_type=onnx.TensorProto.FLOAT, dims=(dnn_w2_shape[1],), vals=dnn2_bias_params.flatten())

    dnn_node1 = helper.make_node("Gemm", inputs=['in','dnn1_weight','dnn1_bias'],
                                outputs=['dnn_node1_out'],
                                name="dnn_node1")
    dnn1_mul_node = helper.make_node("Scale", ['dnn_node1_out'], ['dnn1_mul_node_out'], scale=0.0009765625)
    dnn1_relu_node = helper.make_node("Relu", ['dnn1_mul_node_out'], ['dnn1_relu_node_out'])

    dnn_node2 = helper.make_node("Gemm", inputs=['dnn1_relu_node_out','dnn2_weight','dnn2_bias'],
                                outputs=['dnn_node2_out'],
                                name='dnn_node2')
    dnn2_mul_node = helper.make_node("Scale",['dnn_node2_out'],['dnn2_mul_node_out'], scale=0.0009765625)
    dnn2_relu_node = helper.make_node("Relu", ['dnn2_mul_node_out'], ['out'])

    in_shape = (1,dnn_w1_shape[0])
    out_shape = (1,dnn_w2_shape[1])
    initializer=[dnn1_weight, dnn2_weight, dnn1_bias, dnn2_bais]
    graph = helper.make_graph([dnn_node1,dnn1_mul_node,dnn1_relu_node,dnn_node2,dnn2_mul_node,dnn2_relu_node],
    "dnn_dnn_sparse_matrix_digital",
    inputs=[helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))],
    outputs=[helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))],
    initializer=initializer)

    model = helper.make_model(graph=graph,producer_name="dnn_dnn_sparse_matrix_digital")
    with open('./model/dnn_dnn_sparse_matrix_digital.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate dnn_dnn_sparse_matrix_digital.onnx sucessfully!")

def generate_dnn_gru_model(dnn_shape,arr1_shape,arr2_shape):
    dnn_weight_params = np.random.randint(-128,127, size=dnn_shape, dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128,127, size=(dnn_shape[1],), dtype=np.int32).astype(np.float32)

    gru_weight1_params = np.random.randint(-128,127, size=arr1_shape, dtype=np.int32).astype(np.float32)
    gru_weight2_params = np.random.randint(-128,127, size=arr2_shape, dtype=np.int32).astype(np.float32)
    gru_bias1_params = 128 * np.random.randint(-128,127, size=(arr1_shape[1],), dtype=np.int32).astype(np.float32)
    gru_bias2_params = 128 * np.random.randint(-128,127, size=(arr2_shape[1],), dtype=np.int32).astype(np.float32)

    dnn_weight = helper.make_tensor("dnn_weight", data_type=onnx.TensorProto.FLOAT, dims=dnn_shape, vals=dnn_weight_params.flatten())
    dnn_bias = helper.make_tensor("dnn_bias", data_type=onnx.TensorProto.FLOAT, dims=(dnn_shape[1],), vals=dnn_bias_params.flatten())
    gru_weight1 = helper.make_tensor("gru_weight1", data_type=onnx.TensorProto.FLOAT, dims=arr1_shape, vals=gru_weight1_params.flatten())
    gru_weight2 = helper.make_tensor("gru_weight2", data_type=onnx.TensorProto.FLOAT, dims=arr2_shape, vals=gru_weight2_params.flatten())
    gru_bias1 = helper.make_tensor("gru_bias1", data_type=onnx.TensorProto.FLOAT, dims=(arr1_shape[1],), vals=gru_bias1_params.flatten())
    gru_bias2 = helper.make_tensor("gru_bias2", data_type=onnx.TensorProto.FLOAT, dims=(arr2_shape[1],), vals=gru_bias2_params.flatten())

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

    dnn_node = helper.make_node("Gemm", inputs=['in','dnn_weight', 'dnn_bias'],
                                outputs=['dnn_out'],name='dnn_node')
                            
    gru_node = helper.make_node("Gru", inputs=['dnn_out','gru_weight1','gru_weight2','gru_bias1','gru_bias2'],
                                outputs=['out'], scale_ht=1024, scale_zr=1024,
                                activate_type=['sigmoid','tanh'],
                                activate_table=act_table,
                                shift_bit=[0,4],
                                name="gru_node")

    in_shape = (1,dnn_shape[0])
    out_shape = (1,arr2_shape[1])
    initializer = [dnn_weight,dnn_bias,gru_weight1,gru_weight2,gru_bias1,gru_bias2,act_table]

    graph = helper.make_graph(nodes=[dnn_node,gru_node],
                            name="dnn_gru_sparse_matrix_digital",
                            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
                            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
                            initializer=initializer)

    model = helper.make_model(graph, producer_name='dnn_gru_sparse_matrix_digital')
    with open('./model/dnn_gru_sparse_matrix_digital.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate dnn_gru_sparse_matrix_digital.onnx sucessfully!")

def generate_gru_dnn_model(arr1_shape,arr2_shape,dnn_shape):
    dnn_weight_params = np.random.randint(-128,127, size=dnn_shape, dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128,127, size=(dnn_shape[1],), dtype=np.int32).astype(np.float32)
    gru_weight1_params = np.random.randint(-128,127, size=arr1_shape, dtype=np.int32).astype(np.float32)
    gru_weight2_params = np.random.randint(-128,127, size=arr2_shape, dtype=np.int32).astype(np.float32)
    gru_bias1_params = 128 * np.random.randint(-128,127, size=(arr1_shape[1],), dtype=np.int32).astype(np.float32)
    gru_bias2_params = 128 * np.random.randint(-128,127, size=(arr2_shape[1],), dtype=np.int32).astype(np.float32)

    gru_weight1 = helper.make_tensor("gru_weight1", data_type=onnx.TensorProto.FLOAT, dims=arr1_shape, vals=gru_weight1_params.flatten())    
    gru_weight2 = helper.make_tensor("gru_weight2", data_type=onnx.TensorProto.FLOAT, dims=arr2_shape, vals=gru_weight2_params.flatten())
    gru_bias1 = helper.make_tensor("gru_bias1", data_type=onnx.TensorProto.FLOAT, dims=(arr1_shape[1],), vals=gru_bias1_params.flatten())
    gru_bias2 = helper.make_tensor("gru_bias2", data_type=onnx.TensorProto.FLOAT, dims=(arr2_shape[1],), vals=gru_bias2_params.flatten())
    dnn_weight = helper.make_tensor("dnn_weight", data_type=onnx.TensorProto.FLOAT, dims=dnn_shape, vals=dnn_weight_params.flatten())
    dnn_bias = helper.make_tensor("dnn_bias", data_type=onnx.TensorProto.FLOAT, dims=(dnn_shape[1],), vals=dnn_bias_params.flatten())

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

    gru_node = helper.make_node("Gru", inputs=['in','gru_weight1','gru_weight2','gru_bias1','gru_bias2'],
                                outputs=['gru_out'], scale_ht=1024, scale_zr=1024,
                                activate_type=['sigmoid','tanh'],
                                activate_table=act_table,
                                shift_bit=[0,4],
                                name="gru_node")

    dnn_node = helper.make_node("Gemm", inputs=['gru_out','dnn_weight','dnn_bias'],
                                outputs=['out'],name='dnn_node')
    
    in_shape = (1,arr2_shape[1])
    out_shape = (1,dnn_shape[1])
    initializer = [dnn_weight,dnn_bias,gru_weight1,gru_weight2,gru_bias1,gru_bias2,act_table]

    graph = helper.make_graph(nodes=[gru_node,dnn_node],
                            name='gru_dnn_sparse_matrix_digital',
                            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
                            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
                            initializer=initializer)
    
    model = helper.make_model(graph,producer_name='gru_dnn_sparse_matrix_digital')
    with open('./model/gru_dnn_sparse_matrix_digital.onnx','wb') as of:
        of.write(model.SerializeToString())
    
    print("generate gru_dnn_sparse_matrix_digital.onnx sucessfully!")

def generate_dnn_dnn_dnn_model(w1_shape,w2_shape,w3_shape):
    dnn1_weight_params = np.random.randint(-128,127, size=w1_shape, dtype=np.int32).astype(np.float32)
    dnn2_weight_params = np.random.randint(-128,127, size=w2_shape, dtype=np.int32).astype(np.float32)
    dnn3_weight_params = np.random.randint(-128,127, size=w3_shape, dtype=np.int32).astype(np.float32)
    dnn1_bias_params = 128 * np.random.randint(-128,127, size=(w1_shape[1],), dtype=np.int32).astype(np.float32)
    dnn2_bias_params = 128 * np.random.randint(-128,127, size=(w2_shape[1],), dtype=np.int32).astype(np.float32)
    dnn3_bias_params = 128 * np.random.randint(-128,127, size=(w3_shape[1],), dtype=np.int32).astype(np.float32)

    dnn1_weight = helper.make_tensor("dnn1_weight", data_type=onnx.TensorProto.FLOAT, dims=w1_shape, vals=dnn1_weight_params.flatten())
    dnn1_bias = helper.make_tensor("dnn1_bias", data_type=onnx.TensorProto.FLOAT, dims=(w1_shape[1],), vals=dnn1_bias_params.flatten())
    dnn2_weight = helper.make_tensor("dnn2_weight", data_type=onnx.TensorProto.FLOAT, dims=w2_shape, vals=dnn2_weight_params.flatten())
    dnn2_bias = helper.make_tensor("dnn2_bias", data_type=onnx.TensorProto.FLOAT, dims=(w2_shape[1],), vals=dnn2_bias_params.flatten()) 
    dnn3_weight = helper.make_tensor("dnn3_weight", data_type=onnx.TensorProto.FLOAT, dims=w3_shape, vals=dnn3_weight_params.flatten())
    dnn3_bias = helper.make_tensor("dnn3_bias", data_type=onnx.TensorProto.FLOAT, dims=(w3_shape[1],), vals=dnn3_bias_params.flatten())

    dnn_node1 = helper.make_node("Gemm", inputs=['in','dnn1_weight','dnn1_bias'],
                                outputs=['dnn_node1_out'],
                                name="dnn_node1")
    dnn1_relu_node = helper.make_node("Relu", ['dnn_node1_out'], ['dnn1_relu_node_out'])

    dnn_node2 = helper.make_node("Gemm", inputs=['dnn1_relu_node_out','dnn2_weight','dnn2_bias'],
                                outputs=['dnn_node2_out'],
                                name='dnn_node2')
    dnn2_relu_node = helper.make_node("Relu", ['dnn_node2_out'], ['dnn2_relu_node_out'])

    dnn_node3 = helper.make_node("Gemm", inputs=['dnn2_relu_node_out','dnn3_weight','dnn3_bias'],
                                outputs=['dnn_node3_out'],
                                name='dnn_node3')
    dnn3_relu_node = helper.make_node("Relu", ['dnn_node3_out'], ['out'])

    in_shape = (1,w1_shape[0])
    out_shape = (1,w3_shape[1])
    initializer=[dnn1_weight, dnn2_weight, dnn3_weight, dnn3_bias, dnn1_bias, dnn2_bias]
    graph = helper.make_graph([dnn_node1,dnn1_relu_node,dnn_node2,dnn2_relu_node,dnn_node3,dnn3_relu_node],
    "dnn_dnn_dnn_sparse_matrix_digital",
    inputs=[helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))],
    outputs=[helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))],
    initializer=initializer)

    model = helper.make_model(graph=graph,producer_name="dnn_dnn_dnn_sparse_matrix_digital")
    with open('./model/dnn_dnn_dnn_sparse_matrix_digital.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate dnn_dnn_dnn_sparse_matrix_digital.onnx sucessfully!")

cnn_params = [
    # ("cnn", (1,3,7,7), (3,3,3,3), (1,1), (0,0,0,0)),
    ("cnn", (1,3,27,27), (3,3,5,5), (3,3), (0,1,1,0)),
    # ("cnn", (1,3,27,27), (32,3,7,7), (5,5), (0,3,3,0)),
    # ("cnn", (1,5,64,64), (16,5,13,13), (2,2), (0,5,5,0)),
    # ("cnn", (1,5,64,64), (16,5,7,7), (7,7), (0,2,2,0)),
    # ("cnn", (3,3,64,64), (32,3,15,15), (3,3), (0,3,3,0)),
    # ("cnn", (1,5,114,114), (32,5,13,13), (5,5), (0,7,7,0)),
    # ("cnn", (1,5,115,115), (32,5,13,13), (5,5), (0,7,7,0)),
    # ("cnn", (1,3,416,416), (32,3,3,3), (5,5), (0,7,7,0)),
]

def generate_cnn_model():
        cnn_weight_params = np.random.randint(-128,127,size=(3,3,5,5),dtype=np.int32).astype(np.float32)
        cnn_bias_params = 128 * np.random.randint(-127,127,size=(3,),dtype=np.int32).astype(np.float32)

        cnn_weight = helper.make_tensor("cnn_weight", data_type=onnx.TensorProto.FLOAT, dims=(3,3,5,5), vals=cnn_weight_params.flatten())
        cnn_bias = helper.make_tensor("cnn_bias", data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=cnn_bias_params.flatten())
        
        cnn_node = helper.make_node("Conv",
                                    inputs = ['in','cnn_weight','cnn_bias'],
                                    outputs = ['out'],
                                    strides = (3,3),
                                    kernel_shape = (5,5),
                                    padding = (0,1,1,0),
                                    name = 'cnn_node')

        in_shape = (1,3,27,27)
        # o_shape = ((cnn_input[2] - 5) + 2 * padding[0]) / stride[1] + 1
        out_shape = (1,3,9,9)

        nodes = [cnn_node]
        name = "cnn_case"
        inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
        outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
        initializer = [cnn_weight,cnn_bias]

        graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
        model = helper.make_model(graph,producer_name='cnn_sparse_matrix_digital')


        with open('./model/cnn_sparse_matrix_digital.onnx',"wb") as of:
            of.write(model.SerializeToString())

        print("generate cnn_sparse_matrix_digital.onnx sucessfully!")

# @pytest.mark.parametrize("op_type,cnn_input,cnn_weight,stride,padding",cnn_params)
# def test_cnn_model(op_type,cnn_input,cnn_weight,stride,padding):
def test_cnn_model():
    generate_cnn_model()
    filename = './model/cnn_sparse_matrix_digital.onnx'
    data = np.round(np.random.rand(11,3,27,27)*255).astype("float32")
    onnx_model = onnx.load(filename)
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1,3,27,27)
    mod,params = witin_frontend.frontend.from_onnx(onnx_model,shape_dict)
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data)
    opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_cnn.protobuf'
    with witin.transform.PassContext(opt_level=3):
        _,_,_,npu_graph = witin_frontend.build_module.build(mod,
                                                            target=target,
                                                            target_host=target_host,
                                                            params=params,
                                                            input_data=input_dt,
                                                            chip='BB04P1',
                                                            optimize_method_config=opt_config)    
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1")
    m.set_input('in1', witin.nd.array(data))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list)


dnn_spaese_matrix_digital_params = [
    # 2bit
    ((16,8)),
    ((16,128)),
    ((128,32)),
    ((32,64)),
    ((8,64)),
]

# test dnn sparse_martix_digital optimize, all tests OK!
# @pytest.mark.parametrize("size",dnn_spaese_matrix_digital_params)
# def test_dnn_model(size):
#     generate_dnn_model(size)
#     file_name = './model/dnn_sparse_matrix_digital.onnx'
#     onnx_model = onnx.load(file_name)
#     data=np.round(np.random.rand(11, size[0])*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital.protobuf'
#     tvm_out = get_tvm_output(onnx_model,size[0],data,opt_config)

tdnn_sparse_matrix_digital_params = [
    # 2bit
    ((200,40),5,[-2,-1,0,1,2]),
    ((120,40),3,[-1,0,1]),
    ((280,40),7,[-3,-2,-1,0,1,2,3]),
    ((120,40),3,[-1,0,1]),
    ((120,40),3,[0,1,2]),
    ((360,40),9,[-4,-3,-2,-1,0,1,2,3,4]),
    # ((440,40),11,[-5,-4,-3,-2,-1,0,1,2,3,4,5]),

]
# test tdnn optimize sparse_matrix_digital, all tests OK!
# @pytest.mark.parametrize("size,dims,offset_vals",tdnn_sparse_matrix_digital_params)
# def test_tdnn_model(size,dims,offset_vals):
#     generate_tdnn_model(size,dims,offset_vals)
#     file_name = './model/tdnn_sparse_matrix_digital.onnx'
#     onnx_model = onnx.load(file_name)
#     data=np.round(np.random.rand(22, size[0])*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_tdnn.protobuf'
#     tvm_out = get_tvm_output(onnx_model,int(size[0]/dims),data,opt_config)

gru_sparse_matrix_digital_params = [
    ((32,32),(32,16)),
    ((64,64),(64,32)),
    ((16,16),(16,8)),
    ((8,8),(8,4)),
    ((128,128),(128,64)),
    ((256,256),(256,128)),
]
# test gru optimize sparse_matrix_digital, all tests passed!!!!!!
# @pytest.mark.parametrize("array1_shape,array2_shape",gru_sparse_matrix_digital_params)
# def test_gru_model(array1_shape,array2_shape):
#     generate_gru_model(array1_shape,array2_shape)
#     file_name = './model/gru_sparse_matrix_digital.onnx'
#     onnx_model = onnx.load(file_name)
#     data=np.round(np.random.rand(11, array2_shape[1])*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_gru.protobuf'
#     tvm_out = get_tvm_output(onnx_model,array2_shape[1],data,opt_config)

# test lstm optimize sparse_matrix, have some problem, simulator haven't support lstm compare 
# def test_lstm_model():
#     generate_lstm_model()
#     file_name = './model/lstm_sparse_matrix.onnx'
#     onnx_model = onnx.load(file_name)
#     data=np.round(np.random.rand(11, 4)*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_lstm.protobuf'
#     tvm_out = get_tvm_output(onnx_model,4,data,opt_config)

dnn_dnn_sparse_matrix_digital_params = [
    ((8,8),(8,16)),
    ((8,32),(32,16)),
    ((128,64),(64,16)),
    ((32,64),(64,64)),
    ((128,64),(64,64)),
    # failed  the sparse weight lenth should be in 1 ~ 2048, but now is 0
    # ((128,128),(128,64))
]
# test dnn+dnn optimize sparse_matrix, have something problem when open one DIG_MODE
# @pytest.mark.parametrize("dnn_w1_shape,dnn_w2_shape",dnn_dnn_sparse_matrix_digital_params)
# def test_dnn_dnn_model(dnn_w1_shape,dnn_w2_shape):
#     generate_dnn_dnn_model(dnn_w1_shape,dnn_w2_shape)
#     file_name = './model/dnn_dnn_sparse_matrix_digital.onnx'
#     onnx_model = onnx.load(file_name)
#     data=np.round(np.random.rand(11, dnn_w1_shape[0])*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_dnn_dnn.protobuf'
#     tvm_out = get_tvm_output(onnx_model,dnn_w1_shape[0],data,opt_config)

dnn_gru_sparse_matrix_digital_params = [
    ((32,16),(32,32),(32,16)),
    ((16,4),(8,8),(8,4)),
    ((8,4),(8,8),(8,4)),
]
 
# have something problem 
# @pytest.mark.parametrize("dnn_shape,arr1_shape,arr2_shape",dnn_gru_sparse_matrix_digital_params)
# def test_dnn_gru_model(dnn_shape,arr1_shape,arr2_shape):
#     generate_dnn_gru_model(dnn_shape,arr1_shape,arr2_shape)
#     file_name = './model/dnn_gru_sparse_matrix_digital.onnx'
#     onnx_model = onnx.load(file_name)
#     data = np.round(np.random.rand(11,dnn_shape[0])*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_dnn_gru.protobuf'
#     tvm_out = get_tvm_output(onnx_model,dnn_shape[0],data,opt_config)

gru_dnn_sparse_matrix_digital_params = [
    ((32,32),(32,16),(16,8)),
    ((16,16),(16,8),(8,8)),
    ((8,8),(8,4),(4,8)),
    ((8,8),(8,4),(4,4)),
]
# all gru + dnn tests pass !!!
# @pytest.mark.parametrize("arr1_shape,arr2_shape,dnn_shape",gru_dnn_sparse_matrix_digital_params)
# def test_gru_dnn_model(arr1_shape,arr2_shape,dnn_shape):
#     generate_gru_dnn_model(arr1_shape,arr2_shape,dnn_shape)
#     file_name = './model/gru_dnn_sparse_matrix_digital.onnx'
#     onnx_model = onnx.load(file_name)
#     data = np.round(np.random.rand(11,arr2_shape[1])*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_gru_dnn.protobuf'
#     tvm_out = get_tvm_output(onnx_model,arr2_shape[1],data,opt_config)

dnn_x3_sparse_matrix_digital_params = [
    ((64,32),(32,16),(16,8)),
    ((16,8),(8,4),(4,8)),
    ((32,8),(8,16),(16,8)),
]
# test 3 x dnn optimize sparse_matrix, all tests pass!!!!!!!!
# @pytest.mark.parametrize("w1_shape,w2_shape,w3_shape",dnn_x3_sparse_matrix_digital_params)
# def test_dnn_dnn_dnn_model(w1_shape,w2_shape,w3_shape):
#     generate_dnn_dnn_dnn_model(w1_shape,w2_shape,w3_shape)
#     file_name = './model/dnn_dnn_dnn_sparse_matrix_digital.onnx'
#     onnx_model = onnx.load(file_name)
#     data=np.round(np.random.rand(5, w1_shape[0])*255).astype("float32")
#     opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_dnn_dnn_dnn.protobuf'
#     tvm_out = get_tvm_output(onnx_model,w1_shape[0],data,opt_config)

if __name__ == '__main__':
    np.random.seed(10)
    test_cnn_model()
    # test_tdnn_model()
    # test_dnn_model()
    # test_gru_model()
    # test_lstm_model()
    # test_dnn_gru_model()
    # test_gru_dnn_model()
    # test_dnn_dnn_model()
    # test_dnn_dnn_dnn_model()
    
