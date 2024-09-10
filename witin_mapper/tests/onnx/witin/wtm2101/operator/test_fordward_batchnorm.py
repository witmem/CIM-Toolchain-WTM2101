
import onnx
import numpy as np
import torch.nn as nn
import os
import datetime
import witin
from witin import *

onnx_dir = './model/inner/'
if not os.path.exists(onnx_dir):
    os.mkdir(onnx_dir)

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

def tanh_q(x, qx, qy):
    x = x / (2**qx)
    x = np.tanh(x)
    x = (x * (2**qy)).round().clip(-128, 127)
    return x

def sigmoid_q(x, qx, qy):
    x = x / (2**qx)
    x = sigmoid(x)
    x = (x * (2**qy)).round().clip(-128, 127)
    return x

def generate_batchnorm_model():
    #linear1
    linear1_w = np.random.randint(-128, 127, size=(170, 64), dtype=np.int32).astype(np.float32)
    linear1_b = np.random.randint(-128, 127, size=(64,), dtype=np.int32).astype(np.float32) * 128

    linear1_w_tensor = onnx.helper.make_tensor('linear1_w_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear1_w),
                                              vals=linear1_w.flatten())
    linear1_b_tensor = onnx.helper.make_tensor('linear1_b_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear1_b),
                                              vals=linear1_b.flatten())
    
    #linear1_bn   
    linear1_scale = np.ones(64).astype(np.float32)
    linear1_bias = np.zeros(64).astype(np.float32)
    linear1_input_mean = np.zeros(64).astype(np.float32)
    linear1_input_var = np.ones(64).astype(np.float32)

    linear1_scale_tensor = onnx.helper.make_tensor("linear1_scale",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(64,),
                                           vals=linear1_scale)
    linear1_bias_tensor = onnx.helper.make_tensor("linear1_bias",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(64,),
                                           vals=linear1_bias)
    linear1_input_mean_tensor = onnx.helper.make_tensor("linear1_input_mean",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(64,),
                                           vals=linear1_input_mean)
    linear1_input_var_tensor = onnx.helper.make_tensor("linear1_input_var",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(64,),
                                           vals=linear1_input_var)
    
    #gru1 
    input_size = 64
    output_size = 36
    gru_weight_1 = np.random.randint(-128, 127, size=(input_size + output_size, 2 * output_size), dtype=np.int32).astype(np.float32)
    gru_weight_2 = np.random.randint(-128, 127, size=(output_size, output_size), dtype=np.int32).astype(np.float32)
    gru_weight_3 = np.random.randint(-128, 127, size=(input_size, output_size), dtype=np.int32).astype(np.float32)
    gru_bias_1 = np.random.randint(-128, 127, size=(2 * output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_bias_2 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_bias_3 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_weight1 = onnx.helper.make_tensor("gru_weight1",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(input_size + output_size, 2 * output_size),
                                          vals=gru_weight_1.flatten())
    gru_weight2 = onnx.helper.make_tensor("gru_weight2",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(output_size, output_size),
                                          vals=gru_weight_2.flatten())
    gru_weight3 = onnx.helper.make_tensor("gru_weight3",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(input_size, output_size),
                                          vals=gru_weight_3.flatten())
    gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2 * output_size,),
                                        vals=gru_bias_1.flatten())
    gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(output_size,),
                                        vals=gru_bias_2.flatten())
    gru_bias3 = onnx.helper.make_tensor("gru_bias3",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(output_size,),
                                        vals=gru_bias_3.flatten())
    act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
    sigmoid_act_table = np.array(act_input1024)
    sigmoid_table_data = sigmoid_q(sigmoid_act_table, 6, 7)
    tanh_act_table = np.array(act_input1024)
    tanh_table_data = tanh_q(tanh_act_table, 8, 7)
    act_table = np.concatenate((sigmoid_table_data, tanh_table_data), axis=0)
    act_table = np.reshape(act_table, (2, 1024)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 1024),
                                        vals=act_table.flatten())
    
    #gru1_bn    
    gru1_scale = np.ones(36).astype(np.float32)
    gru1_bias = np.zeros(36).astype(np.float32)
    gru1_input_mean = np.zeros(36).astype(np.float32)
    gru1_input_var = np.ones(36).astype(np.float32)

    gru1_scale_tensor = onnx.helper.make_tensor("gru1_scale",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(36,),
                                           vals=gru1_scale)
    gru1_bias_tensor = onnx.helper.make_tensor("gru1_bias",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(36,),
                                           vals=gru1_bias)
    gru1_input_mean_tensor = onnx.helper.make_tensor("gru1_input_mean",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(36,),
                                           vals=gru1_input_mean)
    gru1_input_var_tensor = onnx.helper.make_tensor("gru1_input_var",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(36,),
                                           vals=gru1_input_var)  

    linear1_node = onnx.helper.make_node('Gemm',
                                         inputs=['in', 'linear1_w_tensor', 'linear1_b_tensor'],
                                         outputs=['linear1_out'],
                                         name='linear1')

    linear1_scale_node = onnx.helper.make_node('Scale', ['linear1_out'], ['linear1_scale_out'], scale=1.0 / 1024)

    linear1_bn_node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['linear1_scale_out', 'linear1_scale', 'linear1_bias', 
                'linear1_input_mean', 'linear1_input_var'],
        outputs=['linear1_bn_out'],
        epsilon=1e-5,
        scale_in=0.125,
        scale_out=0.125
    )
    node1 = [linear1_node] + [linear1_scale_node] + [linear1_bn_node]
    initializer1 = [linear1_w_tensor, linear1_b_tensor, linear1_scale_tensor, 
                    linear1_bias_tensor, linear1_input_mean_tensor, linear1_input_var_tensor]
    
    gru1_node = onnx.helper.make_node('Gru',
                                      inputs=['linear1_bn_out', 'gru_weight1', 'gru_weight2', 'gru_weight3', 
                                              'gru_bias1', 'gru_bias2', 'gru_bias3'],
                                      scale_zr=1024,
                                      scale_ht=1024,
                                      scale_in=1024,
                                      scale_ones = 127,
                                      activate_type=['sigmoid', 'tanh'],
                                      activate_table=act_table,
                                      shift_bits=[-8, -8],
                                      clean_ht=0,
                                      outputs=['gru1_out'],
                                      name="gru1_node")
    

    gru1_bn_node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['gru1_out', 'gru1_scale', 'gru1_bias', 
                'gru1_input_mean', 'gru1_input_var'],
        outputs=['out'],
        epsilon=1e-5,
        scale_in=0.125,
        scale_out=0.125
    )

    node2 = [gru1_node] + [gru1_bn_node]
    initializer2 = [gru_weight1, gru_weight2, gru_weight3, gru_bias1, gru_bias2, gru_bias3, act_table, gru1_scale_tensor, 
                    gru1_bias_tensor, gru1_input_mean_tensor, gru1_input_var_tensor]

    in_shape = (1, 170)
    out_shape = (1, 36)
    node = node1 + node2
    initializer = initializer1 + initializer2

    graph = onnx.helper.make_graph(
        node,
        "test_batchnorm_case",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph, producer_name='test_batchnorm_case')
    with open(onnx_dir + 'test_batchnorm_case.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_batchnorm_case.onnx sucessfully!")

def run_batchnorm():
    onnx_model = onnx.load(onnx_dir + "test_batchnorm_case.onnx")
    shape_dict_add = {}
    shape_dict_add['in'] = (1, 170)  #NCHW
    mod, param = witin_frontend.frontend.from_onnx(onnx_model, shape_dict_add)    
    data =  np.random.randint(-128, 127, size=(11, 170), dtype='int8').astype('float32')
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)

    opt_config = onnx_dir + 'batchnorm_case.protobuf'

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    # build
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=param,
                                                               input_data=input_dt,
                                                               chip='BB04P1',
                                                               output_dir=build_dir,
                                                               optimize_method_config=opt_config)
    # execute
    # from tvm.contrib import npu_graph_runtime
    # m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    # m.set_input("in", witin.nd.array(data))
    # m.run()


def test_forward_batchnorm():
    generate_batchnorm_model()
    run_batchnorm()

if __name__ == "__main__":
  test_forward_batchnorm()
