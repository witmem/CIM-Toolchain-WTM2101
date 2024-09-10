import numpy as np
import onnx
from onnx import helper, TensorProto
import os
import witin
import datetime
from witin import *

root_dir = './model/pipeline/gru_model/'

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

def generate_gru_model_case1():
    # dnn0 (514 * 128)
    gemm_params_0 = np.random.randint(-50, 50, size=(514, 128), dtype=np.int32).astype(np.float32)
    gemm_bias_0 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128
    # dnn1 (128 * 128)
    gemm_params_1 = np.random.randint(-50, 50, size=(128, 128), dtype=np.int32).astype(np.float32)
    gemm_bias_1 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128
    # gru
    gru_params_1 = np.random.randint(-128, 127, size=(256, 256), dtype=np.int32).astype(np.float32)
    gru_bias_1 = np.random.randint(-128, 127, size=(256,), dtype=np.int32).astype(np.float32) * 128
    gru_params_2 = np.random.randint(-128, 127, size=(256, 128), dtype=np.int32).astype(np.float32)
    gru_bias_2 = np.random.randint(-128, 127, size=(128,), dtype=np.int32).astype(np.float32) * 128
    # dnn2 (128 * 448)
    gemm_params_2 = np.random.randint(-50, 50, size=(128, 448), dtype=np.int32).astype(np.float32)
    gemm_bias_2 = np.random.randint(-50, 50, size=(448,), dtype=np.int32).astype(np.float32) * 128
    # dnn3 (448 * 128)
    gemm_params_3 = np.random.randint(-50, 50, size=(448, 128), dtype=np.int32).astype(np.float32)
    gemm_bias_3 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    params0 = onnx.helper.make_tensor("params0",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(514, 128),
                                      vals=gemm_params_0.flatten())
    params1 = onnx.helper.make_tensor("params1",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(128, 128),
                                      vals=gemm_params_1.flatten())
    gru_params1 = onnx.helper.make_tensor("gru_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 256),
                                           vals=gru_params_1.flatten())
    gru_params2 = onnx.helper.make_tensor("gru_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=gru_params_2.flatten())
    params2 = onnx.helper.make_tensor("params2",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(128, 448),
                                      vals=gemm_params_2.flatten())
    params3 = onnx.helper.make_tensor("params3",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(448, 128),
                                      vals=gemm_params_3.flatten())

    bias0 = onnx.helper.make_tensor("bias0",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(128,),
                                    vals=gemm_bias_0.flatten())
    bias1 = onnx.helper.make_tensor("bias1",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(128,),
                                    vals=gemm_bias_1.flatten())
    gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(256,),
                                         vals=gru_bias_1.flatten())
    gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=gru_bias_2.flatten())
    bias2 = onnx.helper.make_tensor("bias2",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(448,),
                                    vals=gemm_bias_2.flatten())
    bias3 = onnx.helper.make_tensor("bias3",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(128,),
                                    vals=gemm_bias_3.flatten())

    # dnn0
    node0 = onnx.helper.make_node('Gemm',
                                  inputs=['in', 'params0', 'bias0'],
                                  outputs=['node0'],
                                  name='dnn0')
    node0_mul = onnx.helper.make_node('Scale', ['node0'], ['node0_mul'], scale=0.0009765625)

    # dnn1
    node1 = onnx.helper.make_node('Gemm',
                                  inputs=['node0_mul', 'params1', 'bias1'],
                                  outputs=['node1'],
                                  name='dnn1')
    node1_mul = onnx.helper.make_node('Scale', ['node1'], ['node1_mul'], scale=0.0009765625)

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

    gru_node1 = onnx.helper.make_node(
        'Gru',
        inputs=['node1_mul', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[-8, -8],
        outputs=['gru_out'],
        name="gru_node1")

    # dnn2
    node2 = onnx.helper.make_node('Gemm',
                                  inputs=['gru_out', 'params2', 'bias2'],
                                  outputs=['node2'],
                                  name='dnn2')
    node2_mul = onnx.helper.make_node('Scale', ['node2'], ['node2_mul'], scale=0.0009765625)

    # dnn3
    node3 = onnx.helper.make_node('Gemm',
                                  inputs=['node2_mul', 'params3', 'bias3'],
                                  outputs=['node3'],
                                  name='dnn3')
    node3_mul = onnx.helper.make_node('Scale', ['node3'], ['out'], scale=0.0009765625)

    in_shape = (1, 514)
    out_shape = (1, 128)
    initializer = [
        params0, params1, gru_params1, gru_params2, params2, params3, bias0, bias1, gru_bias1,
        gru_bias2, bias2, bias3, act_table
    ]

    graph = onnx.helper.make_graph(
        [node0, node0_mul, node1, node1_mul, gru_node1, node2, node2_mul, node3, node3_mul],
        "test_gru_case1",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='test_gru_case1')
    with open(root_dir + 'test_gru_case1.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_gru_case1.onnx sucessfully!")


def generate_gru_model_case2():
    # gru
    gru_params_1 = np.random.randint(-50, 50, size=(256, 256), dtype=np.int32).astype(np.float32)
    gru_bias_1 = np.random.randint(-50, 50, size=(256,), dtype=np.int32).astype(np.float32) * 128
    gru_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)
    gru_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128
    # dnn2 (128 * 448)
    gemm_params_2 = np.random.randint(-50, 50, size=(128, 448), dtype=np.int32).astype(np.float32)
    gemm_bias_2 = np.random.randint(-50, 50, size=(448,), dtype=np.int32).astype(np.float32) * 128
    # dnn3 (448 * 128)
    gemm_params_3 = np.random.randint(-50, 50, size=(448, 128), dtype=np.int32).astype(np.float32)
    gemm_bias_3 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    gru_params1 = onnx.helper.make_tensor("gru_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 256),
                                           vals=gru_params_1.flatten())
    gru_params2 = onnx.helper.make_tensor("gru_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=gru_params_2.flatten())
    params2 = onnx.helper.make_tensor("params2",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(128, 448),
                                      vals=gemm_params_2.flatten())
    params3 = onnx.helper.make_tensor("params3",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(448, 128),
                                      vals=gemm_params_3.flatten())

    gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(256,),
                                         vals=gru_bias_1.flatten())
    gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=gru_bias_2.flatten())
    bias2 = onnx.helper.make_tensor("bias2",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(448,),
                                    vals=gemm_bias_2.flatten())
    bias3 = onnx.helper.make_tensor("bias3",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(128,),
                                    vals=gemm_bias_3.flatten())
    
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

    gru_node1 = onnx.helper.make_node(
        'Gru',
        inputs=['in', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        outputs=['gru_out'],
        name="lstm_node1")

    # dnn2
    node2 = onnx.helper.make_node('Gemm',
                                  inputs=['gru_out', 'params2', 'bias2'],
                                  outputs=['node2'],
                                  name='dnn2')
    node2_mul = onnx.helper.make_node('Scale', ['node2'], ['node2_mul'], scale=0.0009765625)

    # dnn3
    node3 = onnx.helper.make_node('Gemm',
                                  inputs=['node2_mul', 'params3', 'bias3'],
                                  outputs=['node3'],
                                  name='dnn3')
    node3_mul = onnx.helper.make_node('Scale', ['node3'], ['out'], scale=0.0009765625)

    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer = [
        gru_params1, gru_params2, params2, params3, gru_bias1, gru_bias2, bias2, bias3,
        act_table
    ]

    graph = onnx.helper.make_graph(
        [gru_node1, node2, node2_mul, node3, node3_mul],
        "test_gru_case2",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='test_gru_case2')
    with open(root_dir + 'test_gru_case2.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_gru_case2.onnx sucessfully!")


def get_tvm_output(graph_def, in_shape, data, opt_config=""):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, in_shape)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target=target,
                                                               target_host=target_host,
                                                               params=params,
                                                               input_data=input_dt,
                                                               chip="BB04P1",
                                                               output_dir=build_dir,
                                                               optimize_method_config=opt_config)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in1', witin.nd.array(data))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]


def test_model_case1():
    generate_gru_model_case1()
    file_name = root_dir + './test_gru_case1.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    data = np.round(np.random.rand(111, 514) * 255).astype("float32")
    opt_config_files = [root_dir + "/optimize_config_gru.protobuf",
           root_dir + "/optimize_config_gru_partial_PN.protobuf"]
    for opt_config in opt_config_files:
      tvm_out = get_tvm_output(onnx_model, 514, data, opt_config)

def test_model_case2():
    generate_gru_model_case2()
    file_name = root_dir + './test_gru_case2.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    opt = root_dir + "/optimize_config_lstm.protobuf"
    data = np.round(np.random.rand(11, 128) * 255 - 128).astype("float32")
    
    tvm_out = get_tvm_output(onnx_model, 128, data, opt)

def test_gru_forward():
    np.random.seed(10)
    test_model_case1()
    test_model_case2()

if __name__ == '__main__':
    test_gru_forward()
