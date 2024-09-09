import numpy as np
import onnx
import os
import witin
from witin import *
import datetime

root_dir = './model/'

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

def generate_lstm_activate_table():
    act_input256 = [i for i in range(128)] + [i - 128 for i in range(128)]
    act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
    act_input256 = np.array(act_input256)
    act_input1024 = np.array(act_input1024)

    sigmoid_table_list = sigmoid_q(act_input1024, 6, 7)
    tanh1_table_list = tanh_q(act_input1024, 8, 7)
    tanh2_table_list = tanh_q(act_input256, 7, 7)

    act_table1 = onnx.helper.make_tensor("act_table1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(1, 1024),
                                         vals=sigmoid_table_list.flatten())
    act_table2 = onnx.helper.make_tensor("act_table2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(1, 1024),
                                         vals=tanh1_table_list.flatten())
    act_table3 = onnx.helper.make_tensor("act_table3",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(1, 256),
                                         vals=tanh2_table_list.flatten())
    return act_table1, act_table2, act_table3

def generate_lstm_model_case1():
    # dnn0 (514 * 128)
    gemm_params_0 = np.random.randint(-50, 50, size=(514, 128), dtype=np.int32).astype(np.float32)
    gemm_bias_0 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128
    # dnn1 (128 * 128)
    gemm_params_1 = np.random.randint(-50, 50, size=(128, 128), dtype=np.int32).astype(np.float32)
    gemm_bias_1 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128
    # lstm
    lstm_params_1 = np.random.randint(-128, 127, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_bias_1 = np.random.randint(-128, 127, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_params_2 = np.random.randint(-128, 127, size=(256, 128), dtype=np.int32).astype(np.float32)
    lstm_bias_2 = np.random.randint(-128, 127, size=(128,), dtype=np.int32).astype(np.float32) * 128
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
    lstm_params1 = onnx.helper.make_tensor("lstm_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_params_1.flatten())
    lstm_params2 = onnx.helper.make_tensor("lstm_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_params_2.flatten())
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
    lstm_bias1 = onnx.helper.make_tensor("lstm_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_bias_1.flatten())
    lstm_bias2 = onnx.helper.make_tensor("lstm_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_bias_2.flatten())
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

    act_table1, act_table2, act_table3 = generate_lstm_activate_table()

    lstm_node1 = onnx.helper.make_node(
        'Lstm',
        inputs=['node1_mul', 'lstm_params1', 'lstm_params2', 'lstm_bias1', 'lstm_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        shift_bits=[-7, -7],
        outputs=['lstm_out'],
        name="lstm_node1")

    # dnn2
    node2 = onnx.helper.make_node('Gemm',
                                  inputs=['lstm_out', 'params2', 'bias2'],
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
        params0, params1, lstm_params1, lstm_params2, params2, params3, bias0, bias1, lstm_bias1,
        lstm_bias2, bias2, bias3, act_table1, act_table2, act_table3
    ]

    graph = onnx.helper.make_graph(
        [node0, node0_mul, node1, node1_mul, lstm_node1, node2, node2_mul, node3, node3_mul],
        "test_lstm_case1",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph, producer_name='test_lstm_case1')
    with open(root_dir + 'test_lstm_case1.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_lstm_case1.onnx sucessfully!")


def generate_lstm_model_case2():
    # lstm
    lstm_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)
    lstm_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128
    # dnn2 (128 * 448)
    gemm_params_2 = np.random.randint(-50, 50, size=(128, 448), dtype=np.int32).astype(np.float32)
    gemm_bias_2 = np.random.randint(-50, 50, size=(448,), dtype=np.int32).astype(np.float32) * 128
    # dnn3 (448 * 128)
    gemm_params_3 = np.random.randint(-50, 50, size=(448, 128), dtype=np.int32).astype(np.float32)
    gemm_bias_3 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_params1 = onnx.helper.make_tensor("lstm_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_params_1.flatten())
    lstm_params2 = onnx.helper.make_tensor("lstm_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_params_2.flatten())
    params2 = onnx.helper.make_tensor("params2",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(128, 448),
                                      vals=gemm_params_2.flatten())
    params3 = onnx.helper.make_tensor("params3",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(448, 128),
                                      vals=gemm_params_3.flatten())

    lstm_bias1 = onnx.helper.make_tensor("lstm_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_bias_1.flatten())
    lstm_bias2 = onnx.helper.make_tensor("lstm_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_bias_2.flatten())
    bias2 = onnx.helper.make_tensor("bias2",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(448,),
                                    vals=gemm_bias_2.flatten())
    bias3 = onnx.helper.make_tensor("bias3",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(128,),
                                    vals=gemm_bias_3.flatten())

    act_table1, act_table2, act_table3 = generate_lstm_activate_table()

    lstm_node1 = onnx.helper.make_node(
        'Lstm',
        inputs=['in', 'lstm_params1', 'lstm_params2', 'lstm_bias1', 'lstm_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['lstm_out'],
        name="lstm_node1")

    # dnn2
    node2 = onnx.helper.make_node('Gemm',
                                  inputs=['lstm_out', 'params2', 'bias2'],
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
        lstm_params1, lstm_params2, params2, params3, lstm_bias1, lstm_bias2, bias2, bias3,
        act_table1, act_table2, act_table3
    ]

    graph = onnx.helper.make_graph(
        [lstm_node1, node2, node2_mul, node3, node3_mul],
        "test_lstm_case2",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph, producer_name='test_lstm_case2')
    with open(root_dir + 'test_lstm_case2.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_lstm_case2.onnx sucessfully!")

def generate_lstm_model_case3_clean_ctht():
    # lstm
    lstm_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)
    lstm_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_params1 = onnx.helper.make_tensor("lstm_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_params_1.flatten())
    lstm_params2 = onnx.helper.make_tensor("lstm_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_params_2.flatten())


    lstm_bias1 = onnx.helper.make_tensor("lstm_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_bias_1.flatten())
    lstm_bias2 = onnx.helper.make_tensor("lstm_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_bias_2.flatten())

    act_table1, act_table2, act_table3 = generate_lstm_activate_table()

    lstm_node1 = onnx.helper.make_node(
        'Lstm',
        inputs=['in', 'lstm_params1', 'lstm_params2', 'lstm_bias1', 'lstm_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        clean_ctht=10,
        outputs=['out'],
        name="lstm_node1")

    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer = [
        lstm_params1, lstm_params2, lstm_bias1, lstm_bias2,
        act_table1, act_table2, act_table3
    ]

    graph = onnx.helper.make_graph(
        [lstm_node1],
        "test_lstm_case3_clean_ctht",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph, producer_name='test_lstm_case3_clean_ctht')
    with open(root_dir + 'test_lstm_case3_clean_ctht.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_lstm_case3_clean_ctht.onnx sucessfully!")


def get_tvm_output(graph_def, in_shape, data, opt_config="", run=True):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, in_shape)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
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
    if run:
        from tvm.contrib import npu_graph_runtime
        m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
        m.set_input('in', witin.nd.array(data))
        # execute
        m.run()
        witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
        return witin_output_list


def distance(a, b):
    v1 = np.sqrt(np.sum((np.int32(b) - np.int32(a))**2))
    v2 =  np.sqrt(np.sum(1e-5 + np.int32(b)**2))
    v3 = v1 / v2
    ret = np.sum(v3)
    # print("compare distance is:%.4f"%(ret))
    return ret

def test_model_case1():
    generate_lstm_model_case1()
    file_name = root_dir + './test_lstm_case1.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    opt = root_dir + "/optimize_layer/optimize_config_lstm_case1.protobuf"
    data = np.round(np.random.rand(111, 514) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, 514, data, opt)


def test_model_case2():
    generate_lstm_model_case2()
    file_name = root_dir + './test_lstm_case2.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    opts = [root_dir + "/optimize_layer/optimize_config_lstm.protobuf",
            root_dir + "/optimize_layer/optimize_config_lstm_repeat_array1.protobuf",
            root_dir + "/optimize_layer/optimize_config_lstm_repeat_array2.protobuf",
            root_dir + "/optimize_layer/optimize_config_lstm_repeat_array12.protobuf"]
    data_org = np.round(np.random.rand(111, 128) * 255 - 128).astype("float32")
    data = data_org
    # signed_num = 100
    # data0 = data_org[:, 0:signed_num]
    # data1 = data_org[:, signed_num:] + 128
    # data2 = data0 * -1
    # data = np.concatenate((data0, data1, data2), axis=1)
    # # 将数据中小于0的数置为0
    # data[data < 0] = 0
    base_out = []
    for opt in opts:
        tvm_out = get_tvm_output(onnx_model, 128, data, opt)
        if len(base_out) == 0:
            base_out = tvm_out[0]
        else:
            ret = distance(base_out, tvm_out[0])
            if 1 - ret < 0.95:
                print(base_out)
                print(tvm_out[0])
                raise ValueError("similarity is  '%f' " % (ret))



def test_lstm_case3_clean_ctht():
    # clean ct ht only support function forward for debug data
    generate_lstm_model_case3_clean_ctht()
    file_name = root_dir + 'test_lstm_case3_clean_ctht.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(111, 128) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, 128, data, run=False)

def test_lstm():
    np.random.seed(10)
    test_model_case1()
    test_model_case2()
    test_lstm_case3_clean_ctht()


if __name__ == '__main__':
    test_lstm()
