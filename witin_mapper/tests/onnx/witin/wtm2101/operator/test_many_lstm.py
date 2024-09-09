import numpy as np
import onnx
from onnx import helper, TensorProto
import os
import datetime
import witin
from witin import *

root_dir = './model/lstm_model/'

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

def get_lsmt_activate_table():
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
    # lstm_node1
    lstm_node1_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node1_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node1_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node1_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node1_params1 = helper.make_tensor("lstm_node1_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node1_params_1.flatten())
    lstm_node1_params2 = helper.make_tensor("lstm_node1_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node1_params_2.flatten())

    lstm_node1_bias1 = helper.make_tensor("lstm_node1_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node1_bias_1.flatten())
    lstm_node1_bias2 = helper.make_tensor("lstm_node1_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node1_bias_2.flatten())

    act_table1, act_table2, act_table3 = get_lsmt_activate_table()

    lstm_node1 = helper.make_node(
        'Lstm',
        inputs=['in', 'lstm_node1_params1', 'lstm_node1_params2', 'lstm_node1_bias1', 'lstm_node1_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['out'],
        name="lstm_node1")

    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer = [
        lstm_node1_params1, lstm_node1_params2, lstm_node1_bias1, lstm_node1_bias2,
        act_table1, act_table2, act_table3
    ]

    graph = helper.make_graph(
        [lstm_node1],
        "test_single_lstm",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[
            helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='test_single_lstm')
    with open(root_dir + 'test_single_lstm.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_single_lstm.onnx sucessfully!")

def generate_lstm_model_case2():
    lstm_node1_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node1_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node1_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node1_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node2_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node2_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node2_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node2_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node1_params1 = helper.make_tensor("lstm_node1_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node1_params_1.flatten())
    lstm_node1_params2 = helper.make_tensor("lstm_node1_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node1_params_2.flatten())

    lstm_node1_bias1 = helper.make_tensor("lstm_node1_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node1_bias_1.flatten())
    lstm_node1_bias2 = helper.make_tensor("lstm_node1_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node1_bias_2.flatten())

    lstm_node2_params1 = helper.make_tensor("lstm_node2_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node2_params_1.flatten())
    lstm_node2_params2 = helper.make_tensor("lstm_node2_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node2_params_2.flatten())

    lstm_node2_bias1 = helper.make_tensor("lstm_node2_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node2_bias_1.flatten())
    lstm_node2_bias2 = helper.make_tensor("lstm_node2_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node2_bias_2.flatten())

    act_table1,act_table2, act_table3 = get_lsmt_activate_table()
    # lstm_node1
    lstm_node1 = helper.make_node(
        'Lstm',
        inputs=['in', 'lstm_node1_params1', 'lstm_node1_params2', 'lstm_node1_bias1', 'lstm_node1_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['lstm_node1_out'],
        name="lstm_node1")

    # lstm_node2
    lstm_node2 = helper.make_node(
        'Lstm',
        inputs=['lstm_node1_out', 'lstm_node2_params1', 'lstm_node2_params2', 'lstm_node2_bias1', 'lstm_node2_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['out'],
        name="lstm_node2")

    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer = [
        lstm_node1_params1, lstm_node1_params2, lstm_node1_bias1, lstm_node1_bias2,
        lstm_node2_params1, lstm_node2_params2, lstm_node2_bias1, lstm_node2_bias2,
        act_table1, act_table2, act_table3
    ]

    graph = helper.make_graph(
        [lstm_node1,lstm_node2],
        "test_two_lstm",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[
            helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='test_two_lstm')
    with open(root_dir + 'test_two_lstm.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_two_lstm.onnx sucessfully!")

def generate_lstm_model_case3():
    lstm_node1_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node1_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node1_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node1_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node2_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node2_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node2_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node2_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node3_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node3_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node3_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node3_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node1_params1 = helper.make_tensor("lstm_node1_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node1_params_1.flatten())
    lstm_node1_params2 = helper.make_tensor("lstm_node1_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node1_params_2.flatten())

    lstm_node1_bias1 = helper.make_tensor("lstm_node1_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node1_bias_1.flatten())
    lstm_node1_bias2 = helper.make_tensor("lstm_node1_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node1_bias_2.flatten())

    lstm_node2_params1 = helper.make_tensor("lstm_node2_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node2_params_1.flatten())
    lstm_node2_params2 = helper.make_tensor("lstm_node2_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node2_params_2.flatten())

    lstm_node2_bias1 = helper.make_tensor("lstm_node2_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node2_bias_1.flatten())
    lstm_node2_bias2 = helper.make_tensor("lstm_node2_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node2_bias_2.flatten())

    lstm_node3_params1 = helper.make_tensor("lstm_node3_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node3_params_1.flatten())
    lstm_node3_params2 = helper.make_tensor("lstm_node3_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node3_params_2.flatten())

    lstm_node3_bias1 = helper.make_tensor("lstm_node3_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node3_bias_1.flatten())
    lstm_node3_bias2 = helper.make_tensor("lstm_node3_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node3_bias_2.flatten())

    act_table1,act_table2, act_table3 = get_lsmt_activate_table()
    # lstm_node1
    lstm_node1 = helper.make_node(
        'Lstm',
        inputs=['in', 'lstm_node1_params1', 'lstm_node1_params2', 'lstm_node1_bias1', 'lstm_node1_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['lstm_node1_out'],
        name="lstm_node1")

    # lstm_node2
    lstm_node2 = helper.make_node(
        'Lstm',
        inputs=['lstm_node1_out', 'lstm_node2_params1', 'lstm_node2_params2', 'lstm_node2_bias1', 'lstm_node2_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['lstm_node2_out'],
        name="lstm_node2")

    # lstm_node3
    lstm_node3 = helper.make_node(
        'Lstm',
        inputs=['lstm_node2_out', 'lstm_node3_params1', 'lstm_node3_params2', 'lstm_node3_bias1', 'lstm_node3_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['out'],
        name="lstm_node3")

    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer = [
        lstm_node1_params1, lstm_node1_params2, lstm_node1_bias1, lstm_node1_bias2,
        lstm_node2_params1, lstm_node2_params2, lstm_node2_bias1, lstm_node2_bias2,
        lstm_node3_params1, lstm_node3_params2, lstm_node3_bias1, lstm_node3_bias2,
        act_table1, act_table2, act_table3
    ]

    graph = helper.make_graph(
        [lstm_node1,lstm_node2,lstm_node3],
        "test_many_lstm",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[
            helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='test_many_lstm')
    with open(root_dir + 'test_many_lstm.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_many_lstm.onnx sucessfully!")

def generate_lstm_model_case4():
    lstm_node1_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node1_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node1_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node1_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node2_params_1 = np.random.randint(-50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node2_params_2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node2_bias_1 = np.random.randint(-50, 50, size=(384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node2_bias_2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    gru_node_param_array1 = np.random.randint(-50, 50, size=(256, 256), dtype=np.int32).astype(np.float32)
    gru_node_param_array2 = np.random.randint(-50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    gru_node_param_bias1 = np.random.randint(-50, 50, size=(256,), dtype=np.int32).astype(np.float32) * 128
    gru_node_param_bias2 = np.random.randint(-50, 50, size=(128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node1_params1 = helper.make_tensor("lstm_node1_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node1_params_1.flatten())
    lstm_node1_params2 = helper.make_tensor("lstm_node1_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node1_params_2.flatten())

    lstm_node1_bias1 = helper.make_tensor("lstm_node1_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node1_bias_1.flatten())
    lstm_node1_bias2 = helper.make_tensor("lstm_node1_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node1_bias_2.flatten())

    lstm_node2_params1 = helper.make_tensor("lstm_node2_params1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 384),
                                           vals=lstm_node2_params_1.flatten())
    lstm_node2_params2 = helper.make_tensor("lstm_node2_params2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=lstm_node2_params_2.flatten())

    lstm_node2_bias1 = helper.make_tensor("lstm_node2_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(384,),
                                         vals=lstm_node2_bias_1.flatten())
    lstm_node2_bias2 = helper.make_tensor("lstm_node2_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=lstm_node2_bias_2.flatten())

    gru_node_array1 = helper.make_tensor("gru_node_array1",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 256),
                                           vals=gru_node_param_array1.flatten())
    gru_node_array2 = helper.make_tensor("gru_node_array2",
                                           data_type=TensorProto.FLOAT,
                                           dims=(256, 128),
                                           vals=gru_node_param_array2.flatten())

    gru_node_bias1 = helper.make_tensor("gru_node_bias1",
                                         data_type=TensorProto.FLOAT,
                                         dims=(256,),
                                         vals=gru_node_param_bias1.flatten())
    gru_node_bias2 = helper.make_tensor("gru_node_bias2",
                                         data_type=TensorProto.FLOAT,
                                         dims=(128,),
                                         vals=gru_node_param_bias2.flatten())

    act_table = np.random.randint(-50, 50, size=(3, 256), dtype=np.int32).astype(np.float32)
    act_table = helper.make_tensor("act_table",
                                        data_type=TensorProto.FLOAT,
                                        dims=(3, 256),
                                        vals=act_table.flatten())

    act_table1, act_table2, act_table3 = get_lsmt_activate_table()
    # lstm_node1
    lstm_node1 = helper.make_node(
        'Lstm',
        inputs=['in', 'lstm_node1_params1', 'lstm_node1_params2', 'lstm_node1_bias1', 'lstm_node1_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['lstm_node1_out'],
        name="lstm_node1")

    # lstm_node2
    lstm_node2 = helper.make_node(
        'Lstm',
        inputs=['lstm_node1_out', 'lstm_node2_params1', 'lstm_node2_params2', 'lstm_node2_bias1', 'lstm_node2_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['lstm_node2_out'],
        name="lstm_node2")
    
    act_table_gru = np.random.randint(-50, 50, size=(2, 1024), dtype=np.int32).astype(np.float32)
    act_table_gru = helper.make_tensor("act_table_gru",
                                        data_type=TensorProto.FLOAT,
                                        dims=(2, 256),
                                        vals=act_table_gru.flatten())

    # gru_node
    gru_node1 = helper.make_node(
        'Gru',
        inputs=['lstm_node2_out', 'gru_node_array1', 'gru_node_array2', 'gru_node_bias1', 'gru_node_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table_gru,
        outputs=['out'],
        name="gru_node1")

    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer = [
        lstm_node1_params1, lstm_node1_params2, lstm_node1_bias1, lstm_node1_bias2,
        lstm_node2_params1, lstm_node2_params2, lstm_node2_bias1, lstm_node2_bias2,
        gru_node_array1, gru_node_array2, gru_node_bias1, gru_node_bias2,
        act_table
    ]

    graph = helper.make_graph(
        [lstm_node1,lstm_node2,gru_node1],
        "test_lstm_lstm_gru",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[
            helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='test_lstm_lstm_gru')
    with open(root_dir + 'test_lstm_lstm_gru.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_lstm_lstm_gru.onnx sucessfully!")

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
    # from tvm.contrib import npu_graph_runtime
    # m = npu_graph_runtime.create(npu_graph, "BB04P1")
    # m.set_input('in1', witin.nd.array(data1))
    # # execute
    # m.run()
    # witin_output_list = [m.get_output(i).asnumpy()
    #                         for i in range(1)]

# single lstm
def test_model_case1():
    generate_lstm_model_case1()
    file_name = root_dir + './test_single_lstm.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    data = np.round(np.random.rand(11, 128) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, 128, data)

# two lstm concat
def test_model_case2():
    generate_lstm_model_case2()
    file_name = root_dir + './test_two_lstm.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    data = np.round(np.random.rand(11, 128) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, 128, data)

# lstm + lstm + lstm + optimize
def test_model_case3():
    generate_lstm_model_case3()
    file_name = root_dir + './test_many_lstm.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    opt_config = root_dir + "/optimize_config/optimize_config_lstm.protobuf"
    data = np.round(np.random.rand(11, 128) * 255).astype("float32")
    # data_org = np.round(np.random.rand(11, 128) * 255 - 128).astype("float32")
    # signed_num = 100
    # data0 = data_org[:, 0:signed_num]
    # data1 = data_org[:, signed_num:] + 128
    # data2 = data0 * -1
    # data = np.concatenate((data0, data1, data2), axis=1)
    # # 将数据中小于0的数置为0
    # data[data < 0] = 0
    tvm_out = get_tvm_output(onnx_model, 128, data,opt_config)

# lstm + lstm + gru
def test_model_case4():
    generate_lstm_model_case4()
    file_name = root_dir + './test_lstm_lstm_gru.onnx'
    onnx_model = onnx.load(file_name)
    target = "npu"
    ctx = "npu"
    data = np.round(np.random.rand(11, 128) * 255).astype("float32")
    opt_config = root_dir + "/optimize_config/optimize_config_lstm_gru.protobuf"
    tvm_out = get_tvm_output(onnx_model, 128, data)

def test_lstm():
    np.random.seed(10)
    # test_model_case1()
    test_model_case2()
    # test_model_case3()
    # test_model_case4()

if __name__ == '__main__':
    test_lstm()
