import numpy as np
import onnx
import datetime
import os
import witin
from witin import *

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

def generate_model():
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_paths = []
    # net1
    # conv1->conv2.1
    # layer 1
    weight1 = np.round(
        np.random.rand(256, 11, 7, 1) * 255).astype("float32") - 128
    bias1 = np.round(
        np.random.rand(256) * 127).astype("float32") * 128
    weight1_tensor = onnx.helper.make_tensor(
        'weight1_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(weight1),
        vals=weight1.flatten())
    bias1_tensor = onnx.helper.make_tensor(
        'bias1_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(bias1),
        vals=bias1.flatten())

    conv1 = onnx.helper.make_node(
        "Conv",
        inputs=['in', 'weight1_tensor', 'bias1_tensor'],
        outputs=['conv1_out'],
        kernel_shape=[7, 1],
        strides=[4, 1],
        pads=[0, 0, 0, 0],
        name="conv1")
    conv1_scale = onnx.helper.make_node("Scale",
                                        inputs=['conv1_out'],
                                        outputs=['conv1_scale'],
                                        scale=0.0009765625)
    conv1_relu = onnx.helper.make_node('Relu', ['conv1_scale'],
                                       ['conv1_relu'])
    node_list = [conv1, conv1_scale, conv1_relu]
    initializer = [weight1_tensor, bias1_tensor]
    
    # layer 2
    weight2 = np.round(
        np.random.rand(98, 256, 1, 1) * 255).astype("float32") - 128
    bias2 = np.round(
        np.random.rand(98) * 127).astype("float32") * 128
    weight2_tensor = onnx.helper.make_tensor(
        'weight2_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(weight2),
        vals=weight2.flatten())
    bias2_tensor = onnx.helper.make_tensor(
        'bias2_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(bias2),
        vals=bias2.flatten())

    conv2 = onnx.helper.make_node(
        "Conv",
        inputs=['conv1_relu', 'weight2_tensor', 'bias2_tensor'],
        outputs=['conv2_out'],
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        name="conv2")
    conv2_scale = onnx.helper.make_node("Scale",
                                        inputs=['conv2_out'],
                                        outputs=['conv2_scale'],
                                        scale=0.0009765625)
    conv2_relu = onnx.helper.make_node('Relu', ['conv2_scale'],
                                            ['conv2_relu'])
    node_list = node_list + [conv2, conv2_scale]
    initializer = initializer + [weight2_tensor, bias2_tensor]

    
    # layer 3
    weight3 = np.round(
        np.random.rand(98, 256, 1, 1) * 255).astype("float32") - 128
    bias3 = np.round(
        np.random.rand(98) * 127).astype("float32") * 128
    weight3_tensor = onnx.helper.make_tensor(
        'weight3_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(weight3),
        vals=weight3.flatten())
    bias3_tensor = onnx.helper.make_tensor(
        'bias3_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(bias3),
        vals=bias3.flatten())

    conv3 = onnx.helper.make_node(
        "Conv",
        inputs=['conv1_relu', 'weight3_tensor', 'bias3_tensor'],
        outputs=['conv3_out'],
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        name="conv3")
    conv3_scale = onnx.helper.make_node("Scale",
                                        inputs=['conv3_out'],
                                        outputs=['conv3_scale'],
                                        scale=0.0009765625)
    node_list = node_list + [conv3, conv3_scale]
    initializer = initializer + [weight3_tensor, bias3_tensor]

    # layer 4
    act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
    act_table_numpy = np.array(act_input1024)
    sigmoid_table_data = sigmoid_q(act_table_numpy, 6, 7)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(1024,),
                                        vals=sigmoid_table_data.flatten())
    node4_act_lut = onnx.helper.make_node('ActLut', ['conv3_scale'], [
                                          'node4_act_lut'], act_type='sigmoid', table_params=act_table)
    node_list = node_list + [node4_act_lut]
    initializer = initializer + [act_table]

    # layer5
    # shift_bit>0 is left shift, <0 is right shift
    node5_mul = onnx.helper.make_node(
        'Mul', inputs=['conv2_scale', 'node4_act_lut'], outputs=['out'], shift_bit=-8)
    node_list = node_list + [node5_mul]
    
    input_shape = (1, 11, 35, 1)
    output_shape = (1, 98, 8, 1) #out
   
    graph = onnx.helper.make_graph(
        node_list,
        "heart_cnn2",
        inputs=[
            onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                               list(input_shape)),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                               list(output_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph,
                                   producer_name='wear_heart_model')

    model_path = "./model/wear_heart_1" + ".onnx"  # timestampStr
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    model_paths.append(model_path)

    # lstm
    # lstm_node1
    lstm_node1_params_1 = np.random.randint(
        -50, 50, size=(256, 384), dtype=np.int32).astype(np.float32)
    lstm_node1_params_2 = np.random.randint(
        -50, 50, size=(256, 128), dtype=np.int32).astype(np.float32)

    lstm_node1_bias_1 = np.random.randint(-50, 50, size=(
        384,), dtype=np.int32).astype(np.float32) * 128
    lstm_node1_bias_2 = np.random.randint(-50, 50, size=(
        128,), dtype=np.int32).astype(np.float32) * 128

    lstm_node1_params1 = onnx.helper.make_tensor("lstm_node1_params1",
                                                 data_type=onnx.TensorProto.FLOAT,
                                                 dims=(256, 384),
                                                 vals=lstm_node1_params_1.flatten())
    lstm_node1_params2 = onnx.helper.make_tensor("lstm_node1_params2",
                                                 data_type=onnx.TensorProto.FLOAT,
                                                 dims=(256, 128),
                                                 vals=lstm_node1_params_2.flatten())

    lstm_node1_bias1 = onnx.helper.make_tensor("lstm_node1_bias1",
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=(384,),
                                               vals=lstm_node1_bias_1.flatten())
    lstm_node1_bias2 = onnx.helper.make_tensor("lstm_node1_bias2",
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=(128,),
                                               vals=lstm_node1_bias_2.flatten())

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
                                         
    lstm_node1 = onnx.helper.make_node(
        'Lstm',
        inputs=['lstm_in', 'lstm_node1_params1', 'lstm_node1_params2',
                'lstm_node1_bias1', 'lstm_node1_bias2'],
        scale_ioft=1024,
        scale_ct=512,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        outputs=['lstm_out'],
        name="lstm_node1")

    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer = [
        lstm_node1_params1, lstm_node1_params2, lstm_node1_bias1, lstm_node1_bias2,
        act_table1, act_table2, act_table3
    ]

    graph = onnx.helper.make_graph(
        [lstm_node1],
        "test_single_lstm",
        inputs=[onnx.helper.make_tensor_value_info(
            "lstm_in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "lstm_out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph, producer_name='test_single_lstm')
    model_path = './model/wear_heart_2.onnx'
    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())
    model_paths.append(model_path)

    print("Generate wear_heart-model sucessfully!")
    return model_paths


def get_tvm_output(model_paths, graph_id):
    """ Generic function to execute and get tvm output"""
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")

    target = 'npu'
    target_host = 'npu'
    graph_def0 = onnx.load(model_paths[0])
    graph_def1 = onnx.load(model_paths[1])

    mods = []
    pms = []
    input_dts = []

    # graph0
    shape_dict = {}
    shape_dict['in'] = (1, 11, 35, 1)  # NCHW
    # shape_dict['conv1_relu'] = (1, 256, 8, 1)
    mod, params = witin_frontend.frontend.from_onnx(graph_def0, shape_dict)
    mods.append(mod)
    pms.append(params)
    # NWHC
    data0 = np.round(np.random.rand(100, 1, 35, 11) * 255).astype("float32")

    input_dt = {}
    input_dt['conv1'] = witin.nd.array(data0)
    input_dts.append(input_dt)

    # graph1
    shape_dict = {}
    shape_dict['lstm_in'] = (1, 128)
    mod, params = witin_frontend.frontend.from_onnx(graph_def1, shape_dict)
    mods.append(mod)
    pms.append(params)
    data1 = np.round(np.random.rand(100, 128) * 255).astype("float32")
    input_dt = {}
    input_dt['lstm_node1'] = witin.nd.array(data1)
    input_dts.append(input_dt)

    build_dir = "./output/output_heart_1block_v2" + os.path.basename(__file__).split(".")[0] + timestampStr
    optimize_method_config = "./model/pipeline/optimize_layer/optimize_config_conv_signed_input_-1.protobuf"

    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mods,
                                                               target=target,
                                                               target_host=target_host,
                                                               params=pms, input_data=input_dts,
                                                               output_dir=build_dir,
                                                               optimize_method_config=optimize_method_config,
                                                               chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data0))
    m.set_input('lstm_in', witin.nd.array(data1))
    # execute
    m.run()

def test_forward_wear_heart_cnn():
    np.random.seed(10)
    model_paths = generate_model()
    tvm_out = get_tvm_output(model_paths, 0)


if __name__ == '__main__':
    test_forward_wear_heart_cnn()
