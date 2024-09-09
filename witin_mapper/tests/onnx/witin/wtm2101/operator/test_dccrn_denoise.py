import onnx
import witin
from witin import *
from tvm.contrib import npu_graph_runtime
import numpy as np
import math
import datetime

data_dir = './model/dccrn_denoise_params/'
onnx_dir = "./model/"


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


def distance(a, b):
    v1 = np.sqrt(np.sum((np.int32(b) - np.int32(a))**2))
    v2 = np.sqrt(np.sum(1e-5 + np.int32(b)**2))
    v3 = v1 / v2
    ret = np.sum(v3)
    # print("compare distance is:%.4f" % (ret))
    return ret


def generate_model():

    conv1_w = np.loadtxt(data_dir + "layer1_conv2d_weight.txt", dtype=np.float32).reshape((48, 2, 5, 2))
    conv1_b = np.loadtxt(data_dir + "layer1_conv2d_bias.txt", dtype=np.float32)
    conv2_w = np.loadtxt(data_dir + "layer2_conv2d_weight.txt", dtype=np.float32).reshape((72, 48, 3, 2))
    conv2_b = np.loadtxt(data_dir + "layer2_conv2d_bias.txt", dtype=np.float32)
    conv3_w = np.loadtxt(data_dir + "layer3_conv2d_weight.txt", dtype=np.float32).reshape((112, 72, 3, 2))
    conv3_b = np.loadtxt(data_dir + "layer3_conv2d_bias.txt", dtype=np.float32)

    lstm4_ifo_w = np.loadtxt(data_dir + "layer4_lstm_weight_ifo.txt", dtype=np.float32).reshape((384, 576))
    lstm4_ifo_w = np.transpose(lstm4_ifo_w)
    lstm4_ifo_b = np.loadtxt(data_dir + "layer4_lstm_bias_ifo.txt", dtype=np.float32)  # 384
    lstm4_c_w = np.loadtxt(data_dir + "layer4_lstm_weight_c.txt", dtype=np.float32).reshape((128, 576))
    lstm4_c_w = np.transpose(lstm4_c_w)
    lstm4_c_b = np.loadtxt(data_dir + "layer4_lstm_bias_c.txt", dtype=np.float32)  #128

    linear5_w = np.loadtxt(data_dir + "layer5_linear_weight.txt", dtype=np.float32).reshape((448, 128))
    linear5_w = np.transpose(linear5_w)
    linear5_b = np.loadtxt(data_dir + "layer5_linear_bias.txt", dtype=np.float32)  # 448

    deconv6_w = np.loadtxt(data_dir + "layer6_deconv2d_weight.txt", dtype=np.float32).reshape((112, 72, 3, 2))
    # deconv6_w = np.transpose(deconv6_w, (1, 0, 2, 3))
    deconv6_b = np.loadtxt(data_dir + "layer6_deconv2d_bias.txt", dtype=np.float32)  # 72
    deconv7_w = np.loadtxt(data_dir + "layer7_deconv2d_weight.txt", dtype=np.float32).reshape((72, 48, 3, 2))
    # deconv7_w = np.transpose(deconv6_w, (1, 0, 2, 3))
    deconv7_b = np.loadtxt(data_dir + "layer7_deconv2d_bias.txt", dtype=np.float32)  #48
    deconv8_w = np.loadtxt(data_dir + "layer8_deconv2d_weight.txt", dtype=np.float32).reshape((48, 2, 5, 2))
    # deconv8_w = np.transpose(deconv6_w, (1, 0, 2, 3))
    deconv8_b = np.loadtxt(data_dir + "layer8_deconv2d_bias.txt", dtype=np.float32)  #2

    onnx_node_list = []
    onnx_initializer = []

    # layer 1
    conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_w),
                                             vals=conv1_w.flatten())
    conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_b),
                                             vals=conv1_b.flatten())

    conv1 = onnx.helper.make_node("Conv",
                                  inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                  outputs=['conv1'],
                                  kernel_shape=[5, 2],
                                  strides=[4, 1],
                                  pads=[0, 1, 1, 0],
                                  name="conv1")
    conv1_scale = onnx.helper.make_node("Scale", inputs=['conv1'], outputs=['conv1_scale'], scale=1.0 / 1024)
    conv1_relu = onnx.helper.make_node('Relu', ['conv1_scale'], ['conv1_relu'])
    onnx_node_list = onnx_node_list + [conv1, conv1_scale, conv1_relu]
    onnx_initializer = onnx_initializer + [conv1_w_tensor, conv1_b_tensor]

    # layer 2
    conv2_w_tensor = onnx.helper.make_tensor('conv2_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_w),
                                             vals=conv2_w.flatten())
    conv2_b_tensor = onnx.helper.make_tensor('conv2_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_b),
                                             vals=conv2_b.flatten())

    conv2 = onnx.helper.make_node("Conv",
                                  inputs=['conv1_relu', 'conv2_w_tensor', 'conv2_b_tensor'],
                                  outputs=['conv2'],
                                  kernel_shape=[3, 2],
                                  strides=[2, 1],
                                  pads=[0, 1, 1, 0],
                                  name="conv2")
    conv2_scale = onnx.helper.make_node("Scale", inputs=['conv2'], outputs=['conv2_scale'], scale=1.0 / 1024)
    conv2_relu = onnx.helper.make_node('Relu', ['conv2_scale'], ['conv2_relu'])
    onnx_node_list = onnx_node_list + [conv2, conv2_scale, conv2_relu]
    onnx_initializer = onnx_initializer + [conv2_w_tensor, conv2_b_tensor]

    # layer 3
    conv3_w_tensor = onnx.helper.make_tensor('conv3_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv3_w),
                                             vals=conv3_w.flatten())
    conv3_b_tensor = onnx.helper.make_tensor('conv3_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv3_b),
                                             vals=conv3_b.flatten())

    conv3 = onnx.helper.make_node("Conv",
                                  inputs=['conv2_relu', 'conv3_w_tensor', 'conv3_b_tensor'],
                                  outputs=['conv3'],
                                  kernel_shape=[3, 2],
                                  strides=[2, 1],
                                  pads=[0, 1, 1, 0],
                                  name="conv3")
    conv3_scale = onnx.helper.make_node("Scale", inputs=['conv3'], outputs=['conv3_scale'], scale=1.0 / 1024)
    conv3_relu = onnx.helper.make_node('Relu', ['conv3_scale'], ['conv3_relu'])
    onnx_node_list = onnx_node_list + [conv3, conv3_scale, conv3_relu]
    onnx_initializer = onnx_initializer + [conv3_w_tensor, conv3_b_tensor]

    block1_input_shape = (1, 2, 64, 4)
    block1_output_shape = (1, 112, 4, 4)

    reshape4_shape = np.array([4,448])
    reshape4_shape = onnx.helper.make_tensor('reshape4_shape',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(reshape4_shape),
                                             vals=reshape4_shape.flatten())

    reshape4_node = onnx.helper.make_node("Reshape",inputs = ['conv3_relu','reshape4_shape'] ,outputs=['reshape4'])
    onnx_node_list = onnx_node_list + [reshape4_node]
    onnx_initializer = onnx_initializer + [reshape4_shape]

    # dccrn_graph = onnx.helper.make_graph(
    #     onnx_node_list,
    #     "Conv_block",
    #     inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(block1_input_shape))],
    #     outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(block1_output_shape))],
    #     initializer=onnx_initializer)
    # dccrn_conv_model = onnx.helper.make_model(dccrn_graph, producer_name='dccrn_denoise_block1_conv')

    # with open(onnx_dir + 'dccrn_denoise_block1_conv.onnx', "wb") as f:
    #     f.write(dccrn_conv_model.SerializeToString())
    # print("Generate dccrn_denoise_block1_conv sucessfully!")

    ###################################################################################################
    # net 2: lstm +FC
    ###################################################################################################

    # onnx_node_list = []
    # onnx_initializer = []
    # layer 4 lstm
    lstm_w_ifo_tensor = onnx.helper.make_tensor('lstm_w_ifo_tensor',
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=np.shape(lstm4_ifo_w),
                                                vals=lstm4_ifo_w.flatten())
    lstm_b_ifo_tensor = onnx.helper.make_tensor('lstm_b_ifo_tensor',
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=np.shape(lstm4_ifo_b),
                                                vals=lstm4_ifo_b.flatten())
    lstm_w_c_tensor = onnx.helper.make_tensor('lstm_w_c_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(lstm4_c_w),
                                              vals=lstm4_c_w.flatten())
    lstm_b_c_tensor = onnx.helper.make_tensor('lstm_b_c_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(lstm4_c_b),
                                              vals=lstm4_c_b.flatten())

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
    lstm_node = onnx.helper.make_node(
        'Lstm',
        inputs=['reshape4', 'lstm_w_ifo_tensor', 'lstm_w_c_tensor', 'lstm_b_ifo_tensor', 'lstm_b_c_tensor'],
        scale_ioft=2048,
        scale_ct=1024,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        shift_bits=[-7, -7],
        outputs=['lstm_out'],
        name="lstm_node")

    onnx_node_list = onnx_node_list + [lstm_node]
    onnx_initializer = onnx_initializer + [
        lstm_w_ifo_tensor, lstm_w_c_tensor, lstm_b_ifo_tensor, lstm_b_c_tensor, act_table1, act_table2, act_table3
    ]

    linear_w_tensor = onnx.helper.make_tensor('linear_w_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear5_w),
                                              vals=linear5_w.flatten())
    linear_b_tensor = onnx.helper.make_tensor('linear_b_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear5_b),
                                              vals=linear5_b.flatten())

    linear5_node = onnx.helper.make_node('Gemm',
                                         inputs=['lstm_out', 'linear_w_tensor', 'linear_b_tensor'],
                                         outputs=['linear5_out'],
                                         name='linear5')
    linear5_scale = onnx.helper.make_node('Scale', ['linear5_out'], ['linear5_scale'], scale=1.0 / 1024)

    onnx_node_list = onnx_node_list + [linear5_node, linear5_scale]
    onnx_initializer = onnx_initializer + [linear_w_tensor, linear_b_tensor]

    block2_input_shape = (4, 448)
    block2_output_shape = (4, 448)

    reshape6_shape = np.array([1,112,4,4])
    reshape6_shape = onnx.helper.make_tensor('reshape6_shape',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(reshape6_shape),
                                             vals=reshape6_shape.flatten())

    reshape6_node = onnx.helper.make_node("Reshape",inputs = ['linear5_scale','reshape6_shape'] ,outputs=['reshape6'])
    onnx_node_list = onnx_node_list + [reshape6_node]
    onnx_initializer = onnx_initializer + [reshape6_shape]

    # dccrn_graph2 = onnx.helper.make_graph(
    #     onnx_node_list,
    #     "lstm_block",
    #     inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(block2_input_shape))],
    #     outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(block2_output_shape))],
    #     initializer=onnx_initializer)
    # dccrn_lstm_model = onnx.helper.make_model(dccrn_graph2, producer_name='dccrn_denoise_block2_lstm')

    # with open(onnx_dir + 'dccrn_denoise_block2_lstm.onnx', "wb") as f:
    #     f.write(dccrn_lstm_model.SerializeToString())
    # print("Generate dccrn_denoise_block2_lstm sucessfully!")

    ###################################################################################################
    # net 3: deconv
    ###################################################################################################
    # onnx_node_list = []
    # onnx_initializer = []

    # layer 6
    deconv6_w_tensor = onnx.helper.make_tensor('deconv6_w_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(deconv6_w),
                                               vals=deconv6_w.flatten())
    deconv6_b_tensor = onnx.helper.make_tensor('deconv6_b_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(deconv6_b),
                                               vals=deconv6_b.flatten())

    deconv6 = onnx.helper.make_node("ConvTranspose",
                                    inputs=['reshape6', 'deconv6_w_tensor', 'deconv6_b_tensor'],
                                    outputs=['deconv6'],
                                    kernel_shape=[3, 2],
                                    strides=[2, 1],
                                    pads=[1, 0],
                                    output_padding=[1, 0],
                                    name="deconv6")
    deconv6_scale = onnx.helper.make_node("Scale", inputs=['deconv6'], outputs=['deconv6_scale'], scale=1.0 / 1024)
    deconv6_relu = onnx.helper.make_node('Relu', ['deconv6_scale'], ['deconv6_relu'])
    onnx_node_list = onnx_node_list + [deconv6, deconv6_scale, deconv6_relu]
    onnx_initializer = onnx_initializer + [deconv6_w_tensor, deconv6_b_tensor]

    # layer 7
    deconv7_w_tensor = onnx.helper.make_tensor('deconv7_w_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(deconv7_w),
                                               vals=deconv7_w.flatten())
    deconv7_b_tensor = onnx.helper.make_tensor('deconv7_b_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(deconv7_b),
                                               vals=deconv7_b.flatten())

    deconv7 = onnx.helper.make_node("ConvTranspose",
                                    inputs=['deconv6_relu', 'deconv7_w_tensor', 'deconv7_b_tensor'],
                                    outputs=['deconv7'],
                                    kernel_shape=[3, 2],
                                    strides=[2, 1],
                                    pads=[1, 0],
                                    output_padding=[1, 0],
                                    name="deconv7")
    deconv7_scale = onnx.helper.make_node("Scale", inputs=['deconv7'], outputs=['deconv7_scale'], scale=1.0 / 1024)
    deconv7_relu = onnx.helper.make_node('Relu', ['deconv7_scale'], ['deconv7_relu'])
    onnx_node_list = onnx_node_list + [deconv7, deconv7_scale, deconv7_relu]
    onnx_initializer = onnx_initializer + [deconv7_w_tensor, deconv7_b_tensor]

    # layer 8
    deconv8_w_tensor = onnx.helper.make_tensor('deconv8_w_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(deconv8_w),
                                               vals=deconv8_w.flatten())
    deconv8_b_tensor = onnx.helper.make_tensor('deconv8_b_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(deconv8_b),
                                               vals=deconv8_b.flatten())

    deconv8 = onnx.helper.make_node("ConvTranspose",
                                    inputs=['deconv7_relu', 'deconv8_w_tensor', 'deconv8_b_tensor'],
                                    outputs=['deconv8'],
                                    kernel_shape=[5, 2],
                                    strides=[4, 1],
                                    pads=[1, 0],
                                    output_padding=[1, 0],
                                    name="deconv8")
    deconv8_scale = onnx.helper.make_node("Scale", inputs=['deconv8'], outputs=['out'], scale=1.0 / 1024)
    onnx_node_list = onnx_node_list + [deconv8, deconv8_scale]
    onnx_initializer = onnx_initializer + [deconv8_w_tensor, deconv8_b_tensor]

    # block3_input_shape = (1, 112, 4, 4)
    # block3_output_shape = (1, 72, 8, 1)  #layer8 out
    # # block3_output_shape = (1, 72, 8, 3)  #layer6 deconv6_relu
    # # block3_output_shape = (1, 48, 16, 2)  #layer7 deconv7_relu

    # dccrn_graph3 = onnx.helper.make_graph(
    #     onnx_node_list,
    #     "deconv_block",
    #     inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(block3_input_shape))],
    #     outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(block3_output_shape))],
    #     initializer=onnx_initializer)
    # dccrn_deconv_model = onnx.helper.make_model(dccrn_graph3, producer_name='dccrn_denoise_block3_deconv')

    # with open(onnx_dir + 'dccrn_denoise_block3_deconv.onnx', "wb") as f:
    #     f.write(dccrn_deconv_model.SerializeToString())

    net_input_shape = (1, 2, 64, 4)
    net_output_shape = (1, 72, 8, 1)
    dccrn_graph = onnx.helper.make_graph(
        onnx_node_list,
        "dccrn_graph",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(net_input_shape))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(net_output_shape))],
        initializer=onnx_initializer)
    dccrn_deconv_model = onnx.helper.make_model(dccrn_graph, producer_name='dccrn_denoise_graph')

    with open(onnx_dir + 'dccrn_denoise_graph.onnx', "wb") as f:
        f.write(dccrn_deconv_model.SerializeToString())

    print("Generate dccrn_denoise_graph sucessfully!")


def test_dccrn_denoise():
    np.random.seed(0)
    generate_model()

    mods = []
    params = []
    input_datas = []

    '''
    ############## block-1 (conv)
    conv_model = onnx.load(onnx_dir + "dccrn_denoise_block1_conv.onnx")
    shape_dict_conv = {}
    shape_dict_conv['in'] = (1, 2, 64, 4)  # NCHW
    mod1, params1 = witin_frontend.frontend.from_onnx(conv_model,
                                                      shape_dict_conv)
    mods.append(mod1)
    params.append(params1)
    # data1 = np.round(np.random.rand(11, 4, 64, 2) * 255).astype("float32")  # NWHC
    data1 = np.loadtxt(data_dir + 'conv1_in.txt').astype("float32")
    data1 = data1.reshape(-1, 1, 64, 2)
    signed_num = 2
    data1_pn1 = data1[:, :, :, 0:signed_num]
    # 后110为无符号数据需要加128将int8转为uint8
    data1_128 = data1[:, :, :, signed_num:] + 128
    # 将前300有符号数据取反
    data1_pn2 = data1_pn1 * -1
    # 将以上三种数据拼接到一起
    data1 = np.concatenate((data1_pn1, data1_128, data1_pn2), axis=3)
    # 将数据中小于0的数置为0
    data1[data1 < 0] = 0
    # print(data1.shape)

    input_dt1 = {}
    input_dt1['in'] = witin.nd.array(data1)
    input_datas.append(input_dt1)

    ############## block-2 (lstm)
    lstm_model = onnx.load(onnx_dir + "dccrn_denoise_block2_lstm.onnx")
    shape_dict_lstm = {}
    shape_dict_lstm['in'] = (1, 448)
    mod2, params2 = witin_frontend.frontend.from_onnx(lstm_model,
                                                      shape_dict_lstm)
    mods.append(mod2)
    params.append(params2)
    # data2 = np.round(np.random.rand(11, 448) * 255).astype("float32")
    data2 = np.loadtxt(data_dir + 'lstm4_in.txt').astype("float32")
    data2 = data2.reshape(-1, 448)
    input_dt2 = {}
    input_dt2['in'] = witin.nd.array(data2)
    input_datas.append(input_dt2)

    ############## block-3 (deconv)
    deconv_model = onnx.load(onnx_dir + "dccrn_denoise_block3_deconv.onnx")
    shape_dict_deconv = {}
    shape_dict_deconv['in'] = (1, 112, 4, 4)
    mod3, params3 = witin_frontend.frontend.from_onnx(deconv_model, shape_dict_deconv)
    mods.append(mod3)
    params.append(params3)
    # data3 = np.round(np.random.rand(11, 4, 4, 112) * 255).astype("float32")
    data3 = np.loadtxt(data_dir + 'deconv6_in.txt').astype("float32")
    # data3 = data3[:112 * 4 * 4]
    # data3 = np.tile(data3, 11)
    data3 = data3.reshape(-1, 1, 4, 112)
    signed_num = 112
    data3_pn1 = data3[:, :, :, 0:signed_num]
    # 后110为无符号数据需要加128将int8转为uint8
    data3_128 = data3[:, :, :, signed_num:] + 128
    # 将前300有符号数据取反
    data3_pn2 = data3_pn1 * -1
    # 将以上三种数据拼接到一起
    data3 = np.concatenate((data3_pn1, data3_128, data3_pn2), axis=3)
    # print(data3[0, :, :, :])
    # 将数据中小于0的数置为0
    data3[data3 < 0] = 0

    input_dt3 = {}
    input_dt3['in'] = witin.nd.array(data3)
    input_datas.append(input_dt3)
    '''
    dccrn_model = onnx.load(onnx_dir + "dccrn_denoise_graph.onnx")
    shape_dict_conv = {}
    shape_dict_conv['in'] = (1, 2, 64, 4)  # NCHW
    mod1, params1 = witin_frontend.frontend.from_onnx(dccrn_model,
                                                      shape_dict_conv)
    mods.append(mod1)
    params.append(params1)
    # data1 = np.round(np.random.rand(11, 4, 64, 2) * 255).astype("float32")  # NWHC
    data = np.loadtxt(data_dir + 'conv1_in.txt').astype("float32")
    data = data.reshape(-1, 1, 64, 2)
    # merge the adjacent 2 frame data, [101, 1, 256, 4] -> [100, 2, 256, 4]
    new_data = np.zeros([data.shape[0] - 1, data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype)
    for n in range(data.shape[0] - 1):
        new_data[n, 0, :, :] = data[n, :, :, :]
        new_data[n, 1, :, :] = data[n + 1, :, :, :]
    # signed_num = 2
    # data1_pn1 = data1[:, :, :, 0:signed_num]
    # # 后110为无符号数据需要加128将int8转为uint8
    # data1_128 = data1[:, :, :, signed_num:] + 128
    # # 将前300有符号数据取反
    # data1_pn2 = data1_pn1 * -1
    # # 将以上三种数据拼接到一起
    # data1 = np.concatenate((data1_pn1, data1_128, data1_pn2), axis=3)
    # # 将数据中小于0的数置为0
    # data1[data1 < 0] = 0
    # print(data1.shape)
    input_dt1 = {}
    input_dt1['in'] = witin.nd.array(new_data)
    input_datas.append(input_dt1)

    ############## build
    target = 'npu'
    target_host = 'npu'
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0]  + timestampStr
    optimize_method_config = data_dir + "optimize_config.protobuf"
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graphs = witin_frontend.build_module.build(
            mods,
            target=target,
            target_host=target_host,
            params=params,
            input_data=input_datas,
            chip="BB04P1",
            output_dir=build_dir,
            optimize_method_config=optimize_method_config)

    # ############## compare result
    # block1_wm_out = np.loadtxt(build_dir + '/layer_debug/net0/3_conv3/round1/output0.txt').astype("float32")
    # # block2_wm_out = np.loadtxt(build_dir + '/layer_debug/net1/round7/output0.txt').astype("float32")
    # # block3_wm_out = np.loadtxt(build_dir + '/layer_debug/net2/round2/output0.txt').astype("float32")
    # block2_wm_out = np.loadtxt(build_dir + '/layer_debug/net0/6_linear5/round10/output0.txt').astype("float32")
    # block3_wm_out = np.loadtxt(build_dir + '/layer_debug/net0/13_deconv8/round13/output0.txt').astype("float32")

    # block1_py_out = np.loadtxt(data_dir + 'conv3_out.txt').astype("float32")
    # block2_py_out = np.loadtxt(data_dir + 'fc5_out.txt').astype("float32")
    # block3_py_out = np.loadtxt(data_dir + 'deconv8_out.txt').astype("float32")

    # for i in range(block1_wm_out.shape[0]):
    #     dis = distance(block1_wm_out[i], block1_py_out[i])
    #     if (dis > 0.1):
    #         print(block1_wm_out[i], block1_py_out[i])
    #         raise Exception("Block1 npu result is not equal to pytorch result!")
    # print("Block1 npu result is equal to pytorch result!")
    # for i in range(block2_wm_out.shape[0]):
    #     dis = distance(block2_wm_out[i], block2_py_out[i])
    #     if (dis > 0.1):
    #         print(block2_wm_out[i], block2_py_out[i])
    #         raise Exception("Block2 npu result is not equal to pytorch result!")
    # print("Block2 npu result is equal to pytorch result!")
    # for i in range(block3_wm_out.shape[0]):
    #     dis = distance(block3_wm_out[i], block3_py_out[i])
    #     if (dis > 0.1):
    #         print(block3_wm_out[i], block3_py_out[i])
    #         raise Exception("Block3 npu result is not equal to pytorch result!")
    # print("Block3 npu result is equal to pytorch result!")

    m = npu_graph_runtime.create(npu_graphs, "BB04P1", output_dir=build_dir,)
    m.set_input('in', witin.nd.array(new_data))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(1)]


if __name__ == "__main__":
    test_dccrn_denoise()