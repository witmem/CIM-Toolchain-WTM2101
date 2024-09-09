import onnx
import numpy as np
import witin
from witin import *
import os

import google.protobuf
import google.protobuf.text_format
import sys
sys.path.append(r"tests/onnx/witin/wtm2101/")
from engine.auto_optimizer import Optimizer

np.random.seed(0)

onnx_dir = './model/pipeline/auto_optimizer_model/auto_optimizer_tc_resnet/'
if not os.path.exists(onnx_dir):
    os.mkdir(onnx_dir)

def npu_graph_build(mods, params, inputs, optimize_config):
    target = 'npu'
    target_host = 'npu'
    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestamp
    optimize_path = ""
    if optimize_config and optimize_config.ByteSize() > 0:
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        optimize_path = os.path.join(build_dir, "optimize_dccrn_hw.protobuf")
        with open(optimize_path, "w") as f:
            txt_opti = google.protobuf.text_format.MessageToString(optimize_config)
            f.write(txt_opti)

    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(
            mods, target=target, target_host=target_host, params=params,
            input_data=inputs, optimize_method_config=optimize_path,
            output_dir=build_dir, chip="BB04P1")
    assert npu_graph is not None
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    for input_dict in inputs:
        for key, data in input_dict.items():
            m.set_input(key, data)

    return m



def generate_model_left():
    #layer 1
    conv1_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv1_w_left.npy').astype('float32')
    conv1_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv1_b_left.npy').astype('float32')

    #layer 2
    conv2_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv2_w_left.npy').astype('float32')
    conv2_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv2_b_left.npy').astype('float32')
    # conv2_b_left = conv2_b_left * 128
    #layer 3
    conv3_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv3_w_left.npy').astype('float32')
    conv3_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv3_b_left.npy').astype('float32')

    #layer 4
    conv4_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv4_w_left.npy').astype('float32')
    conv4_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv4_b_left.npy').astype('float32')

    #layer 5
    conv5_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv5_w_left.npy').astype('float32')
    conv5_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv5_b_left.npy').astype('float32')

    #layer 6
    conv6_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv6_w_left.npy').astype('float32')
    conv6_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv6_b_left.npy').astype('float32')

    #layer 7
    conv7_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv7_w_left.npy').astype('float32')
    conv7_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv7_b_left.npy').astype('float32')

    #layer 8
    conv8_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv8_w_left.npy').astype('float32')
    conv8_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv8_b_left.npy').astype('float32')

    #layer 9
    conv9_w_left_new = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv9_w_left.npy').astype('float32')
    conv9_w_left = np.zeros((48, 32, 2, 2), dtype=float)
    conv9_w_left[:, :, :1, :1] = conv9_w_left_new
    conv9_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv9_b_left.npy').astype('float32')

    #layer 10
    conv10_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv10_w_left.npy').astype('float32')
    conv10_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv10_b_left.npy').astype('float32')

    #layer 11
    conv11_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv11_w_left.npy').astype('float32')
    conv11_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv11_b_left.npy').astype('float32')

    #layer 12
    conv12_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv12_w_left.npy').astype('float32')
    conv12_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv12_b_left.npy').astype('float32')

    #layer 13
    conv13_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv13_w_left.npy').astype('float32')
    conv13_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv13_b_left.npy').astype('float32')

    #layer 14
    conv14_w_left_new = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv14_w_left.npy').astype('float32')
    conv14_w_left = np.zeros((96, 48, 2, 1), dtype=float)
    conv14_w_left[:, :, :1, :] = conv14_w_left_new
    conv14_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv14_b_left.npy').astype('float32')

    #layer 15
    conv15_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv15_w_left.npy').astype('float32')
    conv15_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv15_b_left.npy').astype('float32')

    #layer 16
    conv16_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv16_w_left.npy').astype('float32')
    conv16_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/left/conv16_b_left.npy').astype('float32')

    #onnx.helper.make_tensor
    #layer1
    conv1_w_left_tensor = onnx.helper.make_tensor('conv1_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv1_w_left),
                                                  vals=conv1_w_left.flatten())
    conv1_b_left_tensor = onnx.helper.make_tensor('conv1_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv1_b_left),
                                                  vals=conv1_b_left.flatten())
    # layer2
    conv2_w_left_tensor = onnx.helper.make_tensor('conv2_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv2_w_left),
                                                  vals=conv2_w_left.flatten())
    conv2_b_left_tensor = onnx.helper.make_tensor('conv2_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv2_b_left),
                                                  vals=conv2_b_left.flatten())
    # layer3
    conv3_w_left_tensor = onnx.helper.make_tensor('conv3_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv3_w_left),
                                                  vals=conv3_w_left.flatten())
    conv3_b_left_tensor = onnx.helper.make_tensor('conv3_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv3_b_left),
                                                  vals=conv3_b_left.flatten())
    #layer4
    conv4_w_left_tensor = onnx.helper.make_tensor('conv4_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv4_w_left),
                                                  vals=conv4_w_left.flatten())
    conv4_b_left_tensor = onnx.helper.make_tensor('conv4_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv4_b_left),
                                                  vals=conv4_b_left.flatten())

    #layer5
    conv5_w_left_tensor = onnx.helper.make_tensor('conv5_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv5_w_left),
                                                  vals=conv5_w_left.flatten())
    conv5_b_left_tensor = onnx.helper.make_tensor('conv5_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv5_b_left),
                                                  vals=conv5_b_left.flatten())
    # layer6
    conv6_w_left_tensor = onnx.helper.make_tensor('conv6_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv6_w_left),
                                                  vals=conv6_w_left.flatten())
    conv6_b_left_tensor = onnx.helper.make_tensor('conv6_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv6_b_left),
                                                  vals=conv6_b_left.flatten())
    # layer7
    conv7_w_left_tensor = onnx.helper.make_tensor('conv7_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv7_w_left),
                                                  vals=conv7_w_left.flatten())
    conv7_b_left_tensor = onnx.helper.make_tensor('conv7_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv7_b_left),
                                                  vals=conv7_b_left.flatten())
    #layer8
    conv8_w_left_tensor = onnx.helper.make_tensor('conv8_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv8_w_left),
                                                  vals=conv8_w_left.flatten())
    conv8_b_left_tensor = onnx.helper.make_tensor('conv8_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv8_b_left),
                                                  vals=conv8_b_left.flatten())
    #layer9
    conv9_w_left_tensor = onnx.helper.make_tensor('conv9_w_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv9_w_left),
                                                  vals=conv9_w_left.flatten())
    conv9_b_left_tensor = onnx.helper.make_tensor('conv9_b_left_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv9_b_left),
                                                  vals=conv9_b_left.flatten())
    # layer10
    conv10_w_left_tensor = onnx.helper.make_tensor('conv10_w_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv10_w_left),
                                                   vals=conv10_w_left.flatten())
    conv10_b_left_tensor = onnx.helper.make_tensor('conv10_b_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv10_b_left),
                                                   vals=conv10_b_left.flatten())
    # layer11
    conv11_w_left_tensor = onnx.helper.make_tensor('conv11_w_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv11_w_left),
                                                   vals=conv11_w_left.flatten())
    conv11_b_left_tensor = onnx.helper.make_tensor('conv11_b_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv11_b_left),
                                                   vals=conv11_b_left.flatten())
    #layer12
    conv12_w_left_tensor = onnx.helper.make_tensor('conv12_w_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv12_w_left),
                                                   vals=conv12_w_left.flatten())
    conv12_b_left_tensor = onnx.helper.make_tensor('conv12_b_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv12_b_left),
                                                   vals=conv12_b_left.flatten())
    #layer13
    conv13_w_left_tensor = onnx.helper.make_tensor('conv13_w_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv13_w_left),
                                                   vals=conv13_w_left.flatten())
    conv13_b_left_tensor = onnx.helper.make_tensor('conv13_b_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv13_b_left),
                                                   vals=conv13_b_left.flatten())
    # layer14
    conv14_w_left_tensor = onnx.helper.make_tensor('conv14_w_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv14_w_left),
                                                   vals=conv14_w_left.flatten())
    conv14_b_left_tensor = onnx.helper.make_tensor('conv14_b_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv14_b_left),
                                                   vals=conv14_b_left.flatten())
    # layer15
    conv15_w_left_tensor = onnx.helper.make_tensor('conv15_w_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv15_w_left),
                                                   vals=conv15_w_left.flatten())
    conv15_b_left_tensor = onnx.helper.make_tensor('conv15_b_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv15_b_left),
                                                   vals=conv15_b_left.flatten())
    #layer16
    conv16_w_left_tensor = onnx.helper.make_tensor('conv16_w_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv16_w_left),
                                                   vals=conv16_w_left.flatten())
    conv16_b_left_tensor = onnx.helper.make_tensor('conv16_b_left_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv16_b_left),
                                                   vals=conv16_b_left.flatten())

    #make node
    conv1_left_node = onnx.helper.make_node("Conv",
                                            inputs=['left_in', 'conv1_w_left_tensor', 'conv1_b_left_tensor'],
                                            outputs=['conv1_left'],
                                            kernel_shape=[3, 3],
                                            strides=[2, 2],
                                            pads=[1, 1, 1, 1],
                                            name="conv_left_prescale_1")
    conv1_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv1_left'],
                                                  outputs=['conv1_left_scale'],
                                                  scale=1.0 / 1024)
    conv1_left_relu_node = onnx.helper.make_node('Relu', ['conv1_left_scale'], ['conv1_left_relu'])

    node1_left = [conv1_left_node] + [conv1_left_scale_node] + [conv1_left_relu_node]
    initializer1_left = [conv1_w_left_tensor, conv1_b_left_tensor]

    conv2_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv1_left_relu', 'conv2_w_left_tensor', 'conv2_b_left_tensor'],
                                            outputs=['conv2_left'],
                                            kernel_shape=[3, 3],
                                            strides=[1, 1],
                                            pads=[1, 1, 1, 1],
                                            name="conv_left_prescale_2")
    conv2_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv2_left'],
                                                  outputs=['conv2_left_scale'],
                                                  scale=1.0 / 1024)
    conv2_left_relu_node = onnx.helper.make_node('Relu', ['conv2_left_scale'], ['conv2_left_relu'])

    node2_left = [conv2_left_node] + [conv2_left_scale_node] + [conv2_left_relu_node]
    initializer2_left = [conv2_w_left_tensor, conv2_b_left_tensor]

    conv3_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv2_left_relu', 'conv3_w_left_tensor', 'conv3_b_left_tensor'],
                                            outputs=['conv3_left'],
                                            kernel_shape=[3, 3],
                                            strides=[1, 1],
                                            pads=[1, 1, 1, 1],
                                            name="conv_left_prescale_3")
    conv3_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv3_left'],
                                                  outputs=['conv3_left_scale_1'],
                                                  scale=1.0 / 1024)
    conv3_left_scale_1_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv3_left_scale_1'],
                                                    outputs=['conv3_left_scale'],
                                                    scale=1.0 / 1024)

    node3_left = [conv3_left_node] + [conv3_left_scale_node] + [conv3_left_scale_1_node]
    initializer3_left = [conv3_w_left_tensor, conv3_b_left_tensor]

    conv4_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv1_left_relu', 'conv4_w_left_tensor', 'conv4_b_left_tensor'],
                                            outputs=['conv4_left'],
                                            kernel_shape=[1, 1],
                                            strides=[1, 1],
                                            pads=[0, 0, 0, 0],
                                            name="conv_left_prescale_4")
    conv4_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv4_left'],
                                                  outputs=['conv4_left_scale'],
                                                  scale=1.0 / 1024)
    # conv4_left_scale_1_node = onnx.helper.make_node("Scale", inputs=['conv4_left_scale_o'], outputs=['conv4_left_scale'], scale=1.0)

    conv3_4_left_add_node = onnx.helper.make_node('Add', ['conv3_left_scale', 'conv4_left_scale'], ['conv3_4_left_add'])

    conv3_4_relu_node = onnx.helper.make_node('Relu', ['conv3_4_left_add'], ['conv3_4_left_relu'])

    node4_left = [conv4_left_node] + [conv4_left_scale_node] + [conv3_4_left_add_node] + [conv3_4_relu_node]
    initializer4_left = [conv4_w_left_tensor, conv4_b_left_tensor]

    conv5_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv3_4_left_relu', 'conv5_w_left_tensor', 'conv5_b_left_tensor'],
                                            outputs=['conv5_left'],
                                            kernel_shape=[3, 3],
                                            strides=[1, 1],
                                            pads=[1, 1, 1, 1],
                                            name="conv_left_prescale_5")
    conv5_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv5_left'],
                                                  outputs=['conv5_left_scale'],
                                                  scale=1.0 / 1024)
    conv5_left_relu_node = onnx.helper.make_node('Relu', ['conv5_left_scale'], ['conv5_left_relu'])

    node5_left = [conv5_left_node] + [conv5_left_scale_node] + [conv5_left_relu_node]
    initializer5_left = [conv5_w_left_tensor, conv5_b_left_tensor]

    conv6_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv5_left_relu', 'conv6_w_left_tensor', 'conv6_b_left_tensor'],
                                            outputs=['conv6_left'],
                                            kernel_shape=[3, 3],
                                            strides=[1, 1],
                                            pads=[1, 1, 1, 1],
                                            name="conv_left_prescale_6")
    conv6_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv6_left'],
                                                  outputs=['conv6_left_scale_1'],
                                                  scale=1.0 / 1024)
    conv6_left_scale_0_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv6_left_scale_1'],
                                                    outputs=['conv6_left_scale'],
                                                    scale=2.0)
    # conv6_left_scale_1_node = onnx.helper.make_node("Scale", inputs=['conv3_4_left_relu'], outputs=['conv6_left_scale_o'], scale=1.0)
    conv6_left_add_node = onnx.helper.make_node('Add', ['conv3_4_left_relu', 'conv6_left_scale'], ['conv6_left_add'])

    conv6_left_relu_node = onnx.helper.make_node('Relu', ['conv6_left_add'], ['conv6_left_relu'])

    node6_left = [conv6_left_node] + [conv6_left_scale_node] + [conv6_left_scale_0_node] + [conv6_left_add_node
                                                                                           ] + [conv6_left_relu_node]
    initializer6_left = [conv6_w_left_tensor, conv6_b_left_tensor]

    conv7_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv6_left_relu', 'conv7_w_left_tensor', 'conv7_b_left_tensor'],
                                            outputs=['conv7_left'],
                                            kernel_shape=[3, 3],
                                            strides=[2, 2],
                                            pads=[1, 1, 1, 1],
                                            name="conv_left_prescale_7")
    conv7_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv7_left'],
                                                  outputs=['conv7_left_scale'],
                                                  scale=1.0 / 1024)

    conv7_left_relu_node = onnx.helper.make_node('Relu', ['conv7_left_scale'], ['conv7_left_relu'])

    node7_left = [conv7_left_node] + [conv7_left_scale_node] + [conv7_left_relu_node]
    initializer7_left = [conv7_w_left_tensor, conv7_b_left_tensor]

    conv8_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv7_left_relu', 'conv8_w_left_tensor', 'conv8_b_left_tensor'],
                                            outputs=['conv8_left'],
                                            kernel_shape=[3, 3],
                                            strides=[1, 1],
                                            pads=[1, 1, 1, 1],
                                            name="conv_left_prescale_8")
    conv8_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv8_left'],
                                                  outputs=['conv8_left_scale_1'],
                                                  scale=1.0 / 1024)
    conv8_left_scale_1_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv8_left_scale_1'],
                                                    outputs=['conv8_left_scale'],
                                                    scale=2.0)

    node8_left = [conv8_left_node] + [conv8_left_scale_node] + [conv8_left_scale_1_node]
    initializer8_left = [conv8_w_left_tensor, conv8_b_left_tensor]

    conv9_left_node = onnx.helper.make_node("Conv",
                                            inputs=['conv6_left_relu', 'conv9_w_left_tensor', 'conv9_b_left_tensor'],
                                            outputs=['conv9_left'],
                                            kernel_shape=[2, 2],
                                            strides=[2, 2],
                                            pads=[0, 0, 0, 0],
                                            name="conv_left_prescale_9")
    conv9_left_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv9_left'],
                                                  outputs=['conv9_left_scale'],
                                                  scale=1.0 / 1024)
    # conv9_left_scale_1_node = onnx.helper.make_node("Scale", inputs=['conv9_left_scale_o'], outputs=['conv9_left_scale'], scale=1.0)
    conv8_9_left_add_node = onnx.helper.make_node('Add', ['conv8_left_scale', 'conv9_left_scale'], ['conv8_9_left_add'])

    conv8_9_left_relu_node = onnx.helper.make_node('Relu', ['conv8_9_left_add'], ['conv8_9_left_relu'])

    node9_left = [conv9_left_node] + [conv9_left_scale_node] + [conv8_9_left_add_node] + [conv8_9_left_relu_node]
    initializer9_left = [conv9_w_left_tensor, conv9_b_left_tensor]

    conv10_left_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv8_9_left_relu', 'conv10_w_left_tensor', 'conv10_b_left_tensor'],
        outputs=['conv10_left'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv_left_prescale_10")
    conv10_left_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv10_left'],
                                                   outputs=['conv10_left_scale'],
                                                   scale=1.0 / 1024)

    conv10_left_relu_node = onnx.helper.make_node('Relu', ['conv10_left_scale'], ['conv10_left_relu'])

    node10_left = [conv10_left_node] + [conv10_left_scale_node] + [conv10_left_relu_node]
    initializer10_left = [conv10_w_left_tensor, conv10_b_left_tensor]

    conv11_left_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv10_left_relu', 'conv11_w_left_tensor', 'conv11_b_left_tensor'],
        outputs=['conv11_left'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv_left_prescale_11")
    conv11_left_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv11_left'],
                                                   outputs=['conv11_left_scale_1'],
                                                   scale=1.0 / 1024)
    conv11_left_scale_0_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv11_left_scale_1'],
                                                     outputs=['conv11_left_scale'],
                                                     scale=4.0)
    # conv11_left_scale_1_node = onnx.helper.make_node("Scale", inputs=['conv8_9_left_relu'], outputs=['conv11_left_scale_o'], scale=1.0)
    conv11_left_add_node = onnx.helper.make_node('Add', ['conv8_9_left_relu', 'conv11_left_scale'], ['conv11_left_add'])

    conv11_left_relu_node = onnx.helper.make_node('Relu', ['conv11_left_add'], ['conv11_left_relu'])

    node11_left = [conv11_left_node] + [conv11_left_scale_node] + [conv11_left_scale_0_node
                                                                  ] + [conv11_left_add_node] + [conv11_left_relu_node]
    initializer11_left = [conv11_w_left_tensor, conv11_b_left_tensor]

    conv12_left_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv11_left_relu', 'conv12_w_left_tensor', 'conv12_b_left_tensor'],
        outputs=['conv12_left'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        name="conv_left_prescale_12")
    conv12_left_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv12_left'],
                                                   outputs=['conv12_left_scale'],
                                                   scale=1.0 / 1024)
    conv12_left_relu_node = onnx.helper.make_node('Relu', ['conv12_left_scale'], ['conv12_left_relu'])

    node12_left = [conv12_left_node] + [conv12_left_scale_node] + [conv12_left_relu_node]
    initializer12_left = [conv12_w_left_tensor, conv12_b_left_tensor]

    conv13_left_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv12_left_relu', 'conv13_w_left_tensor', 'conv13_b_left_tensor'],
        outputs=['conv13_left'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv_left_prescale_13")
    conv13_left_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv13_left'],
                                                   outputs=['conv13_left_scale_1'],
                                                   scale=1.0 / 1024)
    conv13_left_scale_1_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv13_left_scale_1'],
                                                     outputs=['conv13_left_scale'],
                                                     scale=3.0)
    node13_left = [conv13_left_node] + [conv13_left_scale_node] + [conv13_left_scale_1_node]
    initializer13_left = [conv13_w_left_tensor, conv13_b_left_tensor]

    conv14_left_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv11_left_relu', 'conv14_w_left_tensor', 'conv14_b_left_tensor'],
        outputs=['conv14_left'],
        kernel_shape=[2, 1],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
        name="conv_left_prescale_14")
    conv14_left_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv14_left'],
                                                   outputs=['conv14_left_scale'],
                                                   scale=1.0 / 1024)
    # conv14_left_scale_1_node = onnx.helper.make_node("Scale", inputs=['conv14_left_scale_o'], outputs=['conv14_left_scale'], scale=1.0)
    conv13_14_left_add_node = onnx.helper.make_node('Add', ['conv13_left_scale', 'conv14_left_scale'],
                                                    ['conv13_14_left_add'])

    conv13_14_left_relu_node = onnx.helper.make_node('Relu', ['conv13_14_left_add'], ['conv13_14_left_relu'])

    node14_left = [conv14_left_node] + [conv14_left_scale_node] + [conv13_14_left_add_node] + [conv13_14_left_relu_node]
    initializer14_left = [conv14_w_left_tensor, conv14_b_left_tensor]

    conv15_left_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv13_14_left_relu', 'conv15_w_left_tensor', 'conv15_b_left_tensor'],
        outputs=['conv15_left'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv_left_prescale_15")
    conv15_left_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv15_left'],
                                                   outputs=['conv15_left_scale'],
                                                   scale=1.0 / 1024)
    conv15_left_relu_node = onnx.helper.make_node('Relu', ['conv15_left_scale'], ['conv15_left_relu'])

    node15_left = [conv15_left_node] + [conv15_left_scale_node] + [conv15_left_relu_node]
    initializer15_left = [conv15_w_left_tensor, conv15_b_left_tensor]

    conv16_left_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv15_left_relu', 'conv16_w_left_tensor', 'conv16_b_left_tensor'],
        outputs=['conv16_left'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv_left_prescale_16")
    conv16_left_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv16_left'],
                                                   outputs=['conv16_left_scale_1'],
                                                   scale=1.0 / 1024)
    conv16_left_scale_0_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv16_left_scale_1'],
                                                     outputs=['conv16_left_scale'],
                                                     scale=0.5)
    # conv16_left_scale_1_node = onnx.helper.make_node("Scale", inputs=['conv13_14_left_relu'], outputs=['conv16_left_scale_o'], scale=1.0)
    conv16_left_add_node = onnx.helper.make_node('Add', ['conv13_14_left_relu', 'conv16_left_scale'],
                                                 ['conv16_left_add'])

    conv16_left_relu_node = onnx.helper.make_node('Relu', ['conv16_left_add'], ['conv16_left_add_relu'])

    left_ave_pooling_node = onnx.helper.make_node(
        "AveragePool",
        inputs=["conv16_left_add_relu"],
        outputs=["out"],
        kernel_shape=[35, 2],
        pads=[0, 0, 0, 0],  #[top,left,bottom,right]
        strides=[1, 1],
        scale_in=0.0625,
        scale_out=0.0625)

    node16_left = [conv16_left_node] + [conv16_left_scale_node] + [conv16_left_scale_0_node] + [
        conv16_left_add_node
    ] + [conv16_left_relu_node] + [left_ave_pooling_node]
    initializer16_left = [conv16_w_left_tensor, conv16_b_left_tensor]

    nodes_left = node1_left + node2_left + node3_left + node4_left + node5_left + node6_left + node7_left + node8_left \
                + node9_left + node10_left + node11_left + node12_left + node13_left + node14_left + node15_left + node16_left
    initializers_left = initializer1_left + initializer2_left + initializer3_left + initializer4_left + initializer5_left \
                   + initializer6_left + initializer7_left + initializer8_left + initializer9_left + initializer10_left \
                   + initializer11_left + initializer12_left + initializer13_left + initializer14_left + initializer15_left \
                   + initializer16_left

    input_shape = (1, 8, 280, 12)
    output_shape = (1, 96, 1, 1)

    dccrn_graph = onnx.helper.make_graph(
        nodes_left,
        "hw_new_cnn_net_left",
        inputs=[onnx.helper.make_tensor_value_info("left_in", onnx.TensorProto.FLOAT, list(input_shape))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape))],
        initializer=initializers_left)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='hw_new_cnn_net_left')

    with open(onnx_dir + '/hw_new_cnn_net_left.onnx', "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate hw_new_cnn_net_left sucessfully!")


def generate_model_right():
    #layer 1
    conv1_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv1_w_right.npy').astype('float32')
    conv1_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv1_b_right.npy').astype('float32')

    #layer 2
    conv2_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv2_w_right.npy').astype('float32')
    conv2_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv2_b_right.npy').astype('float32')
    # conv2_b_right = conv2_b_right * 128
    #layer 3
    conv3_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv3_w_right.npy').astype('float32')
    conv3_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv3_b_right.npy').astype('float32')

    #layer 4
    conv4_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv4_w_right.npy').astype('float32')
    conv4_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv4_b_right.npy').astype('float32')

    #layer 5
    conv5_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv5_w_right.npy').astype('float32')
    conv5_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv5_b_right.npy').astype('float32')

    #layer 6
    conv6_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv6_w_right.npy').astype('float32')
    conv6_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv6_b_right.npy').astype('float32')

    #layer 7
    conv7_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv7_w_right.npy').astype('float32')
    conv7_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv7_b_right.npy').astype('float32')

    #layer 8
    conv8_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv8_w_right.npy').astype('float32')
    conv8_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv8_b_right.npy').astype('float32')

    #layer 9
    conv9_w_right_new = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv9_w_right.npy').astype('float32')
    conv9_w_right = np.zeros((48, 32, 2, 2), dtype=float)
    conv9_w_right[:, :, :1, :1] = conv9_w_right_new
    conv9_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv9_b_right.npy').astype('float32')

    #layer 10
    conv10_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv10_w_right.npy').astype('float32')
    conv10_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv10_b_right.npy').astype('float32')

    #layer 11
    conv11_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv11_w_right.npy').astype('float32')
    conv11_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv11_b_right.npy').astype('float32')

    #layer 12
    conv12_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv12_w_right.npy').astype('float32')
    conv12_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv12_b_right.npy').astype('float32')

    #layer 13
    conv13_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv13_w_right.npy').astype('float32')
    conv13_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv13_b_right.npy').astype('float32')

    #layer 14
    conv14_w_right_new = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv14_w_right.npy').astype(
        'float32')
    conv14_w_right = np.zeros((96, 48, 2, 1), dtype=float)
    conv14_w_right[:, :, :1, :1] = conv14_w_right_new
    conv14_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv14_b_right.npy').astype('float32')

    #layer 15
    conv15_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv15_w_right.npy').astype('float32')
    conv15_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv15_b_right.npy').astype('float32')

    #layer 16
    conv16_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv16_w_right.npy').astype('float32')
    conv16_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/conv16_b_right.npy').astype('float32')

    #onnx.helper.make_tensor
    #layer1
    conv1_w_right_tensor = onnx.helper.make_tensor('conv1_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv1_w_right),
                                                   vals=conv1_w_right.flatten())
    conv1_b_right_tensor = onnx.helper.make_tensor('conv1_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv1_b_right),
                                                   vals=conv1_b_right.flatten())
    # layer2
    conv2_w_right_tensor = onnx.helper.make_tensor('conv2_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv2_w_right),
                                                   vals=conv2_w_right.flatten())
    conv2_b_right_tensor = onnx.helper.make_tensor('conv2_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv2_b_right),
                                                   vals=conv2_b_right.flatten())
    # layer3
    conv3_w_right_tensor = onnx.helper.make_tensor('conv3_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv3_w_right),
                                                   vals=conv3_w_right.flatten())
    conv3_b_right_tensor = onnx.helper.make_tensor('conv3_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv3_b_right),
                                                   vals=conv3_b_right.flatten())
    #layer4
    conv4_w_right_tensor = onnx.helper.make_tensor('conv4_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv4_w_right),
                                                   vals=conv4_w_right.flatten())
    conv4_b_right_tensor = onnx.helper.make_tensor('conv4_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv4_b_right),
                                                   vals=conv4_b_right.flatten())

    #layer5
    conv5_w_right_tensor = onnx.helper.make_tensor('conv5_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv5_w_right),
                                                   vals=conv5_w_right.flatten())
    conv5_b_right_tensor = onnx.helper.make_tensor('conv5_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv5_b_right),
                                                   vals=conv5_b_right.flatten())
    # layer6
    conv6_w_right_tensor = onnx.helper.make_tensor('conv6_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv6_w_right),
                                                   vals=conv6_w_right.flatten())
    conv6_b_right_tensor = onnx.helper.make_tensor('conv6_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv6_b_right),
                                                   vals=conv6_b_right.flatten())
    # layer7
    conv7_w_right_tensor = onnx.helper.make_tensor('conv7_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv7_w_right),
                                                   vals=conv7_w_right.flatten())
    conv7_b_right_tensor = onnx.helper.make_tensor('conv7_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv7_b_right),
                                                   vals=conv7_b_right.flatten())
    #layer8
    conv8_w_right_tensor = onnx.helper.make_tensor('conv8_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv8_w_right),
                                                   vals=conv8_w_right.flatten())
    conv8_b_right_tensor = onnx.helper.make_tensor('conv8_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv8_b_right),
                                                   vals=conv8_b_right.flatten())
    #layer9
    conv9_w_right_tensor = onnx.helper.make_tensor('conv9_w_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv9_w_right),
                                                   vals=conv9_w_right.flatten())
    conv9_b_right_tensor = onnx.helper.make_tensor('conv9_b_right_tensor',
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=np.shape(conv9_b_right),
                                                   vals=conv9_b_right.flatten())
    # layer10
    conv10_w_right_tensor = onnx.helper.make_tensor('conv10_w_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv10_w_right),
                                                    vals=conv10_w_right.flatten())
    conv10_b_right_tensor = onnx.helper.make_tensor('conv10_b_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv10_b_right),
                                                    vals=conv10_b_right.flatten())
    # layer11
    conv11_w_right_tensor = onnx.helper.make_tensor('conv11_w_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv11_w_right),
                                                    vals=conv11_w_right.flatten())
    conv11_b_right_tensor = onnx.helper.make_tensor('conv11_b_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv11_b_right),
                                                    vals=conv11_b_right.flatten())
    #layer12
    conv12_w_right_tensor = onnx.helper.make_tensor('conv12_w_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv12_w_right),
                                                    vals=conv12_w_right.flatten())
    conv12_b_right_tensor = onnx.helper.make_tensor('conv12_b_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv12_b_right),
                                                    vals=conv12_b_right.flatten())
    #layer13
    conv13_w_right_tensor = onnx.helper.make_tensor('conv13_w_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv13_w_right),
                                                    vals=conv13_w_right.flatten())
    conv13_b_right_tensor = onnx.helper.make_tensor('conv13_b_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv13_b_right),
                                                    vals=conv13_b_right.flatten())
    # layer14
    conv14_w_right_tensor = onnx.helper.make_tensor('conv14_w_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv14_w_right),
                                                    vals=conv14_w_right.flatten())
    conv14_b_right_tensor = onnx.helper.make_tensor('conv14_b_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv14_b_right),
                                                    vals=conv14_b_right.flatten())
    # layer15
    conv15_w_right_tensor = onnx.helper.make_tensor('conv15_w_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv15_w_right),
                                                    vals=conv15_w_right.flatten())
    conv15_b_right_tensor = onnx.helper.make_tensor('conv15_b_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv15_b_right),
                                                    vals=conv15_b_right.flatten())
    #layer16
    conv16_w_right_tensor = onnx.helper.make_tensor('conv16_w_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv16_w_right),
                                                    vals=conv16_w_right.flatten())
    conv16_b_right_tensor = onnx.helper.make_tensor('conv16_b_right_tensor',
                                                    data_type=onnx.TensorProto.FLOAT,
                                                    dims=np.shape(conv16_b_right),
                                                    vals=conv16_b_right.flatten())

    #make node
    conv1_right_node = onnx.helper.make_node("Conv",
                                             inputs=['right_in', 'conv1_w_right_tensor', 'conv1_b_right_tensor'],
                                             outputs=['conv1_right'],
                                             kernel_shape=[3, 3],
                                             strides=[2, 2],
                                             pads=[1, 1, 1, 1],
                                             name="conv1_right")
    conv1_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv1_right'],
                                                   outputs=['conv1_right_scale'],
                                                   scale=1.0 / 1024)
    conv1_right_relu_node = onnx.helper.make_node('Relu', ['conv1_right_scale'], ['conv1_right_relu'])

    node1_right = [conv1_right_node] + [conv1_right_scale_node] + [conv1_right_relu_node]
    initializer1_right = [conv1_w_right_tensor, conv1_b_right_tensor]

    conv2_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv1_right_relu', 'conv2_w_right_tensor', 'conv2_b_right_tensor'],
        outputs=['conv2_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv2_right")
    conv2_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv2_right'],
                                                   outputs=['conv2_right_scale'],
                                                   scale=1.0 / 1024)
    conv2_right_relu_node = onnx.helper.make_node('Relu', ['conv2_right_scale'], ['conv2_right_relu'])

    node2_right = [conv2_right_node] + [conv2_right_scale_node] + [conv2_right_relu_node]
    initializer2_right = [conv2_w_right_tensor, conv2_b_right_tensor]

    conv3_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv2_right_relu', 'conv3_w_right_tensor', 'conv3_b_right_tensor'],
        outputs=['conv3_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv3_right")
    conv3_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv3_right'],
                                                   outputs=['conv3_right_scale'],
                                                   scale=1.0 / 1024)
    conv3_right_scale_1_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv3_right_scale'],
                                                     outputs=['conv3_right_1_scale'],
                                                     scale=0.25)

    node3_right = [conv3_right_node] + [conv3_right_scale_node] + [conv3_right_scale_1_node]
    initializer3_right = [conv3_w_right_tensor, conv3_b_right_tensor]

    conv4_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv1_right_relu', 'conv4_w_right_tensor', 'conv4_b_right_tensor'],
        outputs=['conv4_right'],
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        name="conv4_right")
    conv4_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv4_right'],
                                                   outputs=['conv4_right_scale_o'],
                                                   scale=1.0 / 1024)
    conv4_right_scale_1_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv4_right_scale_o'],
                                                     outputs=['conv4_right_scale'],
                                                     scale=1.0)

    conv3_4_right_add_node = onnx.helper.make_node('Add', ['conv3_right_1_scale', 'conv4_right_scale'],
                                                   ['conv3_4_right_add'])

    conv3_4_relu_node = onnx.helper.make_node('Relu', ['conv3_4_right_add'], ['conv3_4_right_relu'])

    node4_right = [conv4_right_node] + [conv4_right_scale_node] + [conv4_right_scale_1_node] + [conv3_4_right_add_node
                                                                                               ] + [conv3_4_relu_node]
    initializer4_right = [conv4_w_right_tensor, conv4_b_right_tensor]

    conv5_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv3_4_right_relu', 'conv5_w_right_tensor', 'conv5_b_right_tensor'],
        outputs=['conv5_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv5_right")
    conv5_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv5_right'],
                                                   outputs=['conv5_right_scale'],
                                                   scale=1.0 / 1024)
    conv5_right_relu_node = onnx.helper.make_node('Relu', ['conv5_right_scale'], ['conv5_right_relu'])

    node5_right = [conv5_right_node] + [conv5_right_scale_node] + [conv5_right_relu_node]
    initializer5_right = [conv5_w_right_tensor, conv5_b_right_tensor]

    conv6_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv5_right_relu', 'conv6_w_right_tensor', 'conv6_b_right_tensor'],
        outputs=['conv6_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv6_right")
    conv6_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv6_right'],
                                                   outputs=['conv6_right_scale'],
                                                   scale=1.0 / 1024)
    conv6_right_scale_1_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv3_4_right_relu'],
                                                     outputs=['conv6_right_scale_1'],
                                                     scale=1.0)
    conv6_right_scale_2_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv6_right_scale'],
                                                     outputs=['conv6_right_scale_0'],
                                                     scale=1.0)
    conv6_right_add_node = onnx.helper.make_node('Add', ['conv6_right_scale_0', 'conv6_right_scale_1'],
                                                 ['conv6_right_add'])

    conv6_right_relu_node = onnx.helper.make_node('Relu', ['conv6_right_add'], ['conv6_right_relu'])

    node6_right = [conv6_right_node] + [conv6_right_scale_node] + [conv6_right_scale_2_node] + [
        conv6_right_scale_1_node
    ] + [conv6_right_add_node] + [conv6_right_relu_node]
    initializer6_right = [conv6_w_right_tensor, conv6_b_right_tensor]

    conv7_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv6_right_relu', 'conv7_w_right_tensor', 'conv7_b_right_tensor'],
        outputs=['conv7_right'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        name="conv7_right")
    conv7_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv7_right'],
                                                   outputs=['conv7_right_scale'],
                                                   scale=1.0 / 1024)

    conv7_right_relu_node = onnx.helper.make_node('Relu', ['conv7_right_scale'], ['conv7_right_relu'])

    node7_right = [conv7_right_node] + [conv7_right_scale_node] + [conv7_right_relu_node]
    initializer7_right = [conv7_w_right_tensor, conv7_b_right_tensor]

    conv8_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv7_right_relu', 'conv8_w_right_tensor', 'conv8_b_right_tensor'],
        outputs=['conv8_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv8_right")
    conv8_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv8_right'],
                                                   outputs=['conv8_right_scale'],
                                                   scale=1.0 / 1024)
    conv8_right_scale_1_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv8_right_scale'],
                                                     outputs=['conv8_right_scale_1'],
                                                     scale=1.0)

    # conv8_relu_node = onnx.helper.make_node('Relu', ['conv3_scale'], ['conv3_relu'])

    node8_right = [conv8_right_node] + [conv8_right_scale_node] + [conv8_right_scale_1_node]
    initializer8_right = [conv8_w_right_tensor, conv8_b_right_tensor]

    conv9_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv6_right_relu', 'conv9_w_right_tensor', 'conv9_b_right_tensor'],
        outputs=['conv9_right'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
        name="conv9_right")
    conv9_right_scale_node = onnx.helper.make_node("Scale",
                                                   inputs=['conv9_right'],
                                                   outputs=['conv9_right_scale_o'],
                                                   scale=1.0 / 1024)
    conv9_right_scale_1_node = onnx.helper.make_node("Scale",
                                                     inputs=['conv9_right_scale_o'],
                                                     outputs=['conv9_right_scale'],
                                                     scale=1.0)
    conv8_9_right_add_node = onnx.helper.make_node('Add', ['conv8_right_scale_1', 'conv9_right_scale'],
                                                   ['conv8_9_right_add'])

    conv8_9_right_relu_node = onnx.helper.make_node('Relu', ['conv8_9_right_add'], ['conv8_9_right_relu'])

    node9_right = [conv9_right_node] + [conv9_right_scale_node] + [conv9_right_scale_1_node] + [
        conv8_9_right_add_node
    ] + [conv8_9_right_relu_node]
    initializer9_right = [conv9_w_right_tensor, conv9_b_right_tensor]

    conv10_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv8_9_right_relu', 'conv10_w_right_tensor', 'conv10_b_right_tensor'],
        outputs=['conv10_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv10_right")
    conv10_right_scale_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv10_right'],
                                                    outputs=['conv10_right_scale'],
                                                    scale=1.0 / 1024)

    conv10_right_relu_node = onnx.helper.make_node('Relu', ['conv10_right_scale'], ['conv10_right_relu'])

    node10_right = [conv10_right_node] + [conv10_right_scale_node] + [conv10_right_relu_node]
    initializer10_right = [conv10_w_right_tensor, conv10_b_right_tensor]

    conv11_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv10_right_relu', 'conv11_w_right_tensor', 'conv11_b_right_tensor'],
        outputs=['conv11_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv11_right")
    conv11_right_scale_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv11_right'],
                                                    outputs=['conv11_right_scale_1'],
                                                    scale=1.0 / 1024)
    conv11_right_scale_1_node = onnx.helper.make_node("Scale",
                                                      inputs=['conv8_9_right_relu'],
                                                      outputs=['conv11_right_scale_o'],
                                                      scale=1.0)
    conv11_right_scale_2_node = onnx.helper.make_node("Scale",
                                                      inputs=['conv11_right_scale_1'],
                                                      outputs=['conv11_right_scale'],
                                                      scale=1.0)
    conv11_right_add_node = onnx.helper.make_node('Add', ['conv11_right_scale_o', 'conv11_right_scale'],
                                                  ['conv11_right_add'])

    conv11_right_relu_node = onnx.helper.make_node('Relu', ['conv11_right_add'], ['conv11_right_relu'])

    node11_right = [conv11_right_node] + [conv11_right_scale_node] + [conv11_right_scale_2_node] + [
        conv11_right_scale_1_node
    ] + [conv11_right_add_node] + [conv11_right_relu_node]
    initializer11_right = [conv11_w_right_tensor, conv11_b_right_tensor]

    conv12_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv11_right_relu', 'conv12_w_right_tensor', 'conv12_b_right_tensor'],
        outputs=['conv12_right'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        name="conv12_right")
    conv12_right_scale_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv12_right'],
                                                    outputs=['conv12_right_scale'],
                                                    scale=1.0 / 1024)
    conv12_right_relu_node = onnx.helper.make_node('Relu', ['conv12_right_scale'], ['conv12_right_relu'])

    node12_right = [conv12_right_node] + [conv12_right_scale_node] + [conv12_right_relu_node]
    initializer12_right = [conv12_w_right_tensor, conv12_b_right_tensor]

    conv13_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv12_right_relu', 'conv13_w_right_tensor', 'conv13_b_right_tensor'],
        outputs=['conv13_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv13_right")
    conv13_right_scale_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv13_right'],
                                                    outputs=['conv13_right_scale_1'],
                                                    scale=1.0 / 1024)
    conv13_right_scale_1_node = onnx.helper.make_node("Scale",
                                                      inputs=['conv13_right_scale_1'],
                                                      outputs=['conv13_right_scale'],
                                                      scale=1.0 / 1024)

    node13_right = [conv13_right_node] + [conv13_right_scale_node] + [conv13_right_scale_1_node]
    initializer13_right = [conv13_w_right_tensor, conv13_b_right_tensor]

    conv14_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv11_right_relu', 'conv14_w_right_tensor', 'conv14_b_right_tensor'],
        outputs=['conv14_right'],
        kernel_shape=[2, 1],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
        name="conv14_right")
    conv14_right_scale_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv14_right'],
                                                    outputs=['conv14_right_scale_o'],
                                                    scale=1.0 / 1024)
    conv14_right_scale_1_node = onnx.helper.make_node("Scale",
                                                      inputs=['conv14_right_scale_o'],
                                                      outputs=['conv14_right_scale'],
                                                      scale=1.0)
    conv13_14_right_add_node = onnx.helper.make_node('Add', ['conv13_right_scale', 'conv14_right_scale'],
                                                     ['conv13_14_right_add'])

    conv13_14_right_relu_node = onnx.helper.make_node('Relu', ['conv13_14_right_add'], ['conv13_14_right_relu'])

    node14_right = [conv14_right_node] + [conv14_right_scale_node] + [conv14_right_scale_1_node] + [
        conv13_14_right_add_node
    ] + [conv13_14_right_relu_node]
    initializer14_right = [conv14_w_right_tensor, conv14_b_right_tensor]

    conv15_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv13_14_right_relu', 'conv15_w_right_tensor', 'conv15_b_right_tensor'],
        outputs=['conv15_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv15_right")
    conv15_right_scale_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv15_right'],
                                                    outputs=['conv15_right_scale'],
                                                    scale=1.0 / 1024)
    conv15_right_relu_node = onnx.helper.make_node('Relu', ['conv15_right_scale'], ['conv15_right_relu'])

    node15_right = [conv15_right_node] + [conv15_right_scale_node] + [conv15_right_relu_node]
    initializer15_right = [conv15_w_right_tensor, conv15_b_right_tensor]

    conv16_right_node = onnx.helper.make_node(
        "Conv",
        inputs=['conv15_right_relu', 'conv16_w_right_tensor', 'conv16_b_right_tensor'],
        outputs=['conv16_right'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name="conv16_right")
    conv16_right_scale_node = onnx.helper.make_node("Scale",
                                                    inputs=['conv16_right'],
                                                    outputs=['conv16_right_scale_1'],
                                                    scale=1.0 / 1024)
    conv16_right_scale_1_node = onnx.helper.make_node("Scale",
                                                      inputs=['conv13_14_right_relu'],
                                                      outputs=['conv16_right_scale_o'],
                                                      scale=1.0)
    conv16_right_scale_2_node = onnx.helper.make_node("Scale",
                                                      inputs=['conv16_right_scale_1'],
                                                      outputs=['conv16_right_scale'],
                                                      scale=1.0)
    conv16_right_add_node = onnx.helper.make_node('Add', ['conv16_right_scale_o', 'conv16_right_scale'],
                                                  ['conv16_right_add'])

    conv16_right_relu_node = onnx.helper.make_node('Relu', ['conv16_right_add'], ['conv16_right_add_relu'])

    right_ave_pooling_node = onnx.helper.make_node(
        "AveragePool",
        inputs=["conv16_right_add_relu"],
        outputs=["out"],
        kernel_shape=[8, 2],
        pads=[0, 0, 0, 0],  #[top,left,bottom,right]
        strides=[1, 1],
        scale_in=0.0625,
        scale_out=0.0625)

    node16_right = [conv16_right_node] + [conv16_right_scale_node] + [conv16_right_scale_2_node] + [
        conv16_right_scale_1_node
    ] + [conv16_right_add_node] + [conv16_right_relu_node] + [right_ave_pooling_node]
    initializer16_right = [conv16_w_right_tensor, conv16_b_right_tensor]

    nodes_right = node1_right + node2_right + node3_right + node4_right + node5_right + node6_right + node7_right + node8_right \
                + node9_right + node10_right + node11_right + node12_right + node13_right + node14_right + node15_right + node16_right
    initializers_right = initializer1_right + initializer2_right + initializer3_right + initializer4_right + initializer5_right \
                   + initializer6_right + initializer7_right + initializer8_right + initializer9_right + initializer10_right \
                   + initializer11_right + initializer12_right + initializer13_right + initializer14_right + initializer15_right \
                   + initializer16_right

    input_shape = (1, 1, 64, 12)
    output_shape = (1, 96, 1, 1)

    dccrn_graph = onnx.helper.make_graph(
        nodes_right,
        "hw_new_cnn_net_right",
        inputs=[onnx.helper.make_tensor_value_info("right_in", onnx.TensorProto.FLOAT, list(input_shape))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape))],
        initializer=initializers_right)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='hw_new_cnn_net_right')

    with open(onnx_dir + 'hw_new_cnn_net_right.onnx', "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate hw_new_cnn_net_right sucessfully!")


def generate_model_gemm():

    gemm_w_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/gemm/gemm_w_left.npy').astype('float32')
    gemm_b_left = np.load('./model/pipeline/cnn_tc_resnet_model/params/gemm/gemm_b_left.npy').astype('float32')

    gemm_w_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/gemm/gemm_w_right.npy').astype('float32')
    gemm_b_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/gemm/gemm_b_right.npy').astype('float32')

    gemm_w_left_tensor = onnx.helper.make_tensor('gemm_w_left_tensor',
                                                 data_type=onnx.TensorProto.FLOAT,
                                                 dims=np.shape(gemm_w_left),
                                                 vals=gemm_w_left.flatten())
    gemm_b_left_tensor = onnx.helper.make_tensor('gemm_b_left_tensor',
                                                 data_type=onnx.TensorProto.FLOAT,
                                                 dims=np.shape(gemm_b_left),
                                                 vals=gemm_b_left.flatten())

    gemm_w_right_tensor = onnx.helper.make_tensor('gemm_w_right_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(gemm_w_right),
                                                  vals=gemm_w_right.flatten())
    gemm_b_right_tensor = onnx.helper.make_tensor('gemm_b_right_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(gemm_b_right),
                                                  vals=gemm_b_right.flatten())
    # gemm_concat_node = onnx.helper.make_node("Concat",
    #                                              inputs=["in1","in2"],
    #                                              outputs=["in_concat"],
    #                                              axis = 1)

    gemm_left_node = onnx.helper.make_node('Gemm',
                                           inputs=['in1', 'gemm_w_left_tensor', 'gemm_b_left_tensor'],
                                           outputs=['linear_left_out'],
                                           name='linear_left')
    gemm_scale_left_node = onnx.helper.make_node('Scale', ['linear_left_out'], ['linear_scale_left_1_out'],
                                                 scale=1.0 / 1024)
    gemm_scale_left_node_1 = onnx.helper.make_node('Scale', ['linear_scale_left_1_out'], ['linear_scale_left_out'],
                                                   scale=0.5)

    gemm_right_node = onnx.helper.make_node('Gemm',
                                            inputs=['in2', 'gemm_w_right_tensor', 'gemm_b_right_tensor'],
                                            outputs=['linear_right_out'],
                                            name='linear_right')

    gemm_scale_right_node = onnx.helper.make_node('Scale', ['linear_right_out'], ['linear_scale_right_out'],
                                                  scale=1.0 / 1024)

    gemm_add_node = onnx.helper.make_node('Add', ['linear_scale_left_out', 'linear_scale_right_out'], ['out'])

    nodes = [gemm_left_node] + [gemm_scale_left_node] + [gemm_scale_left_node_1] + [gemm_right_node] + [
        gemm_scale_right_node
    ] + [gemm_add_node]
    initializers = [gemm_w_left_tensor, gemm_b_left_tensor, gemm_w_right_tensor, gemm_b_right_tensor]

    input1_shape = (1, 768)
    input2_shape = (1, 768)
    output_shape = (1, 2)

    dccrn_graph = onnx.helper.make_graph(
        nodes,
        "hw_new_gemm_net",
        inputs=[
            onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, list(input1_shape)),
            onnx.helper.make_tensor_value_info("in2", onnx.TensorProto.FLOAT, list(input2_shape))
        ],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape))],
        initializer=initializers)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='hw_new_gemm_net')

    with open(onnx_dir + 'hw_new_gemm_net.onnx', "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate hw_new_gemm_net sucessfully!")


def test_huawei_cnn_new_net_case_new_add_scale():
    generate_model_left()
    generate_model_right()
    generate_model_gemm()
    net_mods = []
    net_params = []
    net_input_datas = []
    conv_model_left = onnx.load(onnx_dir + "hw_new_cnn_net_left.onnx")
    conv_model_right = onnx.load(onnx_dir + "hw_new_cnn_net_right.onnx")
    gemm_model = onnx.load(onnx_dir + "hw_new_gemm_net.onnx")

    models = []
    models.append(conv_model_left)
    models.append(conv_model_right)
    models.append(gemm_model)

    

    shape_dict_conv_left = {}
    shape_dict_conv_left['left_in'] = (1, 8, 280, 12)  #NCHW

    mod_conv_left, param_conv_left = witin_frontend.frontend.from_onnx(conv_model_left, shape_dict_conv_left)
    data_conv_left =  np.load('./model/pipeline/cnn_tc_resnet_model/params/left/input_left.npy').astype("float32")
    
    data_conv_left = data_conv_left[:11,:,:,:]
    
    data_conv_left = np.transpose(data_conv_left,(0,3,2,1)) #NWHC
    input_dt_conv_left = {}
    input_dt_conv_left['left_in'] = witin.nd.array(data_conv_left)

    net_mods.append(mod_conv_left)
    net_params.append(param_conv_left)
    net_input_datas.append(input_dt_conv_left)

    shape_dict_conv_right = {}
    shape_dict_conv_right['right_in'] = (1, 1,64, 12) #NCHW

    mod_conv_right, param_conv_right = witin_frontend.frontend.from_onnx(conv_model_right, shape_dict_conv_right)
    data_conv_right = np.load('./model/pipeline/cnn_tc_resnet_model/params/right/input_right.npy').astype("float32")
    
    data_conv_right = data_conv_right[:11,:,:,:]
    
    
    data_conv_right = np.transpose(data_conv_right,(0,3,2,1)) #NWHC
    print(data_conv_right.shape)
    input_dt_conv_right = {}
    input_dt_conv_right['right_in'] = witin.nd.array(data_conv_right)

    net_mods.append(mod_conv_right)
    net_params.append(param_conv_right)
    net_input_datas.append(input_dt_conv_right)

    shape_dict_gemm = {}
    shape_dict_gemm['in1'] = (1, 768)
    shape_dict_gemm['in2'] = (1, 768)
    mod_gemm, param_gemm = witin_frontend.frontend.from_onnx(gemm_model, shape_dict_gemm)
    data_left = np.loadtxt('./model/pipeline/cnn_tc_resnet_model/params/gemm/gemm_left_input_0116.txt').astype(
        "float32")
    data_right = np.loadtxt('./model/pipeline/cnn_tc_resnet_model/params/gemm/gemm_right_input_0116.txt').astype(
        "float32")
    data1 = data_left.reshape(-1, 8, 96)
    data1 = data1.reshape(-1, 8 * 96)
    data2 = data_right.reshape(-1, 8, 96)
    data2 = data2.reshape(-1, 8 * 96)

    

    input_dt = {}
    input_dt['in1'] = witin.nd.array(data1)
    input_dt['in2'] = witin.nd.array(data2)

    net_mods.append(mod_gemm)
    net_params.append(param_gemm)
    net_input_datas.append(input_dt)

    opt_config = ['./model/pipeline/auto_optimizer_model/auto_optimizer_tc_resnet/hua_dw_tc_opti.protobuf']
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr


    

    for manual_opt_conf in opt_config:
        onnx_models = models
        shape_dicts = [shape_dict_conv_left, shape_dict_conv_right, shape_dict_gemm]
        inputs = net_input_datas
        mods = net_mods
        params = net_params
        optimizer = Optimizer(onnx_models, shape_dicts, inputs, manual_opt_conf)

        for conf in optimizer.get_optimize_configs():
            if conf.ByteSize() > 0:
                graphs_inputs = optimizer.update_graphs_inputs(conf, inputs)
                runtime_model = npu_graph_build(mods, params, graphs_inputs, conf)
                assert runtime_model is not None
                runtime_model.run()

    # #build
    # with witin.transform.PassContext(opt_level=3):
    #     _, _, _, npu_graph = witin_frontend.build_module.build(mods,
    #                                                            target='npu',
    #                                                            target_host='npu',
    #                                                            params=params,
    #                                                            input_data=input_datas,
    #                                                            chip='BB04P1',
    #                                                            output_dir=build_dir,
    #                                                            optimize_method_config=opt_config)
    # # execute
    # from tvm.contrib import npu_graph_runtime
    # m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)

    # m.set_input(0, witin.nd.array(data_conv_left))
    # m.set_input(1, witin.nd.array(data_conv_right))
    # m.set_input(2, witin.nd.array(data1))
    # m.set_input(3, witin.nd.array(data2))

    # m.run()
    # witin_output_list = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    # print(witin_output_list[0].shape)


from onnx.shape_inference import infer_shapes
from onnx import load_model, save_model

if __name__ == "__main__":
    # conv_model = onnx_dir + "dccrn_denoise_net.onnx"
    # onnx_model = load_model(conv_model)
    # onnx_model = infer_shapes(onnx_model)
    # #onnx.checker.check_model(onnx_model)
    # print(onnx_model)
    # save_model(onnx_model, onnx_dir+"infered_test_net.onnx")
    test_huawei_cnn_new_net_case_new_add_scale()
