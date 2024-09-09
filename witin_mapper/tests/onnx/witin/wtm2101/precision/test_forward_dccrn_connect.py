import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = './model/pipeline/dccrn_connect_model/'

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
    #layer 1
    conv1_w = np.load(file_dir+'/params/conv1_w.npy').astype('float32')
    conv1_b = np.load(file_dir+'/params/conv1_b.npy').astype('float32')

    #layer 2
    conv2_w = np.load(file_dir+'/params/conv2_w.npy').astype('float32')
    conv2_b = np.load(file_dir+'/params/conv2_b.npy').astype('float32')

    #layer 3
    conv3_w = np.load(file_dir+'/params/conv3_w.npy').astype('float32')
    conv3_b = np.load(file_dir+'/params/conv3_b.npy').astype('float32')

    #layer 4
    conv4_w = np.load(file_dir+'/params/conv4_w.npy').astype('float32')
    conv4_b = np.load(file_dir+'/params/conv4_b.npy').astype('float32')
    #layer 5
    conv5_w = np.load(file_dir+'/params/conv5_w.npy').astype('float32')
    conv5_b = np.load(file_dir+'/params/conv5_b.npy').astype('float32')

    #layer 6
    lstm6_ifo_w = np.load(file_dir+'/params/lstm6_ifo_w.npy').astype('float32')
    lstm6_ifo_w = lstm6_ifo_w.T
    lstm6_ifo_b = np.load(file_dir+'/params/lstm6_ifo_b.npy').astype('float32')
    lstm6_c_w = np.load(file_dir+'/params/lstm6_c_w.npy').astype('float32')
    lstm6_c_w = lstm6_c_w.T
    lstm6_c_b = np.load(file_dir+'/params/lstm6_c_b.npy').astype('float32')

    #layer 7
    linear7_w = np.load(file_dir+'/params/linear7_w.npy').astype('float32')
    linear7_w = linear7_w.T
    linear7_b = np.load(file_dir+'/params/linear7_b.npy').astype('float32')

    #layer 8
    conv8_w = np.load(file_dir+'/params/deconv8_w.npy').astype('float32')
    conv8_b = np.load(file_dir+'/params/deconv8_b.npy').astype('float32')

    #layer 9
    conv9_w = np.load(file_dir+'/params/deconv9_w.npy').astype('float32')
    conv9_b = np.load(file_dir+'/params/deconv9_b.npy').astype('float32')

    #layer 10

    conv10_w = np.load(file_dir+'/params/deconv10_w.npy').astype('float32')
    conv10_b = np.load(file_dir+'/params/deconv10_b.npy').astype('float32')
    #layer 11
    conv11_w = np.load(file_dir+'/params/deconv11_w.npy').astype('float32')
    conv11_b = np.load(file_dir+'/params/deconv11_b.npy').astype('float32')
    #layer 12

    conv12_w = np.load(file_dir+'/params/deconv12_w.npy').astype('float32')
    conv12_b = np.load(file_dir+'/params/deconv12_b.npy').astype('float32')
    #onnx.helper.make_tensor
    #layer1
    conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_w),
                                             vals=conv1_w.flatten())
    conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_b),
                                             vals=conv1_b.flatten())
    # layer2
    conv2_w_tensor = onnx.helper.make_tensor('conv2_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_w),
                                             vals=conv2_w.flatten())
    conv2_b_tensor = onnx.helper.make_tensor('conv2_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_b),
                                             vals=conv2_b.flatten())
    # layer3
    conv3_w_tensor = onnx.helper.make_tensor('conv3_w_tensor',
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=np.shape(conv3_w),
                                            vals=conv3_w.flatten())
    conv3_b_tensor = onnx.helper.make_tensor('conv3_b_tensor',
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=np.shape(conv3_b),
                                            vals=conv3_b.flatten())
    #layer4
    conv4_w_tensor = onnx.helper.make_tensor('conv4_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv4_w),
                                             vals=conv4_w.flatten())
    conv4_b_tensor = onnx.helper.make_tensor('conv4_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv4_b),
                                             vals=conv4_b.flatten())
    conv5_w_tensor = onnx.helper.make_tensor('conv5_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv5_w),
                                             vals=conv5_w.flatten())
    conv5_b_tensor = onnx.helper.make_tensor('conv5_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv5_b),
                                             vals=conv5_b.flatten())
    #layer5
    lstm6_w_ifo_tensor = onnx.helper.make_tensor('lstm6_w_ifo_tensor',
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=np.shape(lstm6_ifo_w),
                                                vals=lstm6_ifo_w.flatten())
    lstm6_b_ifo_tensor = onnx.helper.make_tensor('lstm6_b_ifo_tensor',
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=np.shape(lstm6_ifo_b),
                                                vals=lstm6_ifo_b.flatten())
    lstm6_w_c_tensor = onnx.helper.make_tensor('lstm6_w_c_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(lstm6_c_w),
                                              vals=lstm6_c_w.flatten())
    lstm6_b_c_tensor = onnx.helper.make_tensor('lstm6_b_c_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(lstm6_c_b),
                                              vals=lstm6_c_b.flatten())

    #layer6
    linear7_w_tensor = onnx.helper.make_tensor('linear7_w_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear7_w),
                                              vals=linear7_w.flatten())
    linear7_b_tensor = onnx.helper.make_tensor('linear7_b_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear7_b),
                                              vals=linear7_b.flatten())

    # layer7
    # print("good",conv7_w.shape)   #(224, 112, 3, 2)
    conv8_w_tensor = onnx.helper.make_tensor('deconv8_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv8_w),
                                             vals=conv8_w.flatten())
    conv8_b_tensor = onnx.helper.make_tensor('deconv8_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv8_b),
                                             vals=conv8_b.flatten())

    conv8_copy_w_tensor = onnx.helper.make_tensor('deconv8_copy_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv8_w),
                                             vals=conv8_w.flatten())
    conv8_copy_b_tensor = onnx.helper.make_tensor('deconv8_copy_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv8_b),
                                             vals=conv8_b.flatten())
    # layer8
    conv9_w_tensor = onnx.helper.make_tensor('deconv9_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv9_w),
                                             vals=conv9_w.flatten())
    conv9_b_tensor = onnx.helper.make_tensor('deconv9_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv9_b),
                                             vals=conv9_b.flatten())
    # layer9
    conv10_w_tensor = onnx.helper.make_tensor('deconv10_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv10_w),
                                             vals=conv10_w.flatten())
    conv10_b_tensor = onnx.helper.make_tensor('deconv10_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv10_b),
                                             vals=conv10_b.flatten())
    # layer10
    conv11_w_tensor = onnx.helper.make_tensor('deconv11_w_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(conv11_w),
                                              vals=conv11_w.flatten())
    conv11_b_tensor = onnx.helper.make_tensor('deconv11_b_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(conv11_b),
                                              vals=conv11_b.flatten())

    # layer10
    conv12_w_tensor = onnx.helper.make_tensor('deconv12_w_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(conv12_w),
                                              vals=conv12_w.flatten())
    conv12_b_tensor = onnx.helper.make_tensor('deconv12_b_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(conv12_b),
                                              vals=conv12_b.flatten())
    #make node
    conv1_node = onnx.helper.make_node("Conv",
                                  inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                  outputs=['conv1'],
                                  kernel_shape=[3, 2],
                                  strides=[2, 1],
                                  pads=[1, 1, 0, 0],
                                  name="conv1")
    conv1_scale_node = onnx.helper.make_node("Scale", inputs=['conv1'], outputs=['conv1_scale'], scale=1.0 / 1024)
    conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['conv1_relu'])

    node1 = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
    initializer1 = [conv1_w_tensor, conv1_b_tensor]

    conv2_node = onnx.helper.make_node("Conv",
                                  inputs=['conv1_relu', 'conv2_w_tensor', 'conv2_b_tensor'],
                                  outputs=['conv2'],
                                  kernel_shape=[3, 2],
                                  strides=[2, 1],
                                  pads=[1, 1, 0, 0],
                                  name="conv2")
    conv2_scale_node = onnx.helper.make_node("Scale", inputs=['conv2'], outputs=['conv2_scale'], scale=1.0 / 1024)
    conv2_relu_node = onnx.helper.make_node('Relu', ['conv2_scale'], ['conv2_relu'])

    node2 = [conv2_node] + [conv2_scale_node] + [conv2_relu_node]
    initializer2 = [conv2_w_tensor, conv2_b_tensor]

    conv3_node = onnx.helper.make_node("Conv",
                                  inputs=['conv2_relu', 'conv3_w_tensor', 'conv3_b_tensor'],
                                  outputs=['conv3'],
                                  kernel_shape=[3, 2],
                                  strides=[2, 1],
                                  pads=[1, 1, 0, 0],
                                  name="conv3")
    conv3_scale_node = onnx.helper.make_node("Scale", inputs=['conv3'], outputs=['conv3_scale'], scale=1.0 / 1024)
    conv3_relu_node = onnx.helper.make_node('Relu', ['conv3_scale'], ['conv3_relu'])

    node3 = [conv3_node] + [conv3_scale_node] + [conv3_relu_node]
    initializer3 = [conv3_w_tensor, conv3_b_tensor]

    conv4_node = onnx.helper.make_node("Conv",
                                  inputs=['conv3_relu', 'conv4_w_tensor', 'conv4_b_tensor'],
                                  outputs=['conv4'],
                                  kernel_shape=[3, 2],
                                  strides=[2, 1],
                                  pads=[1, 1, 0, 0],
                                  name="conv4")
    conv4_scale_node = onnx.helper.make_node("Scale", inputs=['conv4'], outputs=['conv4_scale'], scale=1.0 / 1024)
    conv4_relu_node = onnx.helper.make_node('Relu', ['conv4_scale'], ['conv4_relu'])

    node4 = [conv4_node] + [conv4_scale_node] + [conv4_relu_node]
    initializer4 = [conv4_w_tensor, conv4_b_tensor]

    conv5_node = onnx.helper.make_node("Conv",
                                  inputs=['conv4_relu', 'conv5_w_tensor', 'conv5_b_tensor'],
                                  outputs=['conv5'],
                                  kernel_shape=[3, 2],
                                  strides=[2, 1],
                                  pads=[1, 1, 0, 0],
                                  name="conv5")
    conv5_scale_node = onnx.helper.make_node("Scale", inputs=['conv5'], outputs=['conv5_scale'], scale=1.0 / 1024)
    conv5_relu_node = onnx.helper.make_node('Relu', ['conv5_scale'], ['conv5_relu'])

    conv5_reshape_tensor = onnx.helper.make_tensor("conv5_reshape",
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=(2,),
                                                   vals=[501,112*4])
    conv5_reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["conv5_relu", "conv5_reshape"],
        outputs=["conv5_reshape_out"]
    )

    node5 = [conv5_node] + [conv5_scale_node] + [conv5_relu_node] + [conv5_reshape_node]
    initializer5 = [conv5_w_tensor, conv5_b_tensor, conv5_reshape_tensor]

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

    lstm6_node = onnx.helper.make_node(
        'Lstm',
        inputs=['conv5_reshape_out', 'lstm6_w_ifo_tensor', 'lstm6_w_c_tensor', 'lstm6_b_ifo_tensor', 'lstm6_b_c_tensor'],
        scale_ioft=1024*2,
        scale_ct=1024,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        shift_bits=[-7, -7],
        outputs=['lstm6_out'],
        name="lstm6")

    node6 = [lstm6_node]
    initializer6 = [lstm6_w_ifo_tensor, lstm6_b_ifo_tensor, lstm6_w_c_tensor, lstm6_b_c_tensor]

    linear7_node = onnx.helper.make_node('Gemm',
                                         inputs=['lstm6_out', 'linear7_w_tensor', 'linear7_b_tensor'],
                                         outputs=['linear7_out'],
                                         name='linear7')

    linear7_scale = onnx.helper.make_node('Scale', ['linear7_out'], ['linear7_scale'], scale=1.0 / 1024)
    # linear6_relu_node = onnx.helper.make_node('Relu', ['linear6_scale'], ['linear6_relu'])

    linear7_reshape_tensor = onnx.helper.make_tensor("linear7_reshape",
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=(4,),
                                                   vals=[1,112,4,501])
    linear7_reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["linear7_scale", "linear7_reshape"],
        outputs=["linear7_reshape_out"]
    )

    node7 = [linear7_node] + [linear7_scale] + [linear7_reshape_node]
    initializer7 = [linear7_w_tensor, linear7_b_tensor, linear7_reshape_tensor]

    # deconv7_concat_node = onnx.helper.make_node("Concat",
    #                                              inputs=["linear6_reshape_out","conv4_relu"],
    #                                              outputs=["deconv7_concat"],
    #                                              axis = 1)



    deconv8_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['linear7_reshape_out', 'deconv8_w_tensor', 'deconv8_b_tensor'],
                                         outputs=['deconv8'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 0, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv8")
    deconv8_scale_node = onnx.helper.make_node("Scale", inputs=['deconv8'], outputs=['deconv8_scale'], scale=1.0 / 1024)
    deconv8_relu_node = onnx.helper.make_node('Relu', ['deconv8_scale'], ['deconv8_relu'])
    node8 = [deconv8_node] + [deconv8_scale_node] + [deconv8_relu_node]
    initializer8 = [conv8_w_tensor, conv8_b_tensor]

    deconv8_copy_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['linear7_reshape_out', 'deconv8_copy_w_tensor', 'deconv8_copy_b_tensor'],
                                         outputs=['deconv8_copy'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 0, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv8_copy")
    deconv8_copy_scale_node = onnx.helper.make_node("Scale", inputs=['deconv8_copy'], outputs=['deconv8_copy_scale'], scale=1.0 / 1024)
    deconv8_copy_relu_node = onnx.helper.make_node('Relu', ['deconv8_copy_scale'], ['deconv8_copy_relu'])

    deconv8_add_node = onnx.helper.make_node(
        'Add', ['deconv8_relu', 'deconv8_copy_relu'], ['deconv8_add'],
        shift_bit=-1)
    # deconv8_shift_node = onnx.helper.make_node('Scale', ['deconv8_add'], ['deconv8_shift'], scale = 0.5)

    node8_copy = [deconv8_copy_node] + [deconv8_copy_scale_node] + [
        deconv8_copy_relu_node
    ] + [deconv8_add_node]
    initializer8_copy = [conv8_copy_w_tensor, conv8_copy_b_tensor]

    # deconv9_concat_node = onnx.helper.make_node("Concat",
    #                                             inputs=["deconv8_relu", "conv2_relu"],
    #                                             outputs=["deconv9_concat"],
    #                                             axis=1)

    deconv9_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['deconv8_add', 'deconv9_w_tensor', 'deconv9_b_tensor'],
                                         outputs=['deconv9'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 0, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv9")
    deconv9_scale_node = onnx.helper.make_node("Scale", inputs=['deconv9'], outputs=['deconv9_scale'], scale=1.0 / 1024)
    deconv9_relu_node = onnx.helper.make_node('Relu', ['deconv9_scale'], ['deconv9_relu'])
    node9 =[deconv9_node] + [deconv9_scale_node] + [deconv9_relu_node]
    initializer9 = [conv9_w_tensor, conv9_b_tensor]

    # deconv10_concat_node = onnx.helper.make_node("Concat",
    #                                             inputs=["deconv9_relu", "conv1_relu"],
    #                                             outputs=["deconv10_concat"],
    #                                             axis=1)

    deconv10_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['deconv9_relu', 'deconv10_w_tensor', 'deconv10_b_tensor'],
                                         outputs=['deconv10'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 0, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv10")
    deconv10_scale_node = onnx.helper.make_node("Scale", inputs=['deconv10'], outputs=['deconv10_scale'], scale=1.0 / 1024)
    deconv10_relu_node = onnx.helper.make_node('Relu', ['deconv10_scale'], ['deconv10_relu'])

    node10 = [deconv10_node] + [deconv10_scale_node] + [deconv10_relu_node]
    initializer10 = [conv10_w_tensor, conv10_b_tensor]

    deconv11_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['deconv10_relu', 'deconv11_w_tensor', 'deconv11_b_tensor'],
                                         outputs=['deconv11'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 0, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv11")
    deconv11_scale_node = onnx.helper.make_node("Scale", inputs=['deconv11'], outputs=['deconv11_scale'], scale=1.0 / 1024)
    deconv11_relu_node = onnx.helper.make_node('Relu', ['deconv11_scale'], ['deconv11_relu'])
    node11 =[deconv11_node] + [deconv11_scale_node] + [deconv11_relu_node]
    initializer11 = [conv11_w_tensor, conv11_b_tensor]

    deconv12_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['deconv11_relu', 'deconv12_w_tensor', 'deconv12_b_tensor'],
                                         outputs=['deconv12'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 0, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv12")
    deconv12_scale_node = onnx.helper.make_node("Scale", inputs=['deconv12'], outputs=['out'], scale=1.0 / 1024)

    node12 =[deconv12_node] + [deconv12_scale_node]
    initializer12 = [conv12_w_tensor, conv12_b_tensor]



    nodes = node1 + node2 + node3 + node4 + node5 + node6 + node7 + node8 + node8_copy + node9 + node10 + node11 + node12
    initializers = initializer1 + initializer2 + initializer3 + initializer4 + initializer5 \
                   + initializer6 + initializer7 + initializer8 + initializer8_copy + initializer9 + initializer10 \
                    + initializer11 + initializer12

    input_shape = (1, 2, 128, 501)
    output_shape = (1, 2, 128, 501)

    dccrn_graph = onnx.helper.make_graph(
        nodes,
        "dccrn_net",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input_shape))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape))],
        initializer=initializers)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='dccrn_denoise_net')

    with open(file_dir+'dccrn_denoise_net.onnx', "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate dccrn_denoise_net sucessfully!")

def run_model():
    params = []
    conv_model = onnx.load(file_dir + "/dccrn_denoise_net.onnx")

    data = np.load(file_dir+'/layer_debug/conv1_input.npy').astype('float32')
    data = data.astype('float32')
    
    data = np.transpose(data,(0,3,2,1))
    data = data.reshape(data.shape[0]*data.shape[1],1, data.shape[2], data.shape[3])

    # merge the adjacent 2 frame data, [501, 1, 128, 4] -> [500, 2, 128, 4]
    new_data = np.zeros([data.shape[0] - 1, data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype)
    for n in range(data.shape[0] - 1):
        new_data[n, 0, :, :] = data[n, :, :, :]
        new_data[n, 1, :, :] = data[n + 1, :, :, :]

    shape_dict_conv = {}
    shape_dict_conv['in'] = (1, 2, 128, 501)  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(conv_model, shape_dict_conv)

    input_dt = {}
    input_dt['in'] = witin.nd.array(new_data)
    opt_config = file_dir + 'dccrn_connect_opti.protobuf'
    
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    

    #build
    with witin.transform.PassContext(opt_level=3):
        _,_,_,npu_graph = witin_frontend.build_module.build(mod,
                                                            target='npu',
                                                            target_host='npu',
                                                            params=params,
                                                            input_data=input_dt,
                                                            chip='BB04P1',
                                                            output_dir=build_dir,
                                                            optimize_method_config=opt_config)

    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(new_data))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)

def test_dccrn_connect_opti():
    generate_model()
    run_model()

if __name__ == "__main__":
   test_dccrn_connect_opti()
