import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = './model/pipeline/dccrn_model/'

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
    lstm5_ifo_w = np.load(file_dir+'/params/lstm5_ifo_w.npy').astype('float32')
    lstm5_ifo_w = lstm5_ifo_w.T
    lstm5_ifo_b = np.load(file_dir+'/params/lstm5_ifo_b.npy').astype('float32')
    lstm5_c_w = np.load(file_dir+'/params/lstm5_c_w.npy').astype('float32')
    lstm5_c_w = lstm5_c_w.T
    lstm5_c_b = np.load(file_dir+'/params/lstm5_c_b.npy').astype('float32')

    #layer 6
    linear6_w = np.load(file_dir+'/params/linear6_w.npy').astype('float32')
    linear6_w = linear6_w.T
    linear6_b = np.load(file_dir+'/params/linear6_b.npy').astype('float32')

    #layer 7
    conv7_w = np.load(file_dir+'/params/deconv7_w.npy').astype('float32')
    conv7_b = np.load(file_dir+'/params/deconv7_b.npy').astype('float32')

    #layer 8
    conv8_w = np.load(file_dir+'/params/deconv8_w.npy').astype('float32')
    conv8_b = np.load(file_dir+'/params/deconv8_b.npy').astype('float32')

    #layer 9
    conv9_w = np.load(file_dir+'/params/deconv9_w.npy').astype('float32')
    conv9_b = np.load(file_dir+'/params/deconv9_b.npy').astype('float32')

    #layer 10
    conv10_w = np.load(file_dir+'/params/deconv10_w.npy').astype('float32')
    conv10_b = np.load(file_dir+'/params/deconv10_b.npy').astype('float32')

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
    #layer5
    lstm5_w_ifo_tensor = onnx.helper.make_tensor('lstm5_w_ifo_tensor',
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=np.shape(lstm5_ifo_w),
                                                vals=lstm5_ifo_w.flatten())
    lstm5_b_ifo_tensor = onnx.helper.make_tensor('lstm5_b_ifo_tensor',
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=np.shape(lstm5_ifo_b),
                                                vals=lstm5_ifo_b.flatten())
    lstm5_w_c_tensor = onnx.helper.make_tensor('lstm5_w_c_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(lstm5_c_w),
                                              vals=lstm5_c_w.flatten())
    lstm5_b_c_tensor = onnx.helper.make_tensor('lstm5_b_c_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(lstm5_c_b),
                                              vals=lstm5_c_b.flatten())

    #layer6
    linear6_w_tensor = onnx.helper.make_tensor('linear6_w_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear6_w),
                                              vals=linear6_w.flatten())
    linear6_b_tensor = onnx.helper.make_tensor('linear6_b_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(linear6_b),
                                              vals=linear6_b.flatten())

    # layer7
    # print("good",conv7_w.shape)   #(224, 112, 3, 2)
    conv7_w_tensor = onnx.helper.make_tensor('deconv7_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv7_w),
                                             vals=conv7_w.flatten())
    conv7_b_tensor = onnx.helper.make_tensor('deconv7_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv7_b),
                                             vals=conv7_b.flatten())
    # layer8
    conv8_w_tensor = onnx.helper.make_tensor('deconv8_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv8_w),
                                             vals=conv8_w.flatten())
    conv8_b_tensor = onnx.helper.make_tensor('deconv8_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv8_b),
                                             vals=conv8_b.flatten())
    # layer9
    conv9_w_tensor = onnx.helper.make_tensor('deconv9_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv9_w),
                                             vals=conv9_w.flatten())
    conv9_b_tensor = onnx.helper.make_tensor('deconv9_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv9_b),
                                             vals=conv9_b.flatten())
    # layer10
    conv10_w_tensor = onnx.helper.make_tensor('deconv10_w_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(conv10_w),
                                              vals=conv10_w.flatten())
    conv10_b_tensor = onnx.helper.make_tensor('deconv10_b_tensor',
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(conv10_b),
                                              vals=conv10_b.flatten())
    #make node
    conv1_node = onnx.helper.make_node("Conv",
                                  inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                  outputs=['conv1'],
                                  kernel_shape=[5, 2],
                                  strides=[4, 1],
                                  pads=[1, 1, 1, 0],
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
                                  pads=[1, 1, 1, 0],
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
                                  pads=[1, 1, 1, 0],
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
                                  pads=[1, 1, 1, 0],
                                  name="conv4")
    conv4_scale_node = onnx.helper.make_node("Scale", inputs=['conv4'], outputs=['conv4_scale'], scale=1.0 / 1024)
    conv4_relu_node = onnx.helper.make_node('Relu', ['conv4_scale'], ['conv4_relu'])

    conv4_reshape_tensor = onnx.helper.make_tensor("conv4_reshape",
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=(2,),
                                                   vals=[301,112*4])
    conv4_reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["conv4_relu", "conv4_reshape"],
        outputs=["conv4_reshape_out"]
    )

    node4 = [conv4_node] + [conv4_scale_node] + [conv4_relu_node] + [conv4_reshape_node]
    initializer4 = [conv4_w_tensor, conv4_b_tensor, conv4_reshape_tensor]

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

    lstm5_node = onnx.helper.make_node(
        'Lstm',
        inputs=['conv4_reshape_out', 'lstm5_w_ifo_tensor', 'lstm5_w_c_tensor', 'lstm5_b_ifo_tensor', 'lstm5_b_c_tensor'],
        scale_ioft=1024*2,
        scale_ct=1024,
        activate_type=['sigmoid', 'tanh', 'tanh'],
        activate_table=[act_table1, act_table2, act_table3],
        shift_bits=[-7, -7],
        outputs=['lstm5_out'],
        name="lstm5")

    node5 = [lstm5_node]
    initializer5 = [lstm5_w_ifo_tensor, lstm5_b_ifo_tensor, lstm5_w_c_tensor, lstm5_b_c_tensor]

    linear6_node = onnx.helper.make_node('Gemm',
                                         inputs=['lstm5_out', 'linear6_w_tensor', 'linear6_b_tensor'],
                                         outputs=['linear6_out'],
                                         name='linear6')

    linear6_scale = onnx.helper.make_node('Scale', ['linear6_out'], ['linear6_scale'], scale=1.0 / 1024)

    linear6_reshape_tensor = onnx.helper.make_tensor("linear6_reshape",
                                                   data_type=onnx.TensorProto.FLOAT,
                                                   dims=(4,),
                                                   vals=[1,112,4,301])
    linear6_reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["linear6_scale", "linear6_reshape"],
        outputs=["linear6_reshape_out"]
    )

    node6 = [linear6_node] + [linear6_scale] + [linear6_reshape_node]
    initializer6 = [linear6_w_tensor, linear6_b_tensor, linear6_reshape_tensor]

    deconv7_node = onnx.helper.make_node("ConvTranspose",
                                    inputs=['linear6_reshape_out', 'deconv7_w_tensor', 'deconv7_b_tensor'],
                                    outputs=['deconv7'],
                                    kernel_shape=[3, 2],
                                    strides=[2, 1],
                                    pads=[1, 1, 1, 0],
                                    output_padding=[1, 0, 1, 0],
                                    name="deconv7")
    deconv7_scale_node = onnx.helper.make_node("Scale", inputs=['deconv7'], outputs=['deconv7_scale'], scale=1.0 / 1024)
    deconv7_relu_node = onnx.helper.make_node('Relu', ['deconv7_scale'], ['deconv7_relu'])
    node7 = [deconv7_node] + [deconv7_scale_node] + [deconv7_relu_node]
    initializer7 = [conv7_w_tensor, conv7_b_tensor]

    deconv8_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['deconv7_relu', 'deconv8_w_tensor', 'deconv8_b_tensor'],
                                         outputs=['deconv8'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 1, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv8")
    deconv8_scale_node = onnx.helper.make_node("Scale", inputs=['deconv8'], outputs=['deconv8_scale'], scale=1.0 / 1024)
    deconv8_relu_node = onnx.helper.make_node('Relu', ['deconv8_scale'], ['deconv8_relu'])
    node8 = [deconv8_node] + [deconv8_scale_node] + [deconv8_relu_node]
    initializer8 = [conv8_w_tensor, conv8_b_tensor]

    deconv9_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['deconv8_relu', 'deconv9_w_tensor', 'deconv9_b_tensor'],
                                         outputs=['deconv9'],
                                         kernel_shape=[3, 2],
                                         strides=[2, 1],
                                         pads=[1, 1, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv9")
    deconv9_scale_node = onnx.helper.make_node("Scale", inputs=['deconv9'], outputs=['deconv9_scale'], scale=1.0 / 1024)
    deconv9_relu_node = onnx.helper.make_node('Relu', ['deconv9_scale'], ['deconv9_relu'])
    node9 =[deconv9_node] + [deconv9_scale_node] + [deconv9_relu_node]
    initializer9 = [conv9_w_tensor, conv9_b_tensor]

    deconv10_node = onnx.helper.make_node("ConvTranspose",
                                         inputs=['deconv9_relu', 'deconv10_w_tensor', 'deconv10_b_tensor'],
                                         outputs=['deconv10'],
                                         kernel_shape=[5, 2],
                                         strides=[4, 1],
                                         pads=[1, 1, 1, 0],
                                         output_padding=[1, 0, 1, 0],
                                         name="deconv10")
    deconv10_scale_node = onnx.helper.make_node("Scale", inputs=['deconv10'], outputs=['out'], scale=1.0 / 1024)

    node10 = [deconv10_node] + [deconv10_scale_node]
    initializer10 = [conv10_w_tensor, conv10_b_tensor]

    nodes = node1 + node2 + node3 + node4 + node5 + node6 + node7 + node8 + node9 + node10
    initializers = initializer1 + initializer2 + initializer3 + initializer4 + initializer5 \
                   + initializer6 + initializer7 + initializer8 + initializer9 + initializer10

    input_shape = (1, 2, 128, 301)
    output_shape = (1, 2, 128, 301)

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
    conv_model = onnx.load(file_dir + "dccrn_denoise_net.onnx")
    
    data = np.load(file_dir +'/layer_debug/input1.npy').astype('float32')
    data = data.astype('float32')  # [1, 2, 128, 301]
    
    data = np.transpose(data, (0, 3, 2, 1)) # [1, 301, 128, 4]
    data = data.reshape(data.shape[0] * data.shape[1], 1, data.shape[2], data.shape[3]) # [301, 1, 128, 4]
    print(data.shape)
    # merge the adjacent 2 frame data, [301, 1, 128, 4] -> [300, 2, 128, 4]
    new_data = np.zeros([data.shape[0] - 1, data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype) 
    for n in range(data.shape[0] - 1):
      new_data[n, 0, :, :] = data[n, :, :, :]
      new_data[n, 1, :, :] = data[n + 1, :, :, :]
    print(new_data.shape)
    
    shape_dict_conv = {}
    shape_dict_conv['in'] = (1, 2, 128, 301)  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(conv_model, shape_dict_conv)
    
    input_dt = {}
    input_dt['in'] = witin.nd.array(new_data)
    opt_config_files = [file_dir + 'dccrn_opti_0713_new.protobuf']
    for opt_config in opt_config_files:
      #output file
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

def test_dccrn_opti():
    # generate_model()
    run_model()

if __name__ == "__main__":
    test_dccrn_opti()
