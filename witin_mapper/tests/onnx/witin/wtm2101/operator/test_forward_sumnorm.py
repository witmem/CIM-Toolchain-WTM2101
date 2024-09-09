import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = "./model/"

def generate_model():
    G_scale = [1024, 2048]
    input_shape = [
        [1, 8, 64, 64]
    ]
    output_shape = [
        [1, 8, 28, 28]
    ]
    kernel_tensor_shape = [
        [32, 8, 5, 5],
        [8, 32, 3, 3]
    ]
    bias_tensor_shape = [
        [32],
        [8]
    ]
    kernel_shape = [
        [5, 5],
        [3, 3]
    ]
    stride = [
        [2, 2],
        [1, 1]
    ]
    pad = [
        [2, 2, 2, 2],
        [0, 0, 0, 0]
    ]
    dilation = [
        [2, 2],
        [1, 1]
    ]
    # layer 1
    conv1_w = np.random.randint(-127, 127,size=kernel_tensor_shape[0], dtype='int8').astype('float32')
    conv1_b = np.random.randint(-127, 127, size=bias_tensor_shape[0], dtype='int8').astype('float32')
    conv1_b = conv1_b * 128
    conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_w),
                                             vals=conv1_w.flatten())
    conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_b),
                                             vals=conv1_b.flatten())
    # layer 2
    conv2_w = np.random.randint(-127, 127,size=kernel_tensor_shape[1], dtype='int8').astype('float32')
    conv2_b = np.random.randint(-127, 127, size=bias_tensor_shape[1], dtype='int8').astype('float32')
    conv2_b = conv2_b * 128
    conv2_w_tensor = onnx.helper.make_tensor('conv2_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_w),
                                             vals=conv2_w.flatten())
    conv2_b_tensor = onnx.helper.make_tensor('conv2_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_b),
                                             vals=conv2_b.flatten())		
    						 
	# make conv1 node
    conv1_node = onnx.helper.make_node("Conv",
                                       inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                       outputs=['conv1'],
                                       kernel_shape=kernel_shape[0],
                                       strides=stride[0],
                                       pads=pad[0],
                                       dilations=dilation[0],
                                       name="conv1")
    conv1_scale_node = onnx.helper.make_node("Scale", inputs=['conv1'], outputs=['conv1_scale'], scale=1.0 / G_scale[0])
    conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['conv1_relu_out'])
    
    conv1_nodes = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
    initializer1 = [conv1_w_tensor, conv1_b_tensor]
    
    sumnorm_node = onnx.helper.make_node(
          "SumNorm",
          inputs=["conv1_relu_out"],
          outputs=["sumnorm_out"],
          scale_in=2.0,
          scale_out=1.0/255.0,
          axis=2)
    
    # make conv2 node
    conv2_node = onnx.helper.make_node("Conv",
                                       inputs=['sumnorm_out', 'conv2_w_tensor', 'conv2_b_tensor'],
                                       outputs=['conv2'],
                                       kernel_shape=kernel_shape[1],
                                       strides=stride[1],
                                       pads=pad[1],
                                       dilations=dilation[1],
                                       name="conv2")
    conv2_scale_node = onnx.helper.make_node("Scale", inputs=['conv2'], outputs=['conv2_scale'], scale=1.0 / G_scale[1])
    conv2_relu_node = onnx.helper.make_node('Relu', ['conv2_scale'], ['out'])
    conv2_nodes = [conv2_node] + [conv2_scale_node] + [conv2_relu_node]
    initializer2 = [conv2_w_tensor, conv2_b_tensor]
    
    nodes = conv1_nodes + [sumnorm_node] + conv2_nodes
    initializers = initializer1 + initializer2
	
    dccrn_graph = onnx.helper.make_graph(
        nodes,
        "softmax_net",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input_shape[0]))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape[0]))],
        initializer=initializers)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='softmax_net')
    
    with open(file_dir + "sumnorm.onnx", "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate sumnorm.onnx sucessfully!")
	

def run_model():
    input_shape = [1, 8, 64, 64]
    params = []
    conv_model = onnx.load(file_dir + "sumnorm.onnx")
    data = np.random.randint(0, 127, size=[11,
                                           input_shape[3],
                                           input_shape[2],
                                           input_shape[1]], dtype='uint8').astype('float32')
    data = data.astype('float32')  # NCHW
    shape_dict_conv = {}
    shape_dict_conv['in'] = input_shape # NCHW
    mod, params = witin_frontend.frontend.from_onnx(conv_model, shape_dict_conv)

    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    # build
    chip_type = "BB04P1"
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=params,
                                                               input_data=input_dt,
                                                               chip=chip_type,
                                                               output_dir=build_dir,
                                                               optimize_method_config="")

    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, chip_type, output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)

def test_softmax():
   generate_model()
   run_model()

if __name__ == "__main__":
   test_softmax()	
