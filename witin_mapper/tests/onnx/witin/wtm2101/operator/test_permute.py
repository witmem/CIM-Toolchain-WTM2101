import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = "./model/"

def generate_model():
    G_scale = [1024]
    input1_shape = [1, 8, 64, 64]
    # output1_shape = [1, 32, 30, 30]
    kernel1_tensor_shape = [32, 8, 5, 5]
    bias1_tensor_shape = [32]
    kernel1_shape = [5, 5]
    stride1 = [2, 2]
    pad1 = [2, 2, 2, 2]
    dilation1 = [2, 2]
    
    output2_shape = [1, 3, 15, 16]
    kernel2_tensor_shape = [3, 30, 2, 2]
    bias2_tensor_shape = [3]
    kernel2_shape = [2, 2]
    stride2 = [2, 2]
    pad2 = [0, 0, 0, 0]
    dilation2 = [1, 1]
   
    # conv2d layer    
    conv1_w = np.random.randint(-127, 127,size=kernel1_tensor_shape, dtype='int8').astype('float32')
    conv1_b = np.random.randint(-127, 127, size=bias1_tensor_shape, dtype='int8').astype('float32') * 128
    conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor', data_type=onnx.TensorProto.FLOAT, dims=np.shape(conv1_w), vals=conv1_w.flatten())
    conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor', data_type=onnx.TensorProto.FLOAT, dims=np.shape(conv1_b), vals=conv1_b.flatten())
    conv1_node = onnx.helper.make_node("Conv",
                                       inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                       outputs=['conv1'],
                                       kernel_shape=kernel1_shape,
                                       strides=stride1,
                                       pads=pad1,
                                       dilations=dilation1,
                                       name="conv1")
    conv1_scale_node = onnx.helper.make_node("Scale", inputs=['conv1'], outputs=['conv1_scale'], scale=1.0 / G_scale[0])
    conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['conv1_relu_out'])
    node1 = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
    initializer1 = [conv1_w_tensor, conv1_b_tensor]
    
    # permute layer
    permute_node = onnx.helper.make_node("Permute", inputs=["conv1_relu_out"], dims=[0, 3, 2, 1], outputs=["permute_out"])
    
    # conv2d layer
    conv2_w = np.random.randint(-127, 127, size=kernel2_tensor_shape, dtype='int8').astype('float32')
    conv2_b = np.random.randint(-127, 127, size=bias2_tensor_shape, dtype='int8').astype('float32') * 128
    conv2_w_tensor = onnx.helper.make_tensor('conv2_w_tensor', data_type=onnx.TensorProto.FLOAT, dims=np.shape(conv2_w), vals=conv2_w.flatten())
    conv2_b_tensor = onnx.helper.make_tensor('conv2_b_tensor', data_type=onnx.TensorProto.FLOAT, dims=np.shape(conv2_b), vals=conv2_b.flatten())
    conv2_node = onnx.helper.make_node("Conv",
                                       inputs=['permute_out', 'conv2_w_tensor', 'conv2_b_tensor'],
                                       outputs=['conv2'],
                                       kernel_shape=kernel2_shape,
                                       strides=stride2,
                                       pads=pad2,
                                       dilations=dilation2,
                                       name="conv2")
    conv2_scale_node = onnx.helper.make_node("Scale", inputs=['conv2'], outputs=['conv2_scale'], scale=1.0 / G_scale[0])
    conv2_relu_node = onnx.helper.make_node('Relu', ['conv2_scale'], ['out'])
    node2 = [conv2_node] + [conv2_scale_node] + [conv2_relu_node]
    initializer2 = [conv2_w_tensor, conv2_b_tensor]
    
    nodes = node1 + [permute_node] + node2
    initializers = initializer1 + initializer2
    graph = onnx.helper.make_graph(nodes, "cnn_permute_net",inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input1_shape))],
                                   outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output2_shape))],initializer=initializers)
    model = onnx.helper.make_model(graph, producer_name='cnn_permute_net')
    with open(file_dir + "cnn_permute_net.onnx", "wb") as f:
        f.write(model.SerializeToString())
        print("Generate cnn_permute_net sucessfully!")

def run_model():
    input_shape = [1, 8, 64, 64]
    params = []
    cnn_model = onnx.load(file_dir + "cnn_permute_net.onnx")
    data = np.random.randint(0, 127, size=[11, input_shape[3], input_shape[2], input_shape[1]], dtype='uint8').astype('float32')
    data = data.astype('float32') 
    shape_dict_conv = {}
    shape_dict_conv['in'] = input_shape 
    mod, params = witin_frontend.frontend.from_onnx(cnn_model, shape_dict_conv)
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    
        
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
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, chip_type, output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)
    
def test_cnn_permute():
    generate_model()
    run_model()

if __name__ == "__main__":
    test_cnn_permute()
