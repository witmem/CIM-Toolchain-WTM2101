import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = "./model/"

def generate_model_single_conv2d():
    G_scale = [1024, 512]
    input_shape = [
        [1, 8, 6, 6],
        [1, 3, 10, 10],
    ]
    output_shape = [
        [1, 1024, 3, 3],
        [1, 2000, 4, 4]
    ]
    kernel_tensor_shape = [
        [1024, 8, 2, 2],
        [2000, 3, 4, 4],
    ]
    bias_tensor_shape = [
        [1024],
        [2000],
    ]
    kernel_shape = [
        [2, 2],
        [4, 4],
    ]
    stride = [
        [2, 2],
        [2, 2]
    ]
    pad = [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    dilation = [
        [1, 1],
        [1, 1]
    ]
    for case_idx in range(len(input_shape)):
      #layer 1
      conv1_w = np.random.randint(-127, 127,size=kernel_tensor_shape[case_idx], dtype='int8').astype('float32')
      conv1_b = np.random.randint(-127, 127, size=bias_tensor_shape[case_idx], dtype='int8').astype('float32') * 128
      
	    #layer1
      conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(conv1_w),
                                               vals=conv1_w.flatten())
      conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(conv1_b),
                                               vals=conv1_b.flatten())
	  								 
	    #make node
      conv1_node = onnx.helper.make_node("Conv",
                                         inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                         outputs=['conv1'],
                                         kernel_shape=kernel_shape[case_idx],
                                         strides=stride[case_idx],
                                         pads=pad[case_idx],
                                         dilations=dilation[case_idx],
                                         name="conv1")
      conv1_scale_node = onnx.helper.make_node("Scale", inputs=['conv1'], outputs=['conv1_scale'], scale=1.0 / G_scale[case_idx])
      conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['out'])
      
      node1 = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
      initializer1 = [conv1_w_tensor, conv1_b_tensor]
      
      nodes = node1 
      initializers = initializer1 
	  
      dccrn_graph = onnx.helper.make_graph(
          nodes,
          "conv2d_extend_array_net",
          inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input_shape[case_idx]))],
          outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape[case_idx]))],
          initializer=initializers)
      dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='dilated_net')
      
      with open(file_dir + "conv2d_single_extend_array_net" + str(case_idx) + ".onnx", "wb") as f:
          f.write(dccrn_model.SerializeToString())
      print("Generate conv2d_single_extend_array_net"  + str(case_idx) + " sucessfully!")

def run_model_single_conv2d():
    input_shape = [
      [1, 8, 6, 6],
      [1, 3, 10, 10],
    ]
    for case_idx in range(len(input_shape)):
      params = []
      conv_model = onnx.load(file_dir + "conv2d_single_extend_array_net" + str(case_idx) + ".onnx")
      data = np.random.randint(0, 127, size=[11,
                                             input_shape[case_idx][3],
                                             input_shape[case_idx][2],
                                             input_shape[case_idx][1]], dtype='uint8').astype('float32')
      data = data.astype('float32')  # NCHW

      shape_dict_conv = {}
      shape_dict_conv['in'] = input_shape[case_idx]  # NCHW
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

def generate_model_multiple_conv2d():
    G_scale = [
     [1024, 512],
     [2048, 256]
    ]
    input_shape = [
      [1, 8, 6, 6],
      [1, 3, 10, 10]  
    ]
    output_shape = [
      [1, 1024, 3, 3],
      [1, 2001, 4, 4]
    ]
    kernel_tensor_shape = [
      [[8, 8, 1, 1], [1024, 8, 2, 2]],
      [[3, 3, 1, 1], [2001, 3, 4, 4]]
    ]
    bias_tensor_shape = [
      [[8], [1024]],
      [[3], [2001]]
    ]
    kernel_shape = [
      [[1, 1], [2, 2]],
      [[1, 1], [4, 4]]
    ]
    stride = [
      [[1, 1], [2, 2]],
      [[1, 1], [2, 2]]
    ]
    pad = [
      [[0, 0, 0, 0], [0, 0, 0, 0]],
      [[0, 0, 0, 0], [0, 0, 0, 0]]
    ]
    dilation = [
      [[1, 1], [1, 1]],
      [[1, 1], [1, 1]]
    ]
    for case_idx in range(len(input_shape)):
      conv1_w = np.random.randint(-127, 127,size=kernel_tensor_shape[case_idx][0], dtype='int8').astype('float32')
      conv1_b = np.random.randint(-127, 127, size=bias_tensor_shape[case_idx][0], dtype='int8').astype('float32') * 128
      conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(conv1_w),
                                               vals=conv1_w.flatten())
      conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(conv1_b),
                                               vals=conv1_b.flatten())				 
	    # make layer1
      conv1_node = onnx.helper.make_node("Conv",
                                         inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                         outputs=['conv1'],
                                         kernel_shape=kernel_shape[case_idx][0],
                                         strides=stride[case_idx][0],
                                         pads=pad[case_idx][0],
                                         dilations=dilation[case_idx][0],
                                         name="conv1")
      conv1_scale_node = onnx.helper.make_node("Scale", inputs=['conv1'], outputs=['conv1_scale'], scale=1.0 / G_scale[case_idx][0])
      conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['con1_relu_out'])
      
      node1 = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
      initializer1 = [conv1_w_tensor, conv1_b_tensor]
      
      
      conv2_w = np.random.randint(-127, 127,size=kernel_tensor_shape[case_idx][1], dtype='int8').astype('float32')
      conv2_b = np.random.randint(-127, 127, size=bias_tensor_shape[case_idx][1], dtype='int8').astype('float32') * 128
      conv2_w_tensor = onnx.helper.make_tensor('conv2_w_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(conv2_w),
                                               vals=conv2_w.flatten())
      conv2_b_tensor = onnx.helper.make_tensor('conv2_b_tensor',
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=np.shape(conv2_b),
                                               vals=conv2_b.flatten())				 
	    # make layer1
      conv2_node = onnx.helper.make_node("Conv",
                                         inputs=['con1_relu_out', 'conv2_w_tensor', 'conv2_b_tensor'],
                                         outputs=['conv2'],
                                         kernel_shape=kernel_shape[case_idx][1],
                                         strides=stride[case_idx][1],
                                         pads=pad[case_idx][1],
                                         dilations=dilation[case_idx][1],
                                         name="conv2")
      conv2_scale_node = onnx.helper.make_node("Scale", inputs=['conv2'], outputs=['conv2_scale'], scale=1.0 / G_scale[case_idx][1])
      conv2_relu_node = onnx.helper.make_node('Relu', ['conv2_scale'], ['out'])
      
      node2 = [conv2_node] + [conv2_scale_node] + [conv2_relu_node]
      initializer2 = [conv2_w_tensor, conv2_b_tensor]
      
      nodes = node1 +node2 
      initializers = initializer1 + initializer2 
	  
      dccrn_graph = onnx.helper.make_graph(
          nodes,
          "conv2d_multilpe_extend_array_net",
          inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input_shape[case_idx]))],
          outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape[case_idx]))],
          initializer=initializers)
      dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='dilated_net')
      
      with open(file_dir + "conv2d_multilpe_extend_array_net" + str(case_idx) + ".onnx", "wb") as f:
          f.write(dccrn_model.SerializeToString())
      print("Generate conv2d_multilpe_extend_array_net"  + str(case_idx) + " sucessfully!")

def run_model_multiple_conv2d():
    input_shape = [
      [1, 8, 6, 6],
      [1, 3, 10, 10],
    ]
    for case_idx in range(len(input_shape)):
      params = []
      conv_model = onnx.load(file_dir + "conv2d_multilpe_extend_array_net" + str(case_idx) + ".onnx")
      data = np.random.randint(0, 127, size=[11,
                                             input_shape[case_idx][3],
                                             input_shape[case_idx][2],
                                             input_shape[case_idx][1]], dtype='uint8').astype('float32')
      data = data.astype('float32')  # NCHW

      shape_dict_conv = {}
      shape_dict_conv['in'] = input_shape[case_idx]  # NCHW
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

def test_single_conv2d_array_extend():
  generate_model_single_conv2d()
  run_model_single_conv2d()
  
def test_multiple_conv2d_array_extend():
  generate_model_multiple_conv2d()
  run_model_multiple_conv2d()

if __name__ == "__main__":
  test_single_conv2d_array_extend()
  test_multiple_conv2d_array_extend()
