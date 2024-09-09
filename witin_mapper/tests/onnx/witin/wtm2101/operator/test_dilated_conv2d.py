import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = "./model/"

def generate_model():
    G_scale = [1024, 512, 2048]
    input_shape = [
        [1, 8, 64, 64],
        [1, 3, 128, 128],
        [1, 32, 32, 32]
    ]
    output_shape = [
        [1, 32, 30, 30],
        [1, 16, 62, 62],
        [1, 64, 15, 15]
    ]
    kernel_tensor_shape = [
        [32, 8, 5, 5],
        [16, 3, 3, 3],
        [64, 32, 3, 3]
    ]
    bias_tensor_shape = [
        [32],
        [16],
        [64]
    ]
    kernel_shape = [
        [5, 5],
        [3, 3],
        [3, 3]
    ]
    stride = [
        [2, 2],
        [2, 2],
        [2, 2]
    ]
    pad = [
        [2, 2, 2, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    dilation = [
        [2, 2],
        [3, 3],
        [2, 2]
    ]
    for case_idx in range(len(input_shape)):
      #layer 1
      conv1_w = np.random.randint(-127, 127,size=kernel_tensor_shape[case_idx], dtype='int8').astype('float32')
      conv1_b = np.random.randint(-127, 127, size=bias_tensor_shape[case_idx], dtype='int8').astype('float32')
      conv1_b = conv1_b * 128
      
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
          "dilated_net",
          inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input_shape[case_idx]))],
          outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape[case_idx]))],
          initializer=initializers)
      dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='dilated_net')
      
      with open(file_dir + "dilated_conv2d_" + str(case_idx) + ".onnx", "wb") as f:
          f.write(dccrn_model.SerializeToString())
      print("Generate dilated_conv2d_"  + str(case_idx) + " sucessfully!")
	

def run_model():
    input_shape = [
       [1, 8, 64, 64],
       [1, 3, 128, 128],
       [1, 32, 32, 32]
    ]
    for case_idx in range(len(input_shape)):
      params = []
      conv_model = onnx.load(file_dir + "dilated_conv2d_" + str(case_idx) + ".onnx")
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
      witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
      print(witin_output_list[0].shape)

def test_dilated_conv2d():
   generate_model()
   run_model()

if __name__ == "__main__":
   test_dilated_conv2d()	
