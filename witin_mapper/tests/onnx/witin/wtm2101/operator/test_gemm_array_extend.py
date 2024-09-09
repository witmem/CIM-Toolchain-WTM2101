import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = "./model/"

def generate_model_single_gemm():
    G_scale = [1024, 512, 2048]
    input_shape = [
      [1, 8],
      [1, 224],
      [1, 6],
    ]
    kernel_shape = [
      [8, 1024],
      [224, 1500],
      [6, 1797]
    ]
    bias_shape = [
      [1024],
      [1500],
      [1797]
    ]
    output_shape = [
      [1, 1024],
      [1, 1500],
      [1, 1797]
    ]
    for case_idx in range(len(input_shape)):
      dnn_weight_params = np.random.randint(-100, 100, size=kernel_shape[case_idx],
                                            dtype=np.int32).astype('float32')
      dnn_bias_params = 128 * np.random.randint(-127, 127, size=bias_shape[case_idx],
                                                dtype=np.int32).astype('float32')
  
      dnn_weight = onnx.helper.make_tensor("dnn_weight",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=kernel_shape[case_idx],
                                           vals=dnn_weight_params.flatten())
  
      dnn_bias = onnx.helper.make_tensor("dnn_bias",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=bias_shape[case_idx],
                                         vals=dnn_bias_params.flatten())
  
      dnn_node = onnx.helper.make_node("Gemm",
                                       inputs=['in','dnn_weight','dnn_bias'],
                                       outputs=['dnn_out'],
                                       name="dnn_node")
      # G值
      scale_node = onnx.helper.make_node('Scale', ['dnn_out'], ['scale_out'], scale=1.0 / G_scale[case_idx])

      relu_node = onnx.helper.make_node('Relu', ['scale_out'], ['out'])
  
  
      in_shape = input_shape[case_idx]
      out_shape = output_shape[case_idx]
  
      nodes = [dnn_node,scale_node,relu_node]
      name = "scale_G"
      inputs = [onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))]
      outputs = [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))]
      initializer = [dnn_weight, dnn_bias]
  
      graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer)
      model = onnx.helper.make_model(graph, producer_name="dccrn_gemm")

      with open(file_dir + 'test_gemm_array_extend_' + str(case_idx) + '.onnx', "wb") as f:
          f.write(model.SerializeToString())
      print("Generate {} sucessfully!".format('test_gemm_array_extend_' + str(case_idx) + '.onnx'))

def generate_model_multiple_gemm():
    # dnn1 layer
    dnn1_weight_params = np.random.randint(-100, 100, size=[16, 100],
                                           dtype=np.int32).astype('float32')
    dnn1_bias_params = 128 * np.random.randint(-127, 127, size=[100],
                                               dtype=np.int32).astype('float32')
    dnn1_weight = onnx.helper.make_tensor("dnn1_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=[16, 100],
                                          vals=dnn1_weight_params.flatten())
    dnn1_bias = onnx.helper.make_tensor("dnn1_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[100],
                                        vals=dnn1_bias_params.flatten())
    dnn1_node = onnx.helper.make_node("Gemm",
                                      inputs=['in','dnn1_weight','dnn1_bias'],
                                      outputs=['dnn1_out'],
                                      name="dnn1")
    dnn1_scale_node = onnx.helper.make_node('Scale', ['dnn1_out'], ['dnn1_scale_out'], scale=1.0 / 1024)
    dnn1_relu_node = onnx.helper.make_node('Relu', ['dnn1_scale_out'], ['dnn1_relu_out'])
    dnn1_nodes = [dnn1_node] + [dnn1_scale_node] + [dnn1_relu_node]
    dnn1_initializers = [dnn1_weight, dnn1_bias]
    
    # dnn2 layer
    dnn2_weight_params = np.random.randint(-100, 100, size=[100, 1200],
                                           dtype=np.int32).astype('float32')
    dnn2_bias_params = 128 * np.random.randint(-127, 127, size=[1200],
                                               dtype=np.int32).astype('float32')
    dnn2_weight = onnx.helper.make_tensor("dnn2_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=[100, 1200],
                                          vals=dnn2_weight_params.flatten())
    dnn2_bias = onnx.helper.make_tensor("dnn2_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[1200],
                                        vals=dnn2_bias_params.flatten())
    dnn2_node = onnx.helper.make_node("Gemm",
                                      inputs=['dnn1_relu_out','dnn2_weight','dnn2_bias'],
                                      outputs=['dnn2_out'],
                                      name="dnn2")
    dnn2_scale_node = onnx.helper.make_node('Scale', ['dnn2_out'], ['dnn2_scale_out'], scale=1.0 / 1024)
    dnn2_relu_node = onnx.helper.make_node('Relu', ['dnn2_scale_out'], ['out'])

    dnn2_nodes = [dnn2_node] + [dnn2_scale_node] + [dnn2_relu_node]
    dnn2_initializers = [dnn2_weight, dnn2_bias]
    
    nodes = dnn1_nodes + dnn2_nodes
    initializer = dnn1_initializers + dnn2_initializers
    name = "gemm_array_expand"
    inputs = [onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list([1, 16]))]
    outputs = [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list([1, 1200]))]
    
    graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer)
    model = onnx.helper.make_model(graph, producer_name="gemm_array_expand")
    with open(file_dir + 'test_gemm_array_extend_net.onnx', "wb") as f:
        f.write(model.SerializeToString())
    print("Generate {} sucessfully!".format('test_gemm_array_extend_net.onnx'))

def run_model_single_gemm():
    input_shape = [
      [1, 8],
      [1, 224],
      [1, 6],
    ]
    for case_idx in range(len(input_shape)):
      params = []
      gemm_model = onnx.load(file_dir + "test_gemm_array_extend_" + str(case_idx) + ".onnx")
      data = np.round(np.random.rand(11, input_shape[case_idx][1]) * 255).astype("float32")
      
      shape_dict_conv = {}
      shape_dict_conv['in'] = input_shape[case_idx]  # NCHW
      mod, params = witin_frontend.frontend.from_onnx(gemm_model, shape_dict_conv)
  
      input_dt = {}
      input_dt['in'] = witin.nd.array(data)
      dateTimeObj = datetime.datetime.now()
      timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
      build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
      chip_type = "BB04P1"
      # build
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

def run_model_multiple_gemm():
    input_shape = [1, 16]
    params = []
    gemm_model = onnx.load(file_dir + "test_gemm_array_extend_net.onnx")
    data = np.round(np.random.rand(11, input_shape[1]) * 255).astype("float32")
    
    shape_dict_conv = {}
    shape_dict_conv['in'] = input_shape  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(gemm_model, shape_dict_conv)

    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    chip_type = "BB04P1"
    # build
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

def test_single_gemm():
  generate_model_single_gemm()
  run_model_single_gemm()

def test_multiple_gemm():
  generate_model_multiple_gemm()
  run_model_multiple_gemm()

def test_gemm_extend_array():
  test_single_gemm()
  test_multiple_gemm()

if __name__ == "__main__":
  test_gemm_extend_array()
