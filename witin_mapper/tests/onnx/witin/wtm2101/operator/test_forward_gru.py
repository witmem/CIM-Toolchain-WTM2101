import numpy as np
import onnx
from onnx import helper
import os
import witin
import datetime
from witin import *
np.random.seed(0)

root_dir = './model/pipeline/gru_model/'

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

def generate_gru_model_case1():
    input_size = 64
    output_size = 32
    gemm_params_0 = np.random.randint(-128, 127, size=(input_size, input_size), dtype=np.int32).astype(np.float32)
    gemm_bias_0 = np.random.randint(-128, 127, size=(input_size,), dtype=np.int32).astype(np.float32) * 128
    weight0 = onnx.helper.make_tensor("weight0",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(input_size, input_size),
                                      vals=gemm_params_0.flatten())
    bias0 = onnx.helper.make_tensor("bias0",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(input_size,),
                                    vals=gemm_bias_0.flatten())
    dnn_node = onnx.helper.make_node('Gemm',
                                     inputs=['in', 'weight0', 'bias0'],
                                     outputs=['dnn_out'],
                                     name='dnn0')
    scale_node = onnx.helper.make_node('Scale', ['dnn_out'], ['scale_out'], scale=1.0 / 1024)
    relu_node = onnx.helper.make_node('Relu', ['scale_out'], ['relu_out'])
    node0 = [dnn_node] + [scale_node] + [relu_node]
    initializer0 = [weight0, bias0]
    
    gru_weight_1 = np.random.randint(-128, 127, size=(input_size + output_size, 2 * output_size), dtype=np.int32).astype(np.float32)
    gru_weight_2 = np.random.randint(-128, 127, size=(output_size, output_size), dtype=np.int32).astype(np.float32)
    gru_weight_3 = np.random.randint(-128, 127, size=(input_size, output_size), dtype=np.int32).astype(np.float32)
    gru_bias_1 = np.random.randint(-128, 127, size=(2 * output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_bias_2 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_bias_3 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_weight1 = onnx.helper.make_tensor("gru_weight1",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(input_size + output_size, 2 * output_size),
                                          vals=gru_weight_1.flatten())
    gru_weight2 = onnx.helper.make_tensor("gru_weight2",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(output_size, output_size),
                                          vals=gru_weight_2.flatten())
    gru_weight3 = onnx.helper.make_tensor("gru_weight3",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(input_size, output_size),
                                          vals=gru_weight_3.flatten())
    gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2 * output_size,),
                                        vals=gru_bias_1.flatten())
    gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(output_size,),
                                        vals=gru_bias_2.flatten())
    gru_bias3 = onnx.helper.make_tensor("gru_bias3",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(output_size,),
                                        vals=gru_bias_3.flatten())
    act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
    sigmoid_act_table = np.array(act_input1024)
    sigmoid_table_data = sigmoid_q(sigmoid_act_table, 6, 7)
    tanh_act_table = np.array(act_input1024)
    tanh_table_data = tanh_q(tanh_act_table, 8, 7)
    act_table = np.concatenate((sigmoid_table_data, tanh_table_data), axis=0)
    act_table = np.reshape(act_table, (2, 1024)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 1024),
                                        vals=act_table.flatten())
    gru_node1 = onnx.helper.make_node('Gru',
                                      inputs=['relu_out', 'gru_weight1', 'gru_weight2', 'gru_weight3', 'gru_bias1', 'gru_bias2', 'gru_bias3'],
                                      scale_zr=1024,
                                      scale_ht=1024,
                                      scale_in=1024,
                                      scale_ones = 127,
                                      activate_type=['sigmoid', 'tanh'],
                                      activate_table=act_table,
                                      shift_bits=[-8, -8],
                                      clean_ht=0,
                                      outputs=['out'],
                                      name="gru_node1")
    nodes1 = [gru_node1]
    initializer1 = [gru_weight1, gru_weight2, gru_weight3, gru_bias1, gru_bias2, gru_bias3, act_table]
    
    nodes = node0 + nodes1
    initializers = initializer0 + initializer1
    in_shape = (1, input_size)
    out_shape = (1, output_size)
    onnx_graph_name = "test_gemm_gru_new"
    graph = onnx.helper.make_graph(nodes,
                                   onnx_graph_name,
                                   inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
                                   outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
                                   initializer=initializers)
    model = helper.make_model(graph, producer_name=onnx_graph_name)
    with open(root_dir + onnx_graph_name + ".onnx", "wb") as of:
        of.write(model.SerializeToString())
        print("generate {} sucessfully!".format(onnx_graph_name + ".onnx"))

def generate_gru_model_case():
    for case_idx in range(2):
      input_size = 64
      output_size = 32            
      # gru
      if case_idx == 0:
          gru_weight_1 = np.random.randint(-128, 127, size=(input_size + output_size, 2 * output_size), dtype=np.int32).astype(np.float32)
          gru_weight_2 = np.random.randint(-128, 127, size=(output_size, output_size), dtype=np.int32).astype(np.float32)
          gru_weight_3 = np.random.randint(-128, 127, size=(input_size, output_size), dtype=np.int32).astype(np.float32)
      else:
          gru_weight_1 = np.random.randint(-144, 144, size=(input_size + output_size, 2 * output_size), dtype=np.int32).astype(np.float32)
          gru_weight_2 = np.random.randint(-144, 144, size=(output_size, output_size), dtype=np.int32).astype(np.float32)
          gru_weight_3 = np.random.randint(-144, 144, size=(input_size, output_size), dtype=np.int32).astype(np.float32)
    
      gru_bias_1 = np.random.randint(-128, 127, size=(2 * output_size,), dtype=np.int32).astype(np.float32) * 128 
      gru_bias_2 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
      gru_bias_3 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
      
      gru_weight1 = onnx.helper.make_tensor("gru_weight1",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(input_size + output_size, 2 * output_size),
                                             vals=gru_weight_1.flatten())
      gru_weight2 = onnx.helper.make_tensor("gru_weight2",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(output_size, output_size),
                                             vals=gru_weight_2.flatten())
      gru_weight3 = onnx.helper.make_tensor("gru_weight3",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(input_size, output_size),
                                             vals=gru_weight_3.flatten())
  
      gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(2 * output_size,),
                                           vals=gru_bias_1.flatten())
      gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(output_size,),
                                           vals=gru_bias_2.flatten())
      gru_bias3 = onnx.helper.make_tensor("gru_bias3",
                                          data_type=onnx.TensorProto.FLOAT,    
                                          dims=(output_size,),  
                                          vals=gru_bias_3.flatten())
  
      act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
      sigmoid_act_table = np.array(act_input1024)
      sigmoid_table_data = sigmoid_q(sigmoid_act_table, 6, 7)
      tanh_act_table = np.array(act_input1024)
      tanh_table_data = tanh_q(tanh_act_table, 8, 7)
      act_table = np.concatenate((sigmoid_table_data, tanh_table_data), axis=0)
      act_table = np.reshape(act_table, (2, 1024)).astype(np.float32)
      act_table = onnx.helper.make_tensor("act_table",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(2, 1024),
                                          vals=act_table.flatten())
  
      gru_node1 = onnx.helper.make_node(
          'Gru',
          inputs=['in', 'gru_weight1', 'gru_weight2', 'gru_weight3', 'gru_bias1', 'gru_bias2', 'gru_bias3'],
          scale_zr=1024,
          scale_ht=1024,
          scale_in=1024,
          scale_ones = 127,
          activate_type=['sigmoid', 'tanh'],
          activate_table=act_table,
          shift_bits=[-8, -8],
          clean_ht=0,
          outputs=['out'],
          name="gru_node1")
  
      in_shape = (1, input_size)
      out_shape = (1, output_size)
      initializers = [
          gru_weight1, gru_weight2, gru_weight3, gru_bias1, gru_bias2, gru_bias3, act_table
      ]
      
      if case_idx == 0:
          onnx_graph_name = "test_gru_new"
      else:
          onnx_graph_name = "test_gru_new_sparse"
      graph = onnx.helper.make_graph(
          [gru_node1],
          onnx_graph_name,
          inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
          outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
          initializer=initializers)
      model = helper.make_model(graph, producer_name=onnx_graph_name)
      with open(root_dir + onnx_graph_name + ".onnx", "wb") as of:
          of.write(model.SerializeToString())
      print("generate {} sucessfully!".format(onnx_graph_name + ".onnx"))

def generate_gru_model_shift4_case():
    input_size = 64
    output_size = 32
    gemm_params_0 = np.random.randint(-128, 127, size=(input_size, input_size), dtype=np.int32).astype(np.float32)
    gemm_bias_0 = np.random.randint(-128, 127, size=(input_size,), dtype=np.int32).astype(np.float32) * 128
    weight0 = onnx.helper.make_tensor("weight0",
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=(input_size, input_size),
                                      vals=gemm_params_0.flatten())
    bias0 = onnx.helper.make_tensor("bias0",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(input_size,),
                                    vals=gemm_bias_0.flatten())
    dnn_node = onnx.helper.make_node('Gemm',
                                     inputs=['in', 'weight0', 'bias0'],
                                     outputs=['dnn_out'],
                                     name='dnn0')
    scale_node = onnx.helper.make_node('Scale', ['dnn_out'], ['scale_out'], scale=1.0 / 1024)
    relu_node = onnx.helper.make_node('Relu', ['scale_out'], ['relu_out'])
    node0 = [dnn_node] + [scale_node] + [relu_node]
    initializer0 = [weight0, bias0]
    
    gru_weight_1 = np.random.randint(-128, 127, size=(input_size + output_size, 2 * output_size), dtype=np.int32).astype(np.float32)
    gru_weight_2 = np.random.randint(-128, 127, size=(output_size, output_size), dtype=np.int32).astype(np.float32)
    gru_weight_3 = np.random.randint(-128, 127, size=(input_size, output_size), dtype=np.int32).astype(np.float32)
    gru_bias_1 = np.random.randint(-128, 127, size=(2 * output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_bias_2 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_bias_3 = np.random.randint(-128, 127, size=(output_size,), dtype=np.int32).astype(np.float32) * 128
    gru_weight1 = onnx.helper.make_tensor("gru_weight1",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(input_size + output_size, 2 * output_size),
                                          vals=gru_weight_1.flatten())
    gru_weight2 = onnx.helper.make_tensor("gru_weight2",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(output_size, output_size),
                                          vals=gru_weight_2.flatten())
    gru_weight3 = onnx.helper.make_tensor("gru_weight3",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(input_size, output_size),
                                          vals=gru_weight_3.flatten())
    gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2 * output_size,),
                                        vals=gru_bias_1.flatten())
    gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(output_size,),
                                        vals=gru_bias_2.flatten())
    gru_bias3 = onnx.helper.make_tensor("gru_bias3",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(output_size,),
                                        vals=gru_bias_3.flatten())
    act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
    sigmoid_act_table = np.array(act_input1024)
    sigmoid_table_data = sigmoid_q(sigmoid_act_table, 6, 7)
    tanh_act_table = np.array(act_input1024)
    tanh_table_data = tanh_q(tanh_act_table, 8, 7)
    act_table = np.concatenate((sigmoid_table_data, tanh_table_data), axis=0)
    act_table = np.reshape(act_table, (2, 1024)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 1024),
                                        vals=act_table.flatten())
    gru_node1 = onnx.helper.make_node('Gru',
                                      inputs=['relu_out', 'gru_weight1', 'gru_weight2', 'gru_weight3', 'gru_bias1', 'gru_bias2', 'gru_bias3'],
                                      scale_zr=1024,
                                      scale_ht=1024,
                                      scale_in=1024,
                                      scale_ones = 127,
                                      activate_type=['sigmoid', 'tanh'],
                                      activate_table=act_table,
                                      shift_bits=[-8, -8, -8, -8],
                                      clean_ht=0,
                                      outputs=['out'],
                                      name="gru_node1")
    nodes1 = [gru_node1]
    initializer1 = [gru_weight1, gru_weight2, gru_weight3, gru_bias1, gru_bias2, gru_bias3, act_table]
    
    nodes = node0 + nodes1
    initializers = initializer0 + initializer1
    in_shape = (1, input_size)
    out_shape = (1, output_size)
    onnx_graph_name = "test_gru_shift_bits_new"
    graph = onnx.helper.make_graph(nodes,
                                   onnx_graph_name,
                                   inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
                                   outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))],
                                   initializer=initializers)
    model = helper.make_model(graph, producer_name=onnx_graph_name)
    with open(root_dir + onnx_graph_name + ".onnx", "wb") as of:
        of.write(model.SerializeToString())
        print("generate {} sucessfully!".format(onnx_graph_name + ".onnx"))

def get_tvm_output(graph_def, in_shape, data, opt_config=""):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, in_shape)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target=target,
                                                               target_host=target_host,
                                                               params=params,
                                                               input_data=input_dt,
                                                               chip="BB04P1",
                                                               output_dir=build_dir,
                                                               optimize_method_config=opt_config)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in1', witin.nd.array(data))
    # execute
    m.run()

def run_shift4_input_case():
    generate_gru_model_shift4_case()
    run_gru_new_shift_bits4_forward()

def test_gru_new_forward():
    generate_gru_model_case()
    generate_gru_model_case1()
    run_gru_new_forward()
    run_shift4_input_case()
    run_gru_new_sprase_forward()
    run_gru_new_signed_input_forward()
    run_gru_new_weight_amplify_forward()
    run_gru_new_weight_region_forward()
    run_gru_new_weight_input_amplify_forward()
    run_gru_new_output_repeat_forward()

def run_gru_new_shift_bits4_forward():
    file_name = root_dir + './test_gru_shift_bits_new.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, intput_size, data, "")

def run_gru_new_forward():
    file_name = root_dir + './test_gru_new.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, intput_size, data, "")
    
def run_gru_new_output_repeat_forward():
    file_name = root_dir + './test_gru_new.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255).astype("float32")
    opt_config = root_dir + "/optimize_config_gru_new_output_repeat.protobuf"
    tvm_out = get_tvm_output(onnx_model, intput_size, data, opt_config)

def run_gru_new_sprase_forward():
    file_name = root_dir + './test_gru_new_sparse.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255).astype("float32")
    opt_config =  root_dir + "/optimize_config_gru_new_sparse.protobuf"
    tvm_out = get_tvm_output(onnx_model, intput_size, data, opt_config)

def run_gru_new_signed_input_forward():
    file_name = root_dir + './test_gru_new.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255 - 128).astype("float32")
    opt_config = root_dir + "/optimize_config_gru_new_signed_pn.protobuf"
    tvm_out = get_tvm_output(onnx_model, intput_size, data, opt_config)

def run_gru_new_weight_amplify_forward():
    file_name = root_dir + './test_gru_new.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255).astype("float32")
    opt_config =  root_dir + "/optimize_config_gru_new_weight_amplify.protobuf"
    tvm_out = get_tvm_output(onnx_model, intput_size, data, opt_config)

def run_gru_new_weight_region_forward():
    file_name = root_dir + './test_gru_new.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255).astype("float32")
    opt_config =  root_dir + "/optimize_config_gru_new_weight_region.protobuf"
    tvm_out = get_tvm_output(onnx_model, intput_size, data, opt_config)

def run_gru_new_weight_input_amplify_forward():
    file_name = root_dir + './test_gemm_gru_new.onnx'
    onnx_model = onnx.load(file_name)
    intput_size = 64
    data = np.round(np.random.rand(11, intput_size) * 255).astype("float32")
    opt_config =  root_dir + "/optimize_config_gru_new_input_amplify.protobuf"
    tvm_out = get_tvm_output(onnx_model, intput_size, data, opt_config)

if __name__ == '__main__':
    test_gru_new_forward()
