import numpy as np
import onnx
import os
import datetime
import witin
from witin import *

root_dir = './model/'

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

def get_tvm_output(graph_def, clear_ht_frame):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 4)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    data1 = np.round(np.random.rand(111, 4) * 255).astype("float32")
    input_dt = {}
    input_dt['in'] = witin.nd.array(data1)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        output_dir=build_dir,
                                        chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    if clear_ht_frame == 0:
        from tvm.contrib import npu_graph_runtime
        m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
        m.set_input('in', witin.nd.array(data1))
        # execute
        m.run()
        witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]


def generate_gru_model(clean_ht_frames):
    case_idx = 0
    for clean_ht_frame in clean_ht_frames:
      gru_params_1 = np.random.randint(-256, 255, size=(8, 8), dtype=np.int32).astype(np.float32)
      gru_params_2 = np.random.randint(-256, 255, size=(8, 4), dtype=np.int32).astype(np.float32)
  
      gru_bias_1 = 128 * np.random.randint(-128, 127, size=(8,), dtype=np.int32).astype(np.float32)
      gru_bias_2 = 128 * np.random.randint(-128, 127, size=(4,), dtype=np.int32).astype(np.float32)
  
      gru_params1 = onnx.helper.make_tensor("gru_params1",
                      data_type=onnx.TensorProto.FLOAT, dims=(8, 8), vals=gru_params_1.flatten())
      gru_params2 = onnx.helper.make_tensor("gru_params2",
                      data_type=onnx.TensorProto.FLOAT, dims=(8, 4), vals=gru_params_2.flatten())
  
      gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                      data_type=onnx.TensorProto.FLOAT, dims=(8,), vals=gru_bias_1.flatten())
      gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                      data_type=onnx.TensorProto.FLOAT, dims=(4,), vals=gru_bias_2.flatten())
      
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
          inputs=['in', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],
          scale_zr=1024,
          scale_ht=1024,
          scale_ones = 127,
          activate_type=['sigmoid', 'tanh'],
          activate_table=act_table,
          shift_bits=[0, -4],
          clean_ht=clean_ht_frame,
          outputs=['out'],
          name="gru_node1")
   
      in_shape = (1, 4)
      out_shape = (1, 4)
      initializer=[gru_params1, gru_params2, gru_bias1, gru_bias2, act_table]
  
      graph = onnx.helper.make_graph([gru_node1], "test_gru",
              inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                          list(in_shape))],
              outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                          list(out_shape))],
              initializer=initializer
              )
      model = onnx.helper.make_model(graph, producer_name='test_gru')
      with open(root_dir + 'test_gru_' + str(case_idx) + '.onnx', "wb") as of:
          of.write(model.SerializeToString())
      print("generate {} sucessfully!".format('test_gru_' + str(case_idx) + '.onnx'))
      case_idx = case_idx + 1
      
def test_gru_clear_ht():
    # clear_ht are only supported in gru forward func and the layer config are no supported.
    # so not run the func for clear_ht(non 0) in pipeline test
    clear_ht_frames = [0, 20]
    generate_gru_model(clear_ht_frames)
    case_idx = 0
    for clear_ht_frame in clear_ht_frames:
        file_name = root_dir + './test_gru_' + str(case_idx) + '.onnx'
        onnx_model = onnx.load(file_name)
        get_tvm_output(onnx_model, clear_ht_frame)
        case_idx = case_idx + 1

if __name__ == '__main__':
    np.random.seed(10)
    test_gru_clear_ht()
