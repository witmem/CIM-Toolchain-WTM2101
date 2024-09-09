import numpy as np
import onnx
import os
import datetime
import witin
from witin import *
np.random.seed(10)

root_dir = './tests/onnx/witin/wtm2101/operator/gru_2array/'

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


def generate_gru_model():
    #input_len=130, htidden_len=64
    input_len = 392
    hidden_len = 232
    gru_params_1 = np.random.randint(-127, 127, size=(input_len, hidden_len*3), dtype=np.int32).astype(np.float32)
    gru_params_2 = np.random.randint(-127, 127, size=(hidden_len, hidden_len*3), dtype=np.int32).astype(np.float32)
  
    gru_bias_1 = 128 * np.random.randint(-128, 127, size=(hidden_len*3,), dtype=np.int32).astype(np.float32)
    gru_bias_2 = 128 * np.random.randint(-128, 127, size=(hidden_len*3,), dtype=np.int32).astype(np.float32)
  
    gru_params1 = onnx.helper.make_tensor("gru_params1",
                      data_type=onnx.TensorProto.FLOAT, dims=(input_len, hidden_len*3), vals=gru_params_1.flatten())
    gru_params2 = onnx.helper.make_tensor("gru_params2",
                      data_type=onnx.TensorProto.FLOAT, dims=(hidden_len, hidden_len*3), vals=gru_params_2.flatten())
  
    gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                      data_type=onnx.TensorProto.FLOAT, dims=(hidden_len*3,), vals=gru_bias_1.flatten())
    gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                      data_type=onnx.TensorProto.FLOAT, dims=(hidden_len*3,), vals=gru_bias_2.flatten())
      
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
  
    gru_node1 = onnx.helper.make_node('Gru2Array',
        inputs=['in', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],
        input_offset=0,
        hidden_shift_bit=-1,   #输入scle
        scale_input=1024,
        scale_hidden=1024,
        scale_ones = 128,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[-7, -7, -2, -3], 
        clean_ht=0,
        outputs=['gru_out'],
        name="gru_node1")

    dnn_params_1 = np.random.randint(-127, 127, size=(hidden_len, 40), dtype=np.int32).astype(np.float32)
    dnn_bias_1 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)

    dnn_params1 = onnx.helper.make_tensor("dnn_params1",
                      data_type=onnx.TensorProto.FLOAT, dims=(hidden_len, 40), vals=dnn_params_1.flatten())
    
    dnn_bias1 = onnx.helper.make_tensor("dnn_bias1",
                      data_type=onnx.TensorProto.FLOAT, dims=(40,), vals=dnn_bias_1.flatten())

    
  
    dnn_node=onnx.helper.make_node('Gemm', inputs=['gru_out', 'dnn_params1', 'dnn_bias1'],
                                outputs=['dnn_out'], name="gemm2"
                                )
    dnn_scale_node = onnx.helper.make_node('Scale', ['dnn_out'], ['out'], scale=0.001953125)
   
    in_shape = (1, input_len)
    out_shape = (1, 40)
    nodes = [gru_node1] + [dnn_node] + [dnn_scale_node]
    initializer=[gru_params1, gru_params2, gru_bias1, gru_bias2, act_table, dnn_params1, dnn_bias1]
  
    graph = onnx.helper.make_graph(nodes, "test_gru",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                          list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                          list(out_shape))],
            initializer=initializer
              )
    model = onnx.helper.make_model(graph, producer_name='test_gru')
    with open(root_dir + 'test_gru_2_array_dnn' + '.onnx', "wb") as of:
        of.write(model.SerializeToString())
    print("generate {} sucessfully!".format('test_gru_2_array'  + '.onnx'))
      
def run_model():
    input_len = 392
    file_name = root_dir + 'test_gru_2_array_dnn'  + '.onnx'
    onnx_model = onnx.load(file_name)



    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, input_len)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    # print(mod)
    # data1 = np.round(np.random.rand(111, input_len) * 255).astype("float32")-128
    data1 = np.round(np.random.rand(111, input_len) * 255).astype("float32") - 128
    # print(data1[:10,:])
    input_dt = {}
    input_dt['in'] = witin.nd.array(data1)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    opti = './tests/onnx/witin/wtm2101/operator/gru_2array/gru_dnn.protobuf'
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        output_dir=build_dir,
                                        chip="BB04P1",
                                        optimize_method_config=opti)

    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data1))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)

def test_forward_gru2array_dnn():
    
    generate_gru_model()
    run_model()


if __name__ == '__main__':
    test_forward_gru2array_dnn()
    
