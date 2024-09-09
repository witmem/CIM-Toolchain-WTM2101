from http.client import ImproperConnectionState
from ssl import VERIFY_DEFAULT
import numpy as np
import math
import onnx
from onnx import helper, TensorProto, mapping
import scipy
from scipy.fftpack import shift
import witin
from witin import *
import datetime
root_dir = './model/'

def generate_mul_shift_model():    
    dnn_weight_params1 = np.random.randint(-100,100,size=(8,8),dtype=np.int32).astype(np.float32)
    dnn_bias_params1 = 128 * np.random.randint(-128,127,size=(8,),dtype=np.int32).astype(np.float32)

    dnn_weight1 = helper.make_tensor("dnn_weight1",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,8),
                                            vals=dnn_weight_params1.flatten())

    dnn_bias1 = helper.make_tensor("dnn_bias1",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(8,),
                                        vals=dnn_bias_params1.flatten())

    dnn_node1 = helper.make_node("Gemm",
                                    inputs=['in1','dnn_weight1','dnn_bias1'],
                                    outputs=['dnn_out1'],
                                    name="dnn_node1")

    dnn_weight_params2 = np.random.randint(-100,100,size=(8,8),dtype=np.int32).astype(np.float32)
    dnn_bias_params2 = 128 * np.random.randint(-128,127,size=(8,),dtype=np.int32).astype(np.float32)

    dnn_weight2 = helper.make_tensor("dnn_weight2",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,8),
                                            vals=dnn_weight_params2.flatten())

    dnn_bias2 = helper.make_tensor("dnn_bias2",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(8,),
                                        vals=dnn_bias_params2.flatten())

    dnn_node2 = helper.make_node("Gemm",
                                    inputs=['in2','dnn_weight2','dnn_bias2'],
                                    outputs=['dnn_out2'],
                                    name="dnn_node2")
    # shift_bit is opti_params
    mul_node = onnx.helper.make_node('Mul', inputs = ['dnn_out1','dnn_out2'], outputs = ['out'], shift_bit = 3)

    in_shape1 = (1,8)
    in_shape2 = (1,8)
    out_shape = (1,8)

    nodes = [dnn_node1,dnn_node2,mul_node]
    name = "mul_shift"
    inputs = [helper.make_tensor_value_info("in1",onnx.TensorProto.FLOAT,list(in_shape1)),
                helper.make_tensor_value_info("in2",onnx.TensorProto.FLOAT,list(in_shape2))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [dnn_weight1,dnn_bias1,dnn_weight2,dnn_bias2]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name="mul_shift")

    with open(root_dir + 'mul_shift.onnx',"wb") as of:
        of.write(model.SerializeToString())

def generate_add_shift_model():    
    dnn_weight_params1 = np.random.randint(-100,100,size=(8,8),dtype=np.int32).astype(np.float32)
    dnn_bias_params1 = 128 * np.random.randint(-128,127,size=(8,),dtype=np.int32).astype(np.float32)

    dnn_weight1 = helper.make_tensor("dnn_weight1",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,8),
                                            vals=dnn_weight_params1.flatten())

    dnn_bias1 = helper.make_tensor("dnn_bias1",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(8,),
                                        vals=dnn_bias_params1.flatten())

    dnn_node1 = helper.make_node("Gemm",
                                    inputs=['in1','dnn_weight1','dnn_bias1'],
                                    outputs=['dnn_out1'],
                                    name="dnn_node1")

    dnn_weight_params2 = np.random.randint(-100,100,size=(8,8),dtype=np.int32).astype(np.float32)
    dnn_bias_params2 = 128 * np.random.randint(-128,127,size=(8,),dtype=np.int32).astype(np.float32)

    dnn_weight2 = helper.make_tensor("dnn_weight2",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,8),
                                            vals=dnn_weight_params2.flatten())

    dnn_bias2 = helper.make_tensor("dnn_bias2",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(8,),
                                        vals=dnn_bias_params2.flatten())

    dnn_node2 = helper.make_node("Gemm",
                                    inputs=['in2','dnn_weight2','dnn_bias2'],
                                    outputs=['dnn_out2'],
                                    name="dnn_node2")
    
    add_node = onnx.helper.make_node('Add', inputs = ['dnn_out1','dnn_out2'], outputs = ['out'])

    in_shape1 = (1,8)
    in_shape2 = (1,8)
    out_shape = (1,8)

    nodes = [dnn_node1,dnn_node2,add_node]
    name = "add_shift"
    inputs = [helper.make_tensor_value_info("in1",onnx.TensorProto.FLOAT,list(in_shape1)),
                helper.make_tensor_value_info("in2",onnx.TensorProto.FLOAT,list(in_shape2))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [dnn_weight1,dnn_bias1,dnn_weight2,dnn_bias2]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name="add_shift")

    with open(root_dir + 'add_shift.onnx',"wb") as of:
        of.write(model.SerializeToString())

def test_mul_shift():
    generate_mul_shift_model()
    file_name = root_dir + './mul_shift.onnx'
    onnx_model = onnx.load(file_name)
    graph = onnx_model.graph
    node = graph.node

    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in1'] = (1, 8)
    shape_dict['in2'] = (1, 8)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data1 = np.round(np.random.rand(11, 8) * 255).astype("float32")
    data2 = np.round(np.random.rand(11, 8) * 255).astype("float32")
    input_dt = {}
    input_dt['in1'] = witin.nd.array(data1)
    input_dt['in2'] = witin.nd.array(data2)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        output_dir=build_dir,optimize_method_config="",
                                        chip="BB04P1")
    # ######################################################################
    # # create npu module with npu_graph and run graph
    # ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in1', witin.nd.array(data1))
    m.set_input('in2', witin.nd.array(data2))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list)
    
def test_add_shift():
    generate_add_shift_model()
    file_name = root_dir + './add_shift.onnx'
    onnx_model = onnx.load(file_name)
    graph = onnx_model.graph
    node = graph.node

    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in1'] = (1, 8)
    shape_dict['in2'] = (1, 8)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data = np.round(np.random.rand(11, 8) * 255).astype("float32")
    input_dt = {}
    input_dt['in1'] = witin.nd.array(data)
    input_dt['in2'] = witin.nd.array(data)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        output_dir=build_dir,optimize_method_config="",
                                        chip="BB04P1")
    # ######################################################################
    # # create npu module with npu_graph and run graph
    # ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in1', witin.nd.array(data))
    m.set_input('in2', witin.nd.array(data))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list)
if __name__ == '__main__':
    np.random.seed(10)
    test_mul_shift()
    test_add_shift()