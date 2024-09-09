import numpy as np
import math
import onnx
from onnx import helper, TensorProto, mapping
import scipy
from tvm.relay.op.tensor import add
import witin
import datetime
from witin import *
root_dir = './model/'

def get_tvm_output(graph_def,in_shape,data):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, in_shape)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
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
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list)

def generate_scale_G_model():
    dnn_weight_params = np.random.randint(-100,100,size=(8,4),dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128,127,size=(4,),dtype=np.int32).astype(np.float32)

    dnn_weight = helper.make_tensor("dnn_weight",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,4),
                                            vals=dnn_weight_params.flatten())

    dnn_bias = helper.make_tensor("dnn_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(4,),
                                        vals=dnn_bias_params.flatten())

    dnn_node = helper.make_node("Gemm",
                                    inputs=['in','dnn_weight','dnn_bias'],
                                    outputs=['dnn_out'],
                                    name="dnn_node")
    # G值
    scale_node = helper.make_node('Scale', ['dnn_out'], ['scale_out'], scale=0.001953125)

    relu_node = helper.make_node('Relu', ['scale_out'], ['out'])


    in_shape = (1,8)
    out_shape = (1,4)

    nodes = [dnn_node,scale_node,relu_node]
    name = "scale_G"
    inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [dnn_weight,dnn_bias]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name="scale_G")

    with open(root_dir + 'scale_G.onnx',"wb") as of:
        of.write(model.SerializeToString())

def generate_ScalarOpNode_Mul_model():
    dnn_weight_params = np.random.randint(-100,100,size=(8,4),dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128,127,size=(4,),dtype=np.int32).astype(np.float32)

    dnn_weight = helper.make_tensor("dnn_weight",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,4),
                                            vals=dnn_weight_params.flatten())

    dnn_bias = helper.make_tensor("dnn_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(4,),
                                        vals=dnn_bias_params.flatten())

    dnn_node = helper.make_node("Gemm",
                                    inputs=['in','dnn_weight','dnn_bias'],
                                    outputs=['dnn_out'],
                                    name="dnn_node")

    # ScalarOpNode
    scale_node = helper.make_node('Scale', ['dnn_out'], ['scale_out'], scale=2, shift_bit=2, op_types="mul")

    relu_node = helper.make_node('Relu', ['scale_out'], ['out'])


    in_shape = (1,8)
    out_shape = (1,4)

    nodes = [dnn_node,scale_node,relu_node]
    name = "ScalarOpNode_Mul"
    inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [dnn_weight,dnn_bias]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name="ScalarOpNode_Mul")

    with open(root_dir + 'ScalarOpNode_Mul.onnx',"wb") as of:
        of.write(model.SerializeToString())

def generate_ScalarOpNode_Add_model():
    dnn_weight_params = np.random.randint(-100,100,size=(8,4),dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128,127,size=(4,),dtype=np.int32).astype(np.float32)

    dnn_weight = helper.make_tensor("dnn_weight",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,4),
                                            vals=dnn_weight_params.flatten())

    dnn_bias = helper.make_tensor("dnn_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(4,),
                                        vals=dnn_bias_params.flatten())

    dnn_node = helper.make_node("Gemm",
                                    inputs=['in','dnn_weight','dnn_bias'],
                                    outputs=['dnn_out'],
                                    name="dnn_node")

    # ScalarOpNode
    scale_node = helper.make_node('Scale', ['dnn_out'], ['scale_out'], scale=4, shift_bit=0, op_types="add")

    relu_node = helper.make_node('Relu', ['scale_out'], ['out'])


    in_shape = (1,8)
    out_shape = (1,4)

    nodes = [dnn_node,scale_node,relu_node]
    name = "ScalarOpNode_Add"
    inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [dnn_weight,dnn_bias]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name="ScalarOpNode_Add")

    with open(root_dir + 'ScalarOpNode_Add.onnx',"wb") as of:
        of.write(model.SerializeToString())

def generate_ScalarOpNode_model():
    dnn_weight_params = np.random.randint(-100,100,size=(8,4),dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128,127,size=(4,),dtype=np.int32).astype(np.float32)

    dnn_weight = helper.make_tensor("dnn_weight",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=(8,4),
                                            vals=dnn_weight_params.flatten())

    dnn_bias = helper.make_tensor("dnn_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(4,),
                                        vals=dnn_bias_params.flatten())

    dnn_node = helper.make_node("Gemm",
                                    inputs=['in','dnn_weight','dnn_bias'],
                                    outputs=['dnn_out'],
                                    name="dnn_node")

    # ScalarOpNode
    scale_node = helper.make_node('Scale', ['dnn_out'], ['scale_out'], scale=0.125)
    
    relu_node = helper.make_node('Relu', ['scale_out'], ['out'])


    in_shape = (1,8)
    out_shape = (1,4)

    nodes = [dnn_node,scale_node,relu_node]
    name = "ScalarOpNode"
    inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [dnn_weight,dnn_bias]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name="ScalarOpNode")

    with open(root_dir + 'ScalarOpNode.onnx',"wb") as of:
        of.write(model.SerializeToString())

def test_scale_G_model():
    generate_scale_G_model()
    file_name = root_dir + './scale_G.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, 8) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model,8,data)

def test_ScalarOpNode_model():
    generate_ScalarOpNode_model()
    file_name = root_dir + './ScalarOpNode.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, 8) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model,8,data)

def test_ScalarOpNode_Add_model():
    generate_ScalarOpNode_Add_model()
    file_name = root_dir + './ScalarOpNode_Add.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, 8) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model,8,data)

def test_ScalarOpNode_Mul_model():
    generate_ScalarOpNode_Mul_model()
    file_name = root_dir + './ScalarOpNode_Mul.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, 8) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model,8,data)

if __name__ == '__main__':
    np.random.seed(10)
    test_scale_G_model()
    test_ScalarOpNode_model()
    test_ScalarOpNode_Add_model()
    test_ScalarOpNode_Mul_model()
