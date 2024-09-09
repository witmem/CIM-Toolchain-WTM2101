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

def generate_model():    
    
    mul_node = onnx.helper.make_node('Mul', inputs = ['in1','in2'], outputs = ['out'], shift_bit = 3)

    in_shape1 = (1,8)
    in_shape2 = (1,8)
    out_shape = (1,8)

    nodes = [mul_node]
    name = "directly_two_input_mul"
    inputs = [helper.make_tensor_value_info("in1",onnx.TensorProto.FLOAT,list(in_shape1)),
                helper.make_tensor_value_info("in2",onnx.TensorProto.FLOAT,list(in_shape2))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = []

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name="directly_two_input_mul")

    with open(root_dir + 'directly_two_input_mul.onnx',"wb") as of:
        of.write(model.SerializeToString())

def test_directly_two_input_mul():
    generate_model()
    file_name = root_dir + './directly_two_input_mul.onnx'
    onnx_model = onnx.load(file_name)

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
    

if __name__ == '__main__':
    np.random.seed(10)
    test_directly_two_input_mul()