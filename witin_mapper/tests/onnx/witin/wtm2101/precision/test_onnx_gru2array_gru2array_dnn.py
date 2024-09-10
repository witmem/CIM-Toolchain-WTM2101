
    



import numpy as np
import onnx
import os
import datetime
import witin
from witin import *
np.random.seed(10)

root_dir = './model/pipeline/gru2array_model/' 
      
def run_model():
    input_len = 50
    file_name = root_dir + 'gru2array_gru2array_dnn_network.onnx'
    onnx_model = onnx.load(file_name)
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, input_len)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    # print(mod)
    # data1 = np.round(np.random.rand(111, input_len) * 255).astype("float32")-128
    data1 = np.round(np.random.rand(111, input_len) * 127).astype("float32")
    print(data1[:10,:])
    input_dt = {}
    input_dt['in'] = witin.nd.array(data1)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    opti = root_dir + '/gru_gru_dnn.protobuf'
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

def test_forward_gru2array_gru2array_dnn():
    
    # generate_gru_model()
    run_model()


if __name__ == '__main__':
    test_forward_gru2array_gru2array_dnn()
    
