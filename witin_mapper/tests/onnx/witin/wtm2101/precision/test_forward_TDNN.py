import numpy as np
import math
import onnx
from onnx import helper, TensorProto, mapping
import scipy
import sys
import witin
from witin import *
import os

file_dir = './model/pipeline/tdnn_mdoel/'

def get_tvm_output(graph_def, config):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in_kws'] = (1, 40)
    # fifo_enable = os.getenv('key_name')
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)

    feats1 = np.loadtxt(os.path.join(file_dir,'tmp_cal.txt'), skiprows=70, max_rows=1020).astype("float32")

    input_dt = {}
    input_dt['input_data'] = witin.nd.array(feats1)

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
                                                               optimize_method_config=config)

    
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime

    data2 = np.round(np.random.rand(150, 200) * 255).astype("float32")
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data2))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy()
                            for i in range(1)]



def test_model(modelpath, config):
    file_name = os.path.join(modelpath, 'tdnn.onnx')
    onnx_model = onnx.load(file_name)

    get_tvm_output(onnx_model, config)


if __name__ == '__main__':
    
    config = os.path.join(file_dir,'opt.protobuf')     
    test_model(file_dir, config)
