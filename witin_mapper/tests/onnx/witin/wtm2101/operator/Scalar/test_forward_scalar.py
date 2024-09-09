import numpy as np
import math
import onnx
import os
from onnx import helper, TensorProto, mapping
import scipy
import sys
import witin
from witin import *

#### 改成自己路径
DIR_ROOT = "./tests/onnx/witin/wtm2101/operator/Scalar/"
optimize_config = DIR_ROOT+ "base.protobuf"

def get_tvm_output(onnx_path, quant_path):

    ## 名字与graph输入名称保持一致
    input_name = 'grin' 
    shape_dict = dict()
    shape_dict[input_name] = (1, 175)
    
    onnx_model = onnx.load(onnx_path)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    
    feats = np.load(quant_path).astype('float32')
    print("src shape:",feats.shape)

    input_dt = {}
    ## 名字与节点保持一致
    # input_dt['Conv_0'] = witin.nd.array(feats)
    # input_dt['Conv_4'] = witin.nd.array(feats)
    input_dt[input_name] = witin.nd.array(feats)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target='npu',
                                        target_host='npu',
                                        params=params, input_data=input_dt,
                                        chip="BB04P1",
                                        output_dir=build_dir,
                                        optimize_method_config=optimize_config
                                        )
    
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('grin', witin.nd.array(feats))
    m.run()

def test_forward_scalar():
    onnx_path = DIR_ROOT+'240219_c200_in175.onnx'
    quant_data_path = DIR_ROOT+"in175.npy"
    get_tvm_output(onnx_path,quant_data_path)

if __name__ == '__main__':
    test_forward_scalar()
