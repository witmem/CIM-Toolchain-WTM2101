import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

def test_floatScale():
    params = []
    file_dir = "./tests/onnx/witin/wtm2101/operator/flotascale/"
    onnx_model = onnx.load(file_dir + "yxp_testC.onnx")
    data =  np.loadtxt(file_dir + 'hr_in_onnx.txt').astype('float32')

    shape_dict = {}
    shape_dict['in'] = [1, 387]  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)

    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    chip_type = "BB04P1"
    opt_config = file_dir + "opt.protobuf"
    # build
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=params,
                                                               input_data=input_dt,
                                                               chip=chip_type,
                                                               output_dir=build_dir,
                                                               optimize_method_config=opt_config)

    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, chip_type, output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    m.run()


if __name__ == "__main__":
   test_floatScale()	
