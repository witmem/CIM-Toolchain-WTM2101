import numpy as np
import onnx
import os
import witin
import datetime
from witin import *


def test_forward_huawei_hz_e2e_3layer_1residual():
    chip = "BB04P1" # NB01PP,BB04P1
    model_path = "./model/pipeline/layer_residual_model/hz_e2e_3layer_1residual.onnx"
    target = 'npu'
    target_host = 'npu'

    shape_dict = {}
    shape_dict['fingerprint_input'] = (1, 39)

    onnx_model = onnx.load(model_path)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    input_dt = {}
    np.random.seed(10)
    data = np.random.rand(150, 39) * 255
    data = data.astype('float32')
    input_dt['fingerprint_input'] = witin.nd.array(data)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target=target,
                                                               target_host=target_host,
                                                               params=params,
                                                               input_data=input_dt,
                                                               output_dir=build_dir,
                                                               chip=chip)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    data = np.random.rand(150, 195) * 255
    data = data.astype('float32')
    m.set_input('fingerprint_input', witin.nd.array(data))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(1)]
    print(output[0])


if __name__ == '__main__':
    test_forward_huawei_hz_e2e_3layer_1residual()
