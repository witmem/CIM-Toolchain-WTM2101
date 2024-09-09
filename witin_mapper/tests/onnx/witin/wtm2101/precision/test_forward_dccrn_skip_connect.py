import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

def run_model():
    params = []
    conv_model = onnx.load("./model/pipeline/dccrn_skip_connect_model/test_dccrn_11-7.onnx")

    data = np.load('./model/pipeline/dccrn_skip_connect_model/input1.npy').astype('float32')
    data = data.astype('float32') # [1, 2, 256, 101]
    data = np.transpose(data, (0, 3, 2, 1)) # [1, 101, 256, 4]
    data = data.reshape(data.shape[0] * data.shape[1], 1, data.shape[2], data.shape[3]) # [101, 1, 256, 4]

    # merge the adjacent 2 frame data, [101, 1, 256, 4] -> [100, 2, 256, 4]
    new_data = np.zeros([data.shape[0] - 1, data.shape[1] * 2, data.shape[2], data.shape[3]], dtype=data.dtype) 
    for n in range(data.shape[0] - 1):
      new_data[n, 0, :, :] = data[n, :, :, :]
      new_data[n, 1, :, :] = data[n + 1, :, :, :]

    shape_dict_conv = {}
    shape_dict_conv['in'] = (1, 2, 256, 101)  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(conv_model, shape_dict_conv)
    input_dt = {}
    input_dt['in'] = witin.nd.array(new_data)
    opt_config = './model/pipeline/dccrn_skip_connect_model/dccrn_opti.protobuf'
    #output file
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    #build
    with witin.transform.PassContext(opt_level=3):
        _,_,_,npu_graph = witin_frontend.build_module.build(mod,
                                                            target='npu',
                                                            target_host='npu',
                                                            params=params,
                                                            input_data=input_dt,
                                                            chip='BB04P1',
                                                            output_dir=build_dir,
                                                            optimize_method_config=opt_config)
    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(new_data))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)
    
def test_skip_connect_dccrn_commercial():
    run_model()

if __name__ == "__main__":
    test_skip_connect_dccrn_commercial()
