import onnx
import numpy as np
import witin
from witin import *
import os

np.random.seed(0)

onnx_dir = './tests/onnx/witin/wtm2101/operator/conv_outchannel_split/'
if not os.path.exists(onnx_dir):
    os.mkdir(onnx_dir)

def test_honor_conv():
    mods = []
    params = []
    input_datas = []
    model = onnx.load(onnx_dir + "kws_model.onnx")
    shape_dict = {}
    shape_dict['input'] = (1, 40, 9, 1)  #NCHW

    mod, param = witin_frontend.frontend.from_onnx(model, shape_dict)
    data =  np.random.randint(0, 127, size=(20, 40, 9, 1), dtype=np.int32).astype(np.float32)
    data = np.transpose(data,(0,3,2,1)) #NWHC
    input_dt = {}
    input_dt['input'] = witin.nd.array(data)

    opt_config = onnx_dir + 'kws_opti.protobuf'
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    #build
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=param,
                                                               input_data=input_dt,
                                                               chip='BB04P1',
                                                               output_dir=build_dir,
                                                               optimize_method_config=opt_config)
    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)

    m.set_input(0, witin.nd.array(data))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    print(witin_output_list[0].shape)


if __name__ == "__main__":
    test_honor_conv()