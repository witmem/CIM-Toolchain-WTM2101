from fileinput import filename
import numpy as np
import onnx
from onnx import helper, TensorProto, mapping
import datetime
import witin
from witin import *
#import keras
#from keras.models import load_model
#import pdb;pdb.set_trace()

root_dir = './model/'


def test_concatenate():
    np.random.seed(10)
    #filename = root_dir + './model.onnx'
    #filename = root_dir + './model_without_softmax.onnx'
    filename = root_dir + './huiche_model.onnx'
    onnx_model = onnx.load(filename)
    graph = onnx_model.graph
    node = graph.node

    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['spec_input'] = (1, 180)
    shape_dict['info_input'] = (1, 2)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)

    #data = load_model('./test_data.h5')
    #import pdb;pdb.set_trace()

    data1 = np.round(np.random.rand(11, 180) * 255).astype("float32")
    data2 = np.round(np.random.rand(11, 2) * 255).astype("float32")
    input_dt = {}
    # input_dt['input_data'] = witin.nd.array(data1)
    input_dt['spec_input'] = witin.nd.array(data1)
    # input_dt['MvOpNode1'] = witin.nd.array(data1)
    input_dt['info_input'] = witin.nd.array(data2)
    # input_dt['MvOpNode4'] = witin.nd.array(data2)
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
                                                               chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('spec_input', witin.nd.array(data1))
    m.set_input('info_input', witin.nd.array(data2))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]


if __name__ == '__main__':
    test_concatenate()
