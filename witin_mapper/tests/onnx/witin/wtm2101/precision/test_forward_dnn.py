import onnx
import numpy as np
import witin
from witin import *
import os

np.random.seed(0)

def run_model(file_dir, protobuf_dir, protobuf_name):
    graphs = []
    params = []
    input_datas = []
    conv_model = onnx.load(file_dir + "dnn.onnx")

    data = np.load(file_dir + 'dnn_input0.npy').astype('float32')
    data_shape = data.shape             
    data = data.astype('float32')  
    print(data.shape)
    data = np.clip(data, 0, 255)

    shape_dict_dnn = {}
    shape_dict_dnn['in'] = (1, 800)       
    mod, param = witin_frontend.frontend.from_onnx(conv_model, shape_dict_dnn)

    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data)

    opt_config = protobuf_dir + protobuf_name + '.protobuf'

    # output file
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    # build
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
    m.set_input('in', witin.nd.array(data))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)


if __name__ == "__main__":
    file_dir = './model/pipeline/dnn_model/'          
    protobuf_dir = './model/pipeline/dnn_model/'      
    protobuf_name = 'opt'
    run_model(file_dir, protobuf_dir, protobuf_name)
