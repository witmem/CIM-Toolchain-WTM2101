import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = "./tests/onnx/witin/wtm2101/operator/clearLowBits/"

def generate_model():
    G_scale = [1024, 2048]
    input_shape = [1, 60]
    kernel_shape_1 = [60, 40]
    kernel_shape_2 = [40, 20]
    bias_shape_1 = [40]
    bias_shape_2 = [20]
    output_shape = [1, 20]

    dnn1_weight_params = np.random.randint(-128, 127, size=kernel_shape_1, dtype=np.int32).astype('float32')
    dnn1_bias_params = 128 * np.random.randint(-128, 127, size=bias_shape_1, dtype=np.int32).astype('float32')
  
    dnn1_weight = onnx.helper.make_tensor("dnn1_weight", data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_1,
                                          vals=dnn1_weight_params.flatten())
  
    dnn1_bias = onnx.helper.make_tensor("dnn1_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_1,
                                        vals=dnn1_bias_params.flatten())
    # dnn1 node
    dnn1_node = onnx.helper.make_node("Gemm",
                                      inputs=['in', 'dnn1_weight', 'dnn1_bias'],
                                      outputs=['dnn1_out'],
                                      name="dnn1_node")
    dnn1_scale_node = onnx.helper.make_node('Scale', ['dnn1_out'], ['dnn1_scale_out'], scale=1.0 / G_scale[0])
    dnn1_relu_node = onnx.helper.make_node('Relu', ['dnn1_scale_out'], ['dnn1_relu_out'])
    dnn1_nodes = [dnn1_node, dnn1_scale_node, dnn1_relu_node]
    dnn1_initializers = [dnn1_weight, dnn1_bias]

    # clearLowBits node
    clearLowBits_node = onnx.helper.make_node("ClearLowBits", inputs=["dnn1_relu_out"], bits=3, outputs=["clearLowBits_out"])

    dnn2_weight_params = np.random.randint(-128, 127, size=kernel_shape_2, dtype=np.int32).astype('float32')
    dnn2_bias_params = 128 * np.random.randint(-128, 127, size=bias_shape_2, dtype=np.int32).astype('float32')

    dnn2_weight = onnx.helper.make_tensor("dnn2_weight", data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_2,
                                          vals=dnn2_weight_params.flatten())

    dnn2_bias = onnx.helper.make_tensor("dnn2_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_2,
                                        vals=dnn2_bias_params.flatten())
    # dnn2 node
    dnn2_node = onnx.helper.make_node("Gemm",
                                      inputs=['clearLowBits_out', 'dnn2_weight', 'dnn2_bias'],
                                      outputs=['dnn2_out'],
                                      name="dnn2_node")
    dnn2_scale_node = onnx.helper.make_node('Scale', ['dnn2_out'], ['dnn2_scale_out'], scale=1.0 / G_scale[0])
    dnn2_relu_node = onnx.helper.make_node('Relu', ['dnn2_scale_out'], ['out'])
    dnn2_nodes = [dnn2_node, dnn2_scale_node, dnn2_relu_node]
    dnn2_initializers = [dnn2_weight, dnn2_bias]

    in_shape = input_shape
    out_shape = output_shape

    nodes = dnn1_nodes + [clearLowBits_node] + dnn2_nodes
    name = "clearLowBits"
    inputs = [onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))]
    outputs = [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))]
    initializer = dnn1_initializers + dnn2_initializers

    graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer)
    model = onnx.helper.make_model(graph, producer_name="dccrn_gemm")
    with open(file_dir + 'test_gemm_clearLowBits.onnx', "wb") as f:
        f.write(model.SerializeToString())
    print("Generate {} sucessfully!".format('test_gemm_clearLowBits.onnx'))


def run_model():     
    input_shape = [1, 60]    
    cnn_model = onnx.load(file_dir + "test_gemm_clearLowBits.onnx")  
    data = np.round(np.random.rand(100, input_shape[1]) * 255).astype("float32")
    data = data.astype('float32')   
    shape_dict_conv = {}   
    shape_dict_conv['in'] = input_shape   
    mod, params = witin_frontend.frontend.from_onnx(cnn_model, shape_dict_conv)  
    input_dt = {}     
    input_dt['in'] = witin.nd.array(data)  
    dateTimeObj = datetime.datetime.now()  
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr         
    chip_type = "BB04P1"   
    with witin.transform.PassContext(opt_level=3):       
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,           
                                                               target='npu',                                                                target_host='npu',                                                                params=params,                                                                input_data=input_dt,                                                                chip=chip_type,                                                                output_dir=build_dir,                                 
                                                               optimize_method_config="")    
    from tvm.contrib import npu_graph_runtime   
    m = npu_graph_runtime.create(npu_graph, chip_type, output_dir=build_dir)  
    m.set_input('in', witin.nd.array(data)) 
    m.run() 
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]   
    print(witin_output_list[0].shape)

def test_clearLowBits(): 
    generate_model()   
    run_model()

if __name__ == "__main__":    
    test_clearLowBits()
