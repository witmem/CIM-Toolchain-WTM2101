import onnx
import numpy as np
import witin
from witin import *
import os
np.random.seed(0)

file_dir = "./model/"

def generate_const_input_model():
    G_scale = 1024
    input_shape = [1, 100]
    kernel_shape_1 = [100, 240]
    bias_shape_1 = [240]
    reshape_tensor_shape1 = [10, 24]
    matmul_const_input_shape = [24, 36]
    reshape_tensor_shape2 = [1, 360]
    kernel_shape_2 = [360, 50]
    bias_shape_2 = [50]
    output_shape = [1, 50]
    # gemm1
    dnn1_weight_params = np.random.randint(-100, 100, size=kernel_shape_1,
                                          dtype=np.int32).astype('float32')
    dnn1_bias_params = 128 * np.random.randint(-127, 127, size=bias_shape_1,
                                              dtype=np.int32).astype('float32')
    dnn1_weight = onnx.helper.make_tensor("dnn1_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_1,
                                          vals=dnn1_weight_params.flatten())
    dnn1_bias = onnx.helper.make_tensor("dnn1_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_1,
                                        vals=dnn1_bias_params.flatten())
    dnn1_node = onnx.helper.make_node("Gemm",
                                     inputs=['in','dnn1_weight','dnn1_bias'],
                                     outputs=['dnn1_out'],
                                     name="dnn1_node")

    scale1_node = onnx.helper.make_node('Scale', ['dnn1_out'], ['scale_out_1'], scale=1.0 / G_scale)
    relu1_node = onnx.helper.make_node('Relu', ['scale_out_1'], ['dnn1_relu_out'])
    # reshape1
    reshape1_tensor = onnx.helper.make_tensor("reshape1_tensor",
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=(len(reshape_tensor_shape1),),
                                               vals=reshape_tensor_shape1)
    reshape1_node = onnx.helper.make_node("Reshape", inputs=["dnn1_relu_out", "reshape1_tensor"],
                                          outputs=["reshape1_out"])
    #matmul
    matmul_weight_params = np.random.randint(-100, 100, size=matmul_const_input_shape,
                                             dtype=np.int32).astype('float32')
    matmul_weight = onnx.helper.make_tensor("matmul_weight",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=matmul_const_input_shape,
                                            vals=matmul_weight_params.flatten())
    matmul_node = onnx.helper.make_node("MatMul",
                                        inputs=['reshape1_out', 'matmul_weight'],
                                        outputs=['matmul1_out'],
                                        name="matmul1")
    scale2_node = onnx.helper.make_node('Scale', ['matmul1_out'], ['scale_out_2'], scale=1.0 / G_scale)
    relu2_node = onnx.helper.make_node('Relu', ['scale_out_2'], ['matmul_relu_out'])
    #reshape2
    reshape2_tensor = onnx.helper.make_tensor("reshape2_tensor",
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=(len(reshape_tensor_shape2),),
                                              vals=reshape_tensor_shape2)
    reshape2_node = onnx.helper.make_node("Reshape", inputs=["matmul_relu_out", "reshape2_tensor"],
                                          outputs=["reshape2_out"])
    #gemm2
    dnn2_weight_params = np.random.randint(-100, 100, size=kernel_shape_2,
                                           dtype=np.int32).astype('float32')
    dnn2_bias_params = 128 * np.random.randint(-127, 127, size=bias_shape_2,
                                               dtype=np.int32).astype('float32')
    dnn2_weight = onnx.helper.make_tensor("dnn2_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_2,
                                          vals=dnn2_weight_params.flatten())
    dnn2_bias = onnx.helper.make_tensor("dnn2_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_2,
                                        vals=dnn2_bias_params.flatten())
    dnn2_node = onnx.helper.make_node("Gemm",
                                      inputs=['reshape2_out','dnn2_weight','dnn2_bias'],
                                      outputs=['dnn2_out'],
                                      name="dnn2_node")
    
    scale3_node = onnx.helper.make_node('Scale', ['dnn2_out'], ['scale_out_3'], scale=1.0 / G_scale)
    relu3_node = onnx.helper.make_node('Relu', ['scale_out_3'], ['out'])

    in_shape = input_shape
    out_shape = output_shape

    nodes = [dnn1_node, scale1_node, relu1_node] + [reshape1_node] + [matmul_node, scale2_node, relu2_node] + \
            [reshape2_node] + [dnn2_node, scale3_node, relu3_node]
    initializer = [dnn1_weight, dnn1_bias] + [reshape1_tensor] + [matmul_weight] + [reshape2_tensor] + [dnn2_weight, dnn2_bias] 
    inputs = [onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))]
    outputs = [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))]
    
    graph = onnx.helper.make_graph(nodes, "test_matmul", inputs, outputs, initializer)
    model = onnx.helper.make_model(graph, producer_name="test_matmul")
    with open(file_dir + 'test_mamul_input_const.onnx', "wb") as f:
        f.write(model.SerializeToString())
    print("Generate {} sucessfully!".format('test_mamul_input_const.onnx'))

def run_const_input_model():
    input_shape = [1, 100]
    params = []
    gemm_model = onnx.load(file_dir + "test_mamul_input_const.onnx")
    data = np.round(np.random.rand(11, input_shape[1]) * 255).astype("float32")
    
    shape_dict_conv = {}
    shape_dict_conv['in'] = input_shape  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(gemm_model, shape_dict_conv)

    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    chip_type = "BB04P1"
    # build
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=params,
                                                               input_data=input_dt,
                                                               chip=chip_type,
                                                               output_dir=build_dir,
                                                               optimize_method_config="")

    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, chip_type, output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    m.run()
    
def generate_variable_input_model():
    G_scale = 1024
    input_shape = [1, 50]
    kernel_shape_0 = [50, 100]
    bias_shape_0 = [100]
    kernel_shape_1 = [100, 60]
    bias_shape_1 = [60]
    reshape_tensor_shape1 = [10, 6]
    kernel_shape_2 = [100, 48]
    bias_shape_2 = [48]
    reshape_tensor_shape2 = [6, 8]
    reshape_tensor_shape3 = [1, 80]
    kernel_shape_3 = [80, 60]
    bias_shape_3 = [60]
    output_shape = [1, 60]
    
    # gemm0
    dnn0_weight_params = np.random.randint(-100, 100, size=kernel_shape_0,
                                          dtype=np.int32).astype('float32')
    dnn0_bias_params = 128 * np.random.randint(-127, 127, size=bias_shape_0,
                                              dtype=np.int32).astype('float32')
    dnn0_weight = onnx.helper.make_tensor("dnn0_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_0,
                                          vals=dnn0_weight_params.flatten())
    dnn0_bias = onnx.helper.make_tensor("dnn0_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_0,
                                        vals=dnn0_bias_params.flatten())
    dnn0_node = onnx.helper.make_node("Gemm",
                                      inputs=['in','dnn0_weight','dnn0_bias'],
                                      outputs=['dnn0_out'],
                                      name="dnn0_node")

    scale0_node = onnx.helper.make_node('Scale', ['dnn0_out'], ['scale_out_0'], scale=1.0 / G_scale)
    relu0_node = onnx.helper.make_node('Relu', ['scale_out_0'], ['dnn0_relu_out'])
    
    # gemm1
    dnn1_weight_params = np.random.randint(-100, 100, size=kernel_shape_1,
                                           dtype=np.int32).astype('float32')
    dnn1_bias_params = 128 * np.random.randint(-127, 127, size=bias_shape_1,
                                              dtype=np.int32).astype('float32')
    dnn1_weight = onnx.helper.make_tensor("dnn1_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_1,
                                          vals=dnn1_weight_params.flatten())
    dnn1_bias = onnx.helper.make_tensor("dnn1_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_1,
                                        vals=dnn1_bias_params.flatten())
    dnn1_node = onnx.helper.make_node("Gemm",
                                     inputs=['dnn0_relu_out','dnn1_weight','dnn1_bias'],
                                     outputs=['dnn1_out'],
                                     name="dnn1_node")
    scale1_node = onnx.helper.make_node('Scale', ['dnn1_out'], ['scale_out_1'], scale=1.0 / G_scale)
    relu1_node = onnx.helper.make_node('Relu', ['scale_out_1'], ['dnn1_relu_out'])

    # reshape1
    reshape1_tensor = onnx.helper.make_tensor("reshape1_tensor",
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=(len(reshape_tensor_shape1),),
                                               vals=reshape_tensor_shape1)
    reshape1_node = onnx.helper.make_node("Reshape", inputs=["dnn1_relu_out", "reshape1_tensor"],
                                          outputs=["reshape1_out"])
    # gemm2
    dnn2_weight_params = np.random.randint(-100, 100, size=kernel_shape_2,
                                           dtype=np.int32).astype('float32')
    dnn2_bias_params = 128 * np.random.randint(-127, 127, size=bias_shape_2,
                                              dtype=np.int32).astype('float32')
    dnn2_weight = onnx.helper.make_tensor("dnn2_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_2,
                                          vals=dnn2_weight_params.flatten())
    dnn2_bias = onnx.helper.make_tensor("dnn2_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_2,
                                        vals=dnn2_bias_params.flatten())
    dnn2_node = onnx.helper.make_node("Gemm",
                                      inputs=['dnn0_relu_out','dnn2_weight','dnn2_bias'],
                                      outputs=['dnn2_out'],
                                      name="dnn2_node")
    scale2_node = onnx.helper.make_node('Scale', ['dnn2_out'], ['scale_out_2'], scale=1.0 / G_scale)
    relu2_node = onnx.helper.make_node('Relu', ['scale_out_2'], ['dnn2_relu_out'])

    # reshape2
    reshape2_tensor = onnx.helper.make_tensor("reshape2_tensor",
                                               data_type=onnx.TensorProto.FLOAT,
                                               dims=(len(reshape_tensor_shape2),),
                                               vals=reshape_tensor_shape2)
    reshape2_node = onnx.helper.make_node("Reshape", inputs=["dnn2_relu_out", "reshape2_tensor"],
                                          outputs=["reshape2_out"])
    
    # matmul
    matmul_node = onnx.helper.make_node("MatMul",
                                        inputs=['reshape1_out', 'reshape2_out'],
                                        outputs=['matmul1_out'],
                                        name="matmul1")
    relu3_node = onnx.helper.make_node('Relu', ['matmul1_out'], ['matmul_relu_out'])
    # reshape2
    reshape3_tensor = onnx.helper.make_tensor("reshape3_tensor",
                                              data_type=onnx.TensorProto.FLOAT,
                                              dims=(len(reshape_tensor_shape3),),
                                              vals=reshape_tensor_shape3)
    reshape3_node = onnx.helper.make_node("Reshape", inputs=["matmul_relu_out", "reshape3_tensor"],
                                          outputs=["reshape3_out"])
    # gemm3
    dnn3_weight_params = np.random.randint(-100, 100, size=kernel_shape_3,
                                           dtype=np.int32).astype('float32')
    dnn3_bias_params = 128 * np.random.randint(-127, 127, size=bias_shape_3,
                                               dtype=np.int32).astype('float32')
    dnn3_weight = onnx.helper.make_tensor("dnn3_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=kernel_shape_3,
                                          vals=dnn3_weight_params.flatten())
    dnn3_bias = onnx.helper.make_tensor("dnn3_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=bias_shape_3,
                                        vals=dnn3_bias_params.flatten())
    dnn3_node = onnx.helper.make_node("Gemm",
                                      inputs=['reshape3_out','dnn3_weight','dnn3_bias'],
                                      outputs=['dnn3_out'],
                                      name="dnn3_node")
    
    scale4_node = onnx.helper.make_node('Scale', ['dnn3_out'], ['scale_out_4'], scale=1.0 / G_scale)
    relu4_node = onnx.helper.make_node('Relu', ['scale_out_4'], ['out'])

    in_shape = input_shape
    out_shape = output_shape

    nodes = [dnn0_node, scale0_node, relu0_node] + [dnn1_node, scale1_node, relu1_node] + [reshape1_node] + [dnn2_node, scale2_node, relu2_node] + \
            [reshape2_node] + [matmul_node, relu3_node] + [reshape3_node] + [dnn3_node, scale4_node, relu4_node]
    initializer = [dnn0_weight, dnn0_bias] + [dnn1_weight, dnn1_bias] + [reshape1_tensor] + [dnn2_weight, dnn2_bias] + \
                  [reshape2_tensor] + [reshape3_tensor] + [dnn3_weight, dnn3_bias]
    inputs = [onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))]
    outputs = [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))]
    
    graph = onnx.helper.make_graph(nodes, "test_matmul", inputs, outputs, initializer)
    model = onnx.helper.make_model(graph, producer_name="test_matmul")
    with open(file_dir + 'test_mamul_input_variable.onnx', "wb") as f:
        f.write(model.SerializeToString())
    print("Generate {} sucessfully!".format('test_mamul_input_variable.onnx'))


def run_variable_input_model():
    input_shape = [1, 50]
    params = []
    gemm_model = onnx.load(file_dir + "test_mamul_input_variable.onnx")
    data = np.round(np.random.rand(11, input_shape[1]) * 255).astype("float32")
    
    shape_dict_conv = {}
    shape_dict_conv['in'] = input_shape  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(gemm_model, shape_dict_conv)

    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f") 
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    chip_type = "BB04P1"
    # build
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=params,
                                                               input_data=input_dt,
                                                               chip=chip_type,
                                                               output_dir=build_dir,
                                                               optimize_method_config="")

    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, chip_type, output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    m.run()

def test_matmul_const_input():
    generate_const_input_model()
    run_const_input_model()
 
def test_matmul_variable_input():
    generate_variable_input_model()
    run_variable_input_model()

if __name__ == "__main__":
    test_matmul_const_input()
