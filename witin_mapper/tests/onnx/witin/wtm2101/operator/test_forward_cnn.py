import numpy as np
import onnx
from onnx import helper, TensorProto, mapping
import pytest
import os
import datetime
import witin
from witin import *
np.random.seed(100)

cnn_params = [
    # same HW、KHKW, stride, padding
    ((1,3,7,7), (3,3,3,3), (1,1), (0,0,0,0)),
    ((1,3,27,27), (3,3,5,5), (3,3), (0,2,2,0)),
    ((1,3,27,27), (32,3,7,7), (5,5), (2,3,3,2)),
    ((1,5,64,64), (16,5,13,13), (2,2), (0,5,5,0)),
    ((1,5,64,64), (16,5,8,8), (7,7), (0,0,0,7)),
    ((1,3,64,64), (32,3,15,15), (3,3), (2,3,3,2)),
    ((1,5,114,114), (32,5,13,13), (5,5), (2,7,7,2)),

    # # diff HW KW KH stride padding
    ((1,3,32,32),(16,3,5,3),(4,4),(1,1,0,2)),
    ((1,3,32,32),(16,3,3,5),(4,4),(2,0,1,1)),
    ((1,3,32,32),(16,3,3,5),(3,3),(0,0,1,0)),
    ((1,3,32,32),(16,3,3,5),(15,15),(1,0,0,3)),
    ((1,3,32,32),(16,3,3,5),(11,11),(1,3,3,3)),
    ((1,3,32,16),(16,3,3,3),(12,12),(1,5,6,6)),
    ((1,3,16,32),(16,3,3,3),(5,5),(1,0,1,1)),
    ((1,3,32,16),(16,3,3,3),(4,4),(1,2,2,1)),
    ((1,3,16,32),(16,3,3,3),(4,4),(2,1,1,2)),
    ((1,3,16,3),(16,3,3,3),(4,1),(3,0,0,0)),
    ((1,3,16,3),(16,3,5,3),(4,1),(0,0,1,0)),

    ((1,32,3,128),(56,32,3,7),(1,4),(0,3,0,0)),
    ((1,32,16,58),(32,32,4,5),(4,4),(0,2,0,1)),
    ((1,32,16,58),(32,32,4,5),(4,4),(2,2,2,1)),
    ((1,3,36,5),(16,3,5,5),(4,1),(1,0,0,0)),

    ((1,50,38,3),(3,50,5,3),(5,1),(2,0,0,0)),
    ((1,3,38,64),(3,3,5,3),(5,5),(1,1,1,3)),

    ((1,72,16,2),(112,72,5,2),(2,1),(1,0,2,0)),
    ((1,18,16,32),(112,18,5,2),(2,2),(0,0,1,0)),
    ((1,112,8,2),(112,112,3,2),(2,2),(0,0,1,0)),
    ((1,16,64,16),(1,16,7,7),(2,2),(0,1,1,0)),

    # when out_channel > 256
    # ((1,72,16,2),(256,72,5,2),(2,1),(0,0,2,0)),
    # ((1,18,16,32),(300,18,5,2),(2,2),(0,0,2,1)),
    # ((1,112,8,2),(512,112,3,2),(2,2),(0,0,1,0)),
    # ((1,16,64,16),(1024,16,7,7),(2,2),(0,1,1,0)),
]
def generate_cnn_model_from_params(cnn_input,weight,stride,padding):
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    cnn_weight_params = np.random.randint(-128,127,size=weight,dtype=np.int32).astype(np.float32)
    cnn_bias_params = 128 * np.random.randint(-127,127,size=(weight[0],),dtype=np.int32).astype(np.float32)

    cnn_weight = helper.make_tensor("cnn_weight", data_type=onnx.TensorProto.FLOAT, dims=weight, vals=cnn_weight_params.flatten())
    cnn_bias = helper.make_tensor("cnn_bias", data_type=onnx.TensorProto.FLOAT, dims=(weight[0],), vals=cnn_bias_params.flatten())

    cnn_conv = helper.make_node("Conv",
                                inputs = ['in','cnn_weight','cnn_bias'],
                                outputs = ['cnn_conv_out'],
                                strides = stride,
                                kernel_shape = [weight[2],weight[3]],
                                pads = padding,
                                name = 'cnn_node')

    cnn_scale = onnx.helper.make_node("Scale",
                                             inputs=['cnn_conv_out'],
                                             outputs=['cnn_scale_out'],
                                             scale=0.0009765625)
    cnn_relu = onnx.helper.make_node('Relu', ['cnn_scale_out'],
                                            ['out'])
    # in_shape = (1,56,64,3)
    # out_shape = (1,72,16,1)
    in_shape = cnn_input
    o_shape_h = (cnn_input[2] - weight[2] + padding[0] + padding[2]) / stride[0] + 1
    o_shape_w = (cnn_input[3] - weight[3] + padding[1] + padding[3]) / stride[1] + 1
    out_shape = (cnn_input[0],weight[0],int(o_shape_h),int(o_shape_w))

    nodes = [cnn_conv,cnn_scale,cnn_relu]
    name = "cnn_case"
    inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [cnn_weight,cnn_bias]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name='cnn')

    model_path = "./model/cnn" + timestampStr + ".onnx"
    with open(model_path,"wb") as of:
        of.write(model.SerializeToString())

    print("generate cnn.onnx sucessfully!")
    return model_path

@pytest.mark.parametrize("cnn_input,weight,stride,padding", cnn_params)
def test_cnn_model_from_params(cnn_input,weight,stride,padding):
    filename = generate_cnn_model_from_params(cnn_input,weight,stride,padding)
    data = np.round(np.random.rand(11,cnn_input[3],cnn_input[2],cnn_input[1])*255).astype("float32")
    onnx_model = onnx.load(filename)
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = cnn_input
    mod,params = witin_frontend.frontend.from_onnx(onnx_model,shape_dict)
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data)
    # opt_config = './model/optimize_layer/optimize_config_sparse_matrix_digital_cnn.protobuf'
    opt_config = ""
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _,_,_,npu_graph = witin_frontend.build_module.build(mod,
                                                            target=target,
                                                            target_host=target_host,
                                                            params=params,
                                                            input_data=input_dt,
                                                            chip='BB04P1',
                                                            output_dir=build_dir,
                                                            optimize_method_config=opt_config)
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1",output_dir=build_dir)
    m.set_input('in1', witin.nd.array(data))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]


def generate_cnn_model():
    cnn_weight_params = np.random.randint(-128,127,size=(512,16,5,5),dtype=np.int32).astype(np.float32)
    cnn_bias_params = 128 * np.random.randint(-127,127,size=(512,),dtype=np.int32).astype(np.float32)

    cnn_weight = helper.make_tensor("cnn_weight", data_type=onnx.TensorProto.FLOAT, dims=(512,16,5,5), vals=cnn_weight_params.flatten())
    cnn_bias = helper.make_tensor("cnn_bias", data_type=onnx.TensorProto.FLOAT, dims=(512,), vals=cnn_bias_params.flatten())

    cnn_conv = helper.make_node("Conv",
                                inputs = ['in','cnn_weight','cnn_bias'],
                                outputs = ['cnn_conv_out'],
                                strides = [5,5],
                                kernel_shape = [5,5],
                                pads = [2,1,1,2],
                                name = 'cnn_node')

    cnn_scale = onnx.helper.make_node("Scale",
                                             inputs=['cnn_conv_out'],
                                             outputs=['cnn_scale_out'],
                                             scale=0.0009765625)
    cnn_relu = onnx.helper.make_node('Relu', ['cnn_scale_out'],
                                            ['out'])
    in_shape = (1,16,38,38)
    out_shape = (1,512,1,1)

    nodes = [cnn_conv,cnn_scale,cnn_relu]
    name = "cnn_case"
    inputs = [helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]
    outputs = [helper.make_tensor_value_info("out",onnx.TensorProto.FLOAT,list(out_shape))]
    initializer = [cnn_weight,cnn_bias]

    graph = helper.make_graph(nodes,name,inputs,outputs,initializer)
    model = helper.make_model(graph,producer_name='cnn')

    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")

    model_path = './model/cnn_ + ' + 'timestamp' + '.onnx'
    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate cnn.onnx sucessfully!")
    return model_path


def generate_cnn_model2(shape):
    cnn_weight_params = np.random.randint(-128, 127, size=(128, 16, 3, 3), dtype=np.int32).astype(np.float32)
    cnn_bias_params = 128 * np.random.randint(-127, 127, size=(128,), dtype=np.int32).astype(np.float32)

    cnn_weight = helper.make_tensor("cnn_weight",
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=(128, 16, 3, 3),
                                    vals=cnn_weight_params.flatten())
    cnn_bias = helper.make_tensor("cnn_bias",
                                  data_type=onnx.TensorProto.FLOAT,
                                  dims=(128,),
                                  vals=cnn_bias_params.flatten())

    cnn_conv = helper.make_node("Conv",
                                inputs=['in', 'cnn_weight', 'cnn_bias'],
                                outputs=['cnn_conv_out'],
                                strides=[1, 1],
                                kernel_shape=[3, 3],
                                pads=[0, 0, 0, 0],
                                name='cnn_node')

    cnn_scale = onnx.helper.make_node("Scale", inputs=['cnn_conv_out'], outputs=['cnn_scale_out'], scale=0.0009765625)
    cnn_relu = onnx.helper.make_node('Relu', ['cnn_scale_out'], ['out'])
    in_shape = (1, 16, shape, shape)
    out_shape = (1, 128, 8, 8)

    nodes = [cnn_conv, cnn_scale, cnn_relu]
    name = "cnn_case"
    inputs = [helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))]
    outputs = [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))]
    initializer = [cnn_weight, cnn_bias]

    graph = helper.make_graph(nodes, name, inputs, outputs, initializer)
    model = helper.make_model(graph, producer_name='cnn')

    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")

    model_path = './model/cnn_ + ' + 'timestamp' + '.onnx'
    with open(model_path, "wb") as of:
        of.write(model.SerializeToString())

    print("generate cnn.onnx sucessfully!")
    return model_path

def distance(a, b):
    v1 = np.sqrt(np.sum((np.int32(b) - np.int32(a))**2))
    v2 = np.sqrt(np.sum(1e-5 + np.int32(b)**2))
    v3 = v1 / v2
    ret = np.sum(v3)
    # print("compare distance is:%.4f"%(ret))
    return ret


def build_run_model(mod, params, optimize_method_config, data, base_out):
    target = 'npu'
    target_host = 'npu'
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target=target,
                                                               target_host=target_host,
                                                               params=params,
                                                               input_data=input_dt,
                                                               chip='BB04P1',
                                                               output_dir=build_dir,
                                                               optimize_method_config=optimize_method_config)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(1)]
    # print(output[0])
    if base_out is not None:
        ret = distance(base_out, output[0])
        if 1 - ret < 0.95:
            print(base_out)
            print(output[0])
            raise ValueError("similarity is  '%f' " % (ret))
        else:
            print("output same as baseline")

    return output[0]


def test_cnn_model_dw():
    base_out = None
    file_name = generate_cnn_model()
    onnx_model = onnx.load(file_name)
    shape_dict = {}
    shape_dict['in'] = (1, 16, 38, 38)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    opt_config = [ "", "./model/optimize_layer/test_forward_cnn_dw.protobuf"]
    data = np.round(np.random.rand(11, 38, 38, 16) * 255).astype("float32")  #NHWC
    for opt in opt_config:
        if (opt != ""):
            data = np.concatenate((data, data), 3)
        base_out = build_run_model(mod, params, opt, data, base_out)


def test_cnn_model_pn():
    base_out = None
    shape = 10
    file_name = generate_cnn_model2(shape)
    onnx_model = onnx.load(file_name)
    shape_dict = {}
    shape_dict['in'] = (1, 16, shape, shape)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    opt_config = "./model/optimize_layer/test_forward_cnn_pn.protobuf"
    data = np.round(np.random.rand(11, shape, shape, 16) * 255).astype("float32") - 128  #NHWC
    

    build_run_model(mod, params, opt_config, data, base_out)


def test_cnn_model_opt():
    test_cnn_model_dw()
    test_cnn_model_pn()

if __name__ == '__main__':
    test_cnn_model_dw()
    test_cnn_model_pn()
