# Copyright(c) 2022 Witin Tech Corporation. All Rights Reserved.
# Author: shihang.fu
# Email: shihang.fu@witintech.com
# Last modified: 2022-04-07
# Filename: test_forward_fifo_en_dw.py
# Description:
import numpy as np
import onnx
from onnx import helper
import os
import datetime
import witin
from witin import *


def generate_tdnn_model():
    tdnn_weight_0 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_1 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_2 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_3 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_4 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_6 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    tdnn_weight_7 = np.random.randint(-128, 127, size=(120, 40), dtype=np.int32).astype(np.float32)
    gemm_params_1 = np.random.randint(-128, 127, size=(40, 40), dtype=np.int32).astype(np.float32)
    gemm_params_2 = np.random.randint(-128, 127, size=(40, 40), dtype=np.int32).astype(np.float32)
    gemm_params_0 = np.random.randint(-128, 127, size=(40, 40), dtype=np.int32).astype(np.float32)

    tdnn_bias_0 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_1 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_2 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_3 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_5 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    tdnn_bias_7 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    gemm_bias_1 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    gemm_bias_2 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    gemm_bias_0 = 128 * np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)

    out_dim = tdnn_weight_7.shape[1]
    offsets_layer1 = onnx.helper.make_tensor("layer1_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    offsets_layer2 = onnx.helper.make_tensor("layer2_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    offsets_layer3 = onnx.helper.make_tensor("layer3_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    offsets_layer4 = onnx.helper.make_tensor("layer4_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    offsets_layer5 = onnx.helper.make_tensor("layer5_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    offsets_layer6 = onnx.helper.make_tensor("layer6_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    offsets_layer7 = onnx.helper.make_tensor("layer7_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    offsets_layer8 = onnx.helper.make_tensor("layer8_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])

    linear_params1 = onnx.helper.make_tensor("layer1_params",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(120, 40),
                                             vals=tdnn_weight_0.flatten())
    linear_params2 = onnx.helper.make_tensor("layer2_params",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(120, 40),
                                             vals=tdnn_weight_1.flatten())
    linear_params3 = onnx.helper.make_tensor("layer3_params",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(120, 40),
                                             vals=tdnn_weight_2.flatten())
    linear_params4 = onnx.helper.make_tensor("layer4_params",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(120, 40),
                                             vals=tdnn_weight_3.flatten())
    linear_params8 = onnx.helper.make_tensor("layer8_params",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(120, 40),
                                             vals=tdnn_weight_7.flatten())
    gemm_params1 = onnx.helper.make_tensor("gemm_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(40, 40),
                                           vals=gemm_params_1.flatten())
    gemm_params2 = onnx.helper.make_tensor("gemm_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(40, 40),
                                           vals=gemm_params_2.flatten())

    gemm_params = onnx.helper.make_tensor("gemm_params",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(40, 40),
                                          vals=gemm_params_0.flatten())

    linear_bias1 = onnx.helper.make_tensor("layer1_bias",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(40,),
                                           vals=tdnn_bias_0.flatten())
    linear_bias2 = onnx.helper.make_tensor("layer2_bias",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(40,),
                                           vals=tdnn_bias_1.flatten())
    linear_bias3 = onnx.helper.make_tensor("layer3_bias",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(40,),
                                           vals=tdnn_bias_2.flatten())
    linear_bias4 = onnx.helper.make_tensor("layer4_bias",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(40,),
                                           vals=tdnn_bias_3.flatten())
    linear_bias8 = onnx.helper.make_tensor("layer8_bias",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(40,),
                                           vals=tdnn_bias_7.flatten())
    gemm_bias1 = onnx.helper.make_tensor("gemm_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(40,),
                                         vals=gemm_bias_1.flatten())

    gemm_bias2 = onnx.helper.make_tensor("gemm_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(40,),
                                         vals=gemm_bias_2.flatten())

    gemm_bias = onnx.helper.make_tensor("gemm_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(40,),
                                        vals=gemm_bias_0.flatten())

    node1 = onnx.helper.make_node('Tdnn',
                                  inputs=['in', 'layer1_params'],
                                  outputs=['tdnn1'],
                                  time_offsets=offsets_layer1,
                                  bias_params=linear_bias1,
                                  scale_params=1024,
                                  name="tdnn1")

    relu_node1 = onnx.helper.make_node('Relu', inputs=['tdnn1'], outputs=['relu_out1'])

    node2 = onnx.helper.make_node('Tdnn',
                                  inputs=['relu_out1', 'layer2_params'],
                                  outputs=['tdnn2'],
                                  time_offsets=offsets_layer2,
                                  bias_params=linear_bias2,
                                  scale_params=1024)

    relu_node3 = onnx.helper.make_node('Relu', ['tdnn2'], ['relu_out3'])

    gemm_node1 = onnx.helper.make_node('Gemm',
                                       inputs=['relu_out3', 'gemm_params1', 'gemm_bias1'],
                                       outputs=['gemm_out1'])
    gemm_mul_node1 = onnx.helper.make_node('Scale', ['gemm_out1'], ['gemm_mul_node_out1'],
                                           scale=0.0009765625)

    gemm_relu_node1 = onnx.helper.make_node('Relu', ['gemm_mul_node_out1'], ['gemm_relu_out1'])

    node4 = onnx.helper.make_node('Tdnn',
                                  inputs=['gemm_relu_out1', 'layer4_params'],
                                  outputs=['tdnn4'],
                                  time_offsets=offsets_layer4,
                                  bias_params=linear_bias4,
                                  scale_params=1024)

    relu_node4 = onnx.helper.make_node('Relu', ['tdnn4'], ['relu_out4'])

    gemm_node2 = onnx.helper.make_node('Gemm',
                                       inputs=['relu_out4', 'gemm_params2', 'gemm_bias2'],
                                       outputs=['gemm_out2'])
    gemm_mul_node2 = onnx.helper.make_node('Scale', ['gemm_out2'], ['gemm_mul_node_out2'],
                                           scale=0.0009765625)

    gemm_relu_node2 = onnx.helper.make_node('Relu', ['gemm_mul_node_out2'], ['gemm_relu_out2'])

    add_node = onnx.helper.make_node('Add', ['relu_out3', 'gemm_relu_out2'], ['add_out'])

    y = np.random.randint(-128, 127, size=(40,), dtype=np.int32).astype(np.float32)
    y_t = onnx.helper.make_tensor("y_t_bias",
                                  data_type=onnx.TensorProto.FLOAT,
                                  dims=(40,),
                                  vals=y.flatten())

    mul_node = onnx.helper.make_node('Mul', ['add_out', 'y_t_bias'], ['mul_node'])

    relu_node7 = onnx.helper.make_node('Relu', ['mul_node'], ['relu_out7'])

    gemm_node = onnx.helper.make_node('Gemm',
                                      inputs=['relu_out7', 'gemm_params', 'gemm_bias'],
                                      outputs=['gemm_out'])
    gemm_mul_node = onnx.helper.make_node('Scale', ['gemm_out'], ['gemm_mul_node_out'],
                                          scale=0.0009765625)

    gemm_relu_node = onnx.helper.make_node('Relu', ['gemm_mul_node_out'], ['gemm_relu_out'])

    node8 = onnx.helper.make_node('Tdnn',
                                  inputs=['gemm_relu_out', 'layer8_params'],
                                  outputs=['out'],
                                  time_offsets=offsets_layer8,
                                  bias_params=linear_bias8,
                                  scale_params=1024)

    in_shape = (1, 40)
    out_shape = (1, 40)
    initializer = [
        offsets_layer1, offsets_layer2, offsets_layer3, offsets_layer4, offsets_layer8,
        linear_params1, linear_params2, linear_params3, linear_params4, linear_params8,
        linear_bias1, linear_bias2, linear_bias3, linear_bias4, linear_bias8, y_t, gemm_params1,
        gemm_params2, gemm_params, gemm_bias1, gemm_bias2, gemm_bias
    ]

    graph = onnx.helper.make_graph(
        [
            node1, relu_node1, node2, relu_node3, gemm_node1, gemm_mul_node1, gemm_relu_node1,
            node4, relu_node4, gemm_node2, gemm_mul_node2, gemm_relu_node2, add_node, mul_node,
            relu_node7, gemm_node, gemm_mul_node, gemm_relu_node, node8
        ],
        "tdnn_bb04p1_hua_shibie",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name='tdnn_bb04p1_hua_shibie')
    with open('./model/tdnn_bb04p1_hua_shibie.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate tdnn_bb04p1_hua_shibie.onnx sucessfully!")


def distance(a, b):
    v1 = np.sqrt(np.sum((np.int32(b) - np.int32(a))**2))
    v2 = np.sqrt(np.sum(1e-5 + np.int32(b)**2))
    v3 = v1 / v2
    ret = np.sum(v3)
    # print("compare distance is:%.4f"%(ret))
    return ret


base_out = []


def build_run_model(mod, params, optimize_method_config, data, fifo_en):
    global base_out
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    if fifo_en:
        data = np.tile(data, (1,2))
        input_dt['in'] = witin.nd.array(data)
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(
            mod,
            target='npu',
            target_host='npu',
            params=params,
            input_data=input_dt,
            chip="BB04P1",
            output_dir=build_dir,
            optimize_method_config=optimize_method_config)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    data2 = []
    for index in range(data.shape[0] - 3):
        data2.append(data[index:index+3].reshape([-1]))
    data2=np.array(data2, dtype=np.float32)

    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data2))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(1)]
    # print(output[0])
    if (optimize_method_config == ""):
        base_out = output[0]
    else:
        ret = distance(base_out, output[0])
        if 1 - ret < 0.95:
            print(base_out)
            print(output[0])
            print(base_out - output[0])
            # raise ValueError("similarity is  '%f' " % (ret))


def test_model():
    file_name = './model/tdnn_bb04p1_hua_shibie.onnx'
    onnx_model = onnx.load(file_name)
    shape_dict = {}
    shape_dict['in'] = (1, 40)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data = np.round(np.random.rand(300, 40) * 255).astype("float32")

    optimize_method_config = [
        "",
        "./model/optimize_layer/optimize_config_fifo_en_dw.protobuf",
    ]
    for config in optimize_method_config:
        fifo_en = 0
        if (config != ""):
            os.environ['WITIN_FIFO_EN'] = '1'
            fifo_en = 1
        build_run_model(mod, params, config, data, fifo_en)


def test_forward_fifo_en_dw():
    os.environ['WITIN_FIFO_EN'] = '0'
    np.random.seed(10)
    generate_tdnn_model()
    test_model()
    os.environ['WITIN_FIFO_EN'] = '0'


if __name__ == '__main__':
    test_forward_fifo_en_dw()
