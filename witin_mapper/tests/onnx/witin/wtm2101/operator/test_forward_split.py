import numpy as np
import onnx
import datetime
import os
import witin
from witin import *


def generate_model():
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_paths = []
    # net1
    # conv1->conv2.1
    # layer 1
    weight1 = np.round(
        np.random.rand(256, 11, 7, 1) * 255).astype("float32") - 128
    bias1 = np.round(
        np.random.rand(256) * 127).astype("float32") * 128
    weight1_tensor = onnx.helper.make_tensor(
        'weight1_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(weight1),
        vals=weight1.flatten())
    bias1_tensor = onnx.helper.make_tensor(
        'bias1_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(bias1),
        vals=bias1.flatten())

    conv1 = onnx.helper.make_node(
        "Conv",
        inputs=['in', 'weight1_tensor', 'bias1_tensor'],
        outputs=['conv1_out'],
        kernel_shape=[7, 1],
        strides=[4, 1],
        pads=[2, 0, 1, 0],
        name="conv1")
    conv1_scale = onnx.helper.make_node("Scale",
                                        inputs=['conv1_out'],
                                        outputs=['conv1_scale'],
                                        scale=0.0009765625)
    conv1_relu = onnx.helper.make_node('Relu', ['conv1_scale'],
                                       ['conv1_relu'])
    node_list = [conv1, conv1_scale, conv1_relu]
    initializer = [weight1_tensor, bias1_tensor]

    # layer 2
    weight2 = np.round(
        np.random.rand(196, 256, 1, 1) * 255).astype("float32") - 128
    bias2 = np.round(
        np.random.rand(196) * 127).astype("float32") * 128
    weight2_tensor = onnx.helper.make_tensor(
        'weight2_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(weight2),
        vals=weight2.flatten())
    bias2_tensor = onnx.helper.make_tensor(
        'bias2_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=np.shape(bias2),
        vals=bias2.flatten())

    conv2 = onnx.helper.make_node(
        "Conv",
        inputs=['conv1_relu', 'weight2_tensor', 'bias2_tensor'],
        outputs=['conv2_out'],
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        name="conv2")
    conv2_scale = onnx.helper.make_node("Scale",
                                        inputs=['conv2_out'],
                                        outputs=['conv2_scale'],
                                        scale=0.0009765625)
    # conv2_relu = onnx.helper.make_node('Relu', ['conv2_scale'],
    #                                         ['conv2_relu'])
    node_list = node_list + [conv2, conv2_scale]
    initializer = initializer + [weight2_tensor, bias2_tensor]

    # layer 3
    split_node = onnx.helper.make_node(
        'Split',
        inputs=['conv2_scale'],
        outputs=['split_node1', 'split_node2'],
        axis=1,
        # split=[98, 98]
    )
    node_list = node_list + [split_node]

    input_shape = (1, 11, 32, 1)
    output_shape = (1, 96, 8, 1)

    graph = onnx.helper.make_graph(
        node_list,
        "heart_cnn2",
        inputs=[
            onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                               list(input_shape)),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("split_node1", onnx.TensorProto.FLOAT,
                                               list(output_shape)),
            onnx.helper.make_tensor_value_info("split_node2", onnx.TensorProto.FLOAT,
                                               list(output_shape))
        ],
        initializer=initializer)
    model = onnx.helper.make_model(graph,
                                   producer_name='huawei_wear_heart_model')

    model_path = "./model/test_split" + ".onnx"  # timestampStr
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    model_paths.append(model_path)

    print("Generate huawei_wear_heart-model sucessfully!")
    return model_paths


def get_tvm_output(model_paths, graph_id):
    """ Generic function to execute and get tvm output"""
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")

    target = 'npu'
    target_host = 'npu'
    graph_def0 = onnx.load(model_paths[0])

    mods = []
    pms = []
    input_dts = []

    # graph0
    shape_dict = {}
    shape_dict['in'] = (1, 11, 32, 1)  # NCHW
    mod, params = witin_frontend.frontend.from_onnx(graph_def0, shape_dict)
    mods.append(mod)
    pms.append(params)
    # NWHC
    data0 = np.round(np.random.rand(100, 1, 32, 11) * 255).astype("float32")
    input_dt = {}
    input_dt['conv1'] = witin.nd.array(data0)
    input_dts.append(input_dt)

    build_dir = "./output"
    # + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mods,
                                                               target=target,
                                                               target_host=target_host,
                                                               params=pms, input_data=input_dts,
                                                               output_dir=build_dir,
                                                               chip="BB04P1")
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    # from tvm.contrib import npu_graph_runtime

    # m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    # m.set_input('in1', witin.nd.array(data))
    # # execute
    # m.run()
    # witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]


def test_forward_op_split():
    np.random.seed(10)
    model_paths = generate_model()
    tvm_out = get_tvm_output(model_paths, 0)


if __name__ == '__main__':
    test_forward_op_split()
