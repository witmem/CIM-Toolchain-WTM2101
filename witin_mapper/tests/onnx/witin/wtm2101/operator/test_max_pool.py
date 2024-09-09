import numpy as np
import onnx
from onnx import helper, TensorProto
import os
import datetime
import witin
from witin import *

np.random.seed(10)

ROOT_DIR = './model/'


def generate_avgpool_model_case1():
    """
    input_shape: [1, 3, 32, 32]
    output_shape: [1, 3, 31, 31]
    output_shape: [1, 3, 1, 1]
    """
    model_name = "AveragePool_case_1"
    # in_shape = (1, 3, 32, 32)
    node = onnx.helper.make_node(
        "AveragePool",
        inputs=["in"],
        outputs=["out1"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],  #[top,left,bottom,right]
        strides=[1, 1],
    )
    # out_shape = (1, 3, 31, 31)

    node2 = onnx.helper.make_node(
        "GlobalAveragePool",
        inputs=["out1"],
        outputs=["out"],
    )

    in_shape = (1, 3, 32, 32)
    out_shape = (1, 3, 1, 1)
    initializer = []

    graph = onnx.helper.make_graph(
        [node,node2],
        model_name,
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)
    model = helper.make_model(graph, producer_name=model_name)
    with open(ROOT_DIR + model_name + '.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate %s.onnx sucessfully!" % model_name)
    return ROOT_DIR + model_name + '.onnx'


def build_run_model(mod, params, optimize_method_config, data):
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0]  #+ timestampStr
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
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(1)]


def test_average_pool():
    file_path = generate_avgpool_model_case1()
    onnx_model = onnx.load(file_path)
    shape_dict = {}
    shape_dict['in'] = (1, 3, 32, 32)  #NCHW
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data = np.round(np.random.rand(20, 32, 32, 3) * 110).astype("float32")  #NWHC

    build_run_model(mod, params, "", data)


test_average_pool()