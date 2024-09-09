import numpy as np
import onnx
import os
import datetime
import witin
from witin import *

root_dir = './model/'


def get_tvm_output(graph_def, data):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 4)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        output_dir=build_dir,
                                        chip="BB04P1")
    # print("npu_graph:",npu_graph)
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data))
    # execute
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list)


def generate_two_gru_model():
    gru1_params_1 = np.random.randint(-20, 20, size=(8, 8),dtype=np.int32).astype(np.float32)
    gru1_params_2 = np.random.randint(-20, 20, size=(8, 4),dtype=np.int32).astype(np.float32)

    gru1_bias_1 = 128 * np.random.randint(-20, 20, size=(8,),dtype=np.int32).astype(np.float32)
    gru1_bias_2 = 128 * np.random.randint(-20, 20, size=(4,),dtype=np.int32).astype(np.float32)

    gru1_params1 = onnx.helper.make_tensor("gru1_params1",
                    data_type=onnx.TensorProto.FLOAT, dims=(8, 8), vals=gru1_params_1.flatten())
    gru1_params2 = onnx.helper.make_tensor("gru1_params2",
                    data_type=onnx.TensorProto.FLOAT, dims=(8, 4), vals=gru1_params_2.flatten())

    gru1_bias1 = onnx.helper.make_tensor("gru1_bias1",
                    data_type=onnx.TensorProto.FLOAT, dims=(8,), vals=gru1_bias_1.flatten())
    gru1_bias2 = onnx.helper.make_tensor("gru1_bias2",
                    data_type=onnx.TensorProto.FLOAT, dims=(4,), vals=gru1_bias_2.flatten())

    gru2_params_1 = np.random.randint(-20, 20, size=(8, 8),dtype=np.int32).astype(np.float32)
    gru2_params_2 = np.random.randint(-20, 20, size=(8, 4),dtype=np.int32).astype(np.float32)

    gru2_bias_1 = 128 * np.random.randint(-20, 20, size=(8,),dtype=np.int32).astype(np.float32)
    gru2_bias_2 = 128 * np.random.randint(-20, 20, size=(4,),dtype=np.int32).astype(np.float32)

    gru2_params1 = onnx.helper.make_tensor("gru2_params1",
                    data_type=onnx.TensorProto.FLOAT, dims=(8, 8), vals=gru2_params_1.flatten())
    gru2_params2 = onnx.helper.make_tensor("gru2_params2",
                    data_type=onnx.TensorProto.FLOAT, dims=(8, 4), vals=gru2_params_2.flatten())

    gru2_bias1 = onnx.helper.make_tensor("gru2_bias1",
                    data_type=onnx.TensorProto.FLOAT, dims=(8,), vals=gru2_bias_1.flatten())
    gru2_bias2 = onnx.helper.make_tensor("gru2_bias2",
                    data_type=onnx.TensorProto.FLOAT, dims=(4,), vals=gru2_bias_2.flatten())

    act_input1024 = [i for i in range(512)] + [i - 512 for i in range(512)]
    act_table_list = act_input1024 + act_input1024
    act_table = np.array(act_table_list)
    act_table = np.reshape(act_table, (2, 1024)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 1024),
                                        vals=act_table.flatten())

    gru_node1 = onnx.helper.make_node(
        'Gru',
        inputs=['in', 'gru1_params1', 'gru1_params2', 'gru1_bias1', 'gru1_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[0, -4],
        outputs=['gru_node1_out'],
        name="gru_node1")

    gru_node2 = onnx.helper.make_node(
        'Gru',
        inputs=['gru_node1_out', 'gru2_params1', 'gru2_params2', 'gru2_bias1', 'gru2_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[0, -4],
        outputs=['out'],
        name="gru_node2")

    in_shape = (1, 4)
    out_shape = (1, 4)
    initializer = [
        gru1_params1, gru1_params2, gru1_bias1, gru1_bias2, gru2_params1, gru2_params2, gru2_bias1,
        gru2_bias2, act_table
    ]

    graph = onnx.helper.make_graph(
        [gru_node1, gru_node2],
        "test_two_gru",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)

    model = onnx.helper.make_model(graph, producer_name='test_two_gru')
    with open(root_dir + 'test_two_gru.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_two_gru.onnx sucessfully!")


def generate_gru_tdnn_model():
    tdnn_weight_1 = np.random.randint(-50, 50, size=(12, 4), dtype=np.int32).astype(np.float32)
    tdnn_bias_1 = 128 * np.random.randint(-50, 50, size=(4,), dtype=np.int32).astype(np.float32)
    offsets_layer1 = onnx.helper.make_tensor("layer1_offset",
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=(3,),
                                             vals=[-1, 0, 1])
    tdnn_weight = onnx.helper.make_tensor("tdnn_weight",
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=(12, 4),
                                          vals=tdnn_weight_1.flatten())
    tdnn_bias = onnx.helper.make_tensor("tdnn_bias",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(4,),
                                        vals=tdnn_bias_1.flatten())

    gru1_params_1 = np.random.randint(-50, 50, size=(8, 8), dtype=np.int32).astype(np.float32)
    gru1_params_2 = np.random.randint(-50, 50, size=(8, 4), dtype=np.int32).astype(np.float32)

    gru1_bias_1 = 128 * np.random.randint(-50, 50, size=(8,), dtype=np.int32).astype(np.float32)
    gru1_bias_2 = 128 * np.random.randint(-50, 50, size=(4,), dtype=np.int32).astype(np.float32)

    gru1_params1 = onnx.helper.make_tensor("gru1_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(8, 8),
                                           vals=gru1_params_1.flatten())
    gru1_params2 = onnx.helper.make_tensor("gru1_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(8, 4),
                                           vals=gru1_params_2.flatten())

    gru1_bias1 = onnx.helper.make_tensor("gru1_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(8,),
                                         vals=gru1_bias_1.flatten())
    gru1_bias2 = onnx.helper.make_tensor("gru1_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(4,),
                                         vals=gru1_bias_2.flatten())

    act_table_list_1 = [i for i in range(128)]  #0~127
    act_table_list_2 = [i - 128 for i in range(128)]  #-128~-1
    act_table_list = act_table_list_1 + act_table_list_2
    act_table_list = act_table_list + act_table_list
    act_table = np.array(act_table_list)
    act_table = np.reshape(act_table, (2, 256)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 256),
                                        vals=act_table.flatten())

    gru_node = onnx.helper.make_node(
        'Gru',
        inputs=['in', 'gru1_params1', 'gru1_params2', 'gru1_bias1', 'gru1_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[0, -4],
        outputs=['gru_out'],
        name="gru_node")

    tdnn_node = onnx.helper.make_node('Tdnn',
                                      inputs=['gru_out', 'tdnn_weight'],
                                      outputs=['out'],
                                      time_offsets=offsets_layer1,
                                      bias_params=tdnn_bias,
                                      scale_params=1024,
                                      name="tdnn_node")

    in_shape = (1, 4)
    out_shape = (1, 4)
    initializer = [
        gru1_params1, gru1_params2, gru1_bias1, gru1_bias2, tdnn_weight, offsets_layer1, tdnn_bias,
        act_table
    ]

    graph = onnx.helper.make_graph(
        [gru_node, tdnn_node],
        "test_gru_tdnn",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)

    model = onnx.helper.make_model(graph, producer_name='test_gru_tdnn')
    with open(root_dir + 'test_gru_tdnn.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_gru_tdnn.onnx sucessfully!")


def generate_gru_dnn_model():
    dnn_weight_params = np.random.randint(-128, 127, size=(4, 4), dtype=np.int32).astype(np.float32)
    dnn_bias_params = 128 * np.random.randint(-128, 127, size=(4,), dtype=np.int32).astype(
        np.float32)
    dnn_weight = onnx.helper.make_tensor("dnn_weight",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(4, 4),
                                         vals=dnn_weight_params.flatten())
    dnn_bias = onnx.helper.make_tensor("dnn_bias",
                                       data_type=onnx.TensorProto.FLOAT,
                                       dims=(4,),
                                       vals=dnn_bias_params.flatten())

    gru1_params_1 = np.random.randint(-128, 127, size=(8, 8), dtype=np.int32).astype(np.float32)
    gru1_params_2 = np.random.randint(-128, 127, size=(8, 4), dtype=np.int32).astype(np.float32)

    gru1_bias_1 = 128 * np.random.randint(-128, 127, size=(8,), dtype=np.int32).astype(np.float32)
    gru1_bias_2 = 128 * np.random.randint(-128, 127, size=(4,), dtype=np.int32).astype(np.float32)

    gru1_params1 = onnx.helper.make_tensor("gru1_params1",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(8, 8),
                                           vals=gru1_params_1.flatten())
    gru1_params2 = onnx.helper.make_tensor("gru1_params2",
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=(8, 4),
                                           vals=gru1_params_2.flatten())

    gru1_bias1 = onnx.helper.make_tensor("gru1_bias1",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(8,),
                                         vals=gru1_bias_1.flatten())
    gru1_bias2 = onnx.helper.make_tensor("gru1_bias2",
                                         data_type=onnx.TensorProto.FLOAT,
                                         dims=(4,),
                                         vals=gru1_bias_2.flatten())

    act_table_list_1 = [i for i in range(128)]  #0~127
    act_table_list_2 = [i - 128 for i in range(128)]  #-128~-1
    act_table_list = act_table_list_1 + act_table_list_2
    act_table_list = act_table_list + act_table_list
    act_table = np.array(act_table_list)
    act_table = np.reshape(act_table, (2, 256)).astype(np.float32)
    act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(2, 256),
                                        vals=act_table.flatten())

    gru_node = onnx.helper.make_node(
        'Gru',
        inputs=['in', 'gru1_params1', 'gru1_params2', 'gru1_bias1', 'gru1_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[0, -4],
        outputs=['gru_out'],
        name="gru_node")

    dnn_node = onnx.helper.make_node('Gemm', ['gru_out', 'dnn_weight', 'dnn_bias'], ['out'],
                                     name="dnn_node")

    in_shape = (1, 4)
    out_shape = (1, 4)
    initializer = [
        gru1_params1, gru1_params2, gru1_bias1, gru1_bias2, dnn_weight, dnn_bias, act_table
    ]

    graph = onnx.helper.make_graph(
        [gru_node, dnn_node],
        "test_gru_dnn",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(out_shape))
        ],
        initializer=initializer)

    model = onnx.helper.make_model(graph, producer_name='test_gru_dnn')
    with open(root_dir + 'test_gru_dnn.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_gru_dnn.onnx sucessfully!")


def test_two_gru_model():
    generate_two_gru_model()
    file_name = root_dir + './test_two_gru.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, 4) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, data)


def test_gru_tdnn_model():
    generate_gru_tdnn_model()
    file_name = root_dir + './test_gru_tdnn.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(14, 4) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, data)


def test_gru_dnn_model():
    generate_gru_dnn_model()
    file_name = root_dir + './test_gru_dnn.onnx'
    onnx_model = onnx.load(file_name)
    data = np.round(np.random.rand(11, 4) * 255).astype("float32")
    tvm_out = get_tvm_output(onnx_model, data)

def test_many_gru():
    np.random.seed(10)
    test_two_gru_model()
    test_gru_dnn_model()

if __name__ == '__main__':
    np.random.seed(10)
    test_two_gru_model()
    # test_gru_tdnn_model()
    test_gru_dnn_model()
