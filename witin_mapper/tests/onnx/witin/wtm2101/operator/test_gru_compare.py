import numpy as np
import onnx
from onnx import helper
import datetime
import os
import witin
from witin import *

root_dir = './model/'

def get_tvm_output(graph_def):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 128)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)
    data1 = np.round(np.random.rand(11, 128)*255).astype("float32")
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data1)
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
    # from tvm.contrib import npu_graph_runtime
    # m = npu_graph_runtime.create(npu_graph, "BB04P1")
    # m.set_input('in1', witin.nd.array(data1))
    # # execute
    # m.run()
    # witin_output_list = [m.get_output(i).asnumpy()
    #                         for i in range(1)]


def generate_gru_model(params_path):
    # gru_params_1 = np.random.randint(-256, 255, size=(256, 256),dtype=np.int32).astype(np.float32)
    # gru_params_2 = np.random.randint(-256, 255, size=(256, 128),dtype=np.int32).astype(np.float32)

    # gru_bias_1 = 128 * np.random.randint(-128, 127, size=(256,),dtype=np.int32).astype(np.float32)
    # gru_bias_2 = 128 * np.random.randint(-128, 127, size=(128,),dtype=np.int32).astype(np.float32)


    gru_params_1 = np.loadtxt("%s/array1.txt"%(params_path),dtype=np.float32)
    gru_params_2 = np.loadtxt("%s/array2.txt"%(params_path),dtype=np.float32)

    gru_bias_1 = np.loadtxt("%s/bias_zr.txt"%(params_path),dtype=np.float32)
    gru_bias_2 = np.loadtxt("%s/bias_ht.txt"%(params_path),dtype=np.float32)

    gru_params1 = onnx.helper.make_tensor("gru_params1",
                    data_type=onnx.TensorProto.FLOAT, dims=(256, 256), vals=gru_params_1.flatten())
    gru_params2 = onnx.helper.make_tensor("gru_params2",
                    data_type=onnx.TensorProto.FLOAT, dims=(256, 128), vals=gru_params_2.flatten())

    gru_bias1 = onnx.helper.make_tensor("gru_bias1",
                    data_type=onnx.TensorProto.FLOAT, dims=(256,), vals=gru_bias_1.flatten())
    gru_bias2 = onnx.helper.make_tensor("gru_bias2",
                    data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=gru_bias_2.flatten())

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
        inputs=['in', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],
        scale_zr=1024,
        scale_ht=1024,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        outputs=['out'],
        name="gru_node1")


    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer=[gru_params1, gru_params2, gru_bias1, gru_bias2, act_table]

    graph = onnx.helper.make_graph([gru_node1], "test_gru_compare",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )
    model=helper.make_model(graph, producer_name='test_gru_compare')
    with open(root_dir + 'test_gru_compare.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate test_gru_compare.onnx sucessfully!")

def test_model():
    file_name = root_dir + './test_gru_compare.onnx'
    onnx_model = onnx.load(file_name)
    target="npu"
    ctx="npu"
    tvm_out = get_tvm_output(onnx_model)

if __name__ == '__main__':
    np.random.seed(10)
    params_path = "gru_params"
    generate_gru_model(params_path)
    test_model()
