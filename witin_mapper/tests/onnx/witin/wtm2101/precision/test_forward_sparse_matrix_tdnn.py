import onnx
import witin
from witin import *
import numpy as np
import math
from onnx import helper, TensorProto, mapping
import scipy
import os, sys

def get_tvm_output(graph_def, config):
    """ Generic function to execute and get tvm output"""
    target = 'npu'
    target_host = 'npu'
    shape_dict = {}
    shape_dict['in'] = (1, 128)
    mod, params = witin_frontend.frontend.from_onnx(graph_def, shape_dict)

    #feats = np.loadtxt('../data/howl/feats_int8_yuexi.txt').astype("float32")
    feats = np.loadtxt('./model/pipeline/sparse_matrix_model/params/data.txt').astype("float32")[70:170]
    data1 = feats[:][:]
    #print(len(data1), len(data1[0]))
    #data2 = data1
    data2 = []
    tempdata = np.zeros(128*3)
    for i in range(len(data1)):
        tempdata = np.concatenate([tempdata[128:], data1[i]])
        data2.append(tempdata)
    data2 = np.array(data2)
    print(len(data2), len(data2[0]))
    data2 = data2.astype(np.float32)
    
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data2)
    print("using BB04P1!")

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                        target=target,
                                        target_host=target_host,
                                        params=params, input_data=input_dt,
                                        chip = "BB04P1",
                                        output_dir=build_dir,
                                        optimize_method_config = config
                                        )
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    #'''
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in', witin.nd.array(data2))
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
    print(witin_output_list[0].shape)

def generate_tdnn_model(modelpath):
    howl_weight_1 =  np.loadtxt("%s/w_0.txt"%modelpath, dtype=np.float32).transpose()
    howl_weight_2 =  np.loadtxt("%s/w_1.txt"%modelpath, dtype=np.float32).transpose()*2
    howl_weight_3 =  np.loadtxt("%s/w_2.txt"%modelpath, dtype=np.float32).transpose()
    howl_bias_1 = np.loadtxt("%s/b_0.txt"%modelpath, dtype=np.float32)
    howl_bias_2 = np.loadtxt("%s/b_1.txt"%modelpath, dtype=np.float32)*2
    howl_bias_3 = np.loadtxt("%s/b_2.txt"%modelpath, dtype=np.float32)


    offsets_layer1 = onnx.helper.make_tensor("layer1_offset",
                    data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer2 = onnx.helper.make_tensor("layer2_offset",
                    data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])
    offsets_layer3 = onnx.helper.make_tensor("layer3_offset",
                    data_type=onnx.TensorProto.FLOAT, dims=(3,), vals=[-1,0,1])

    linear_params1 = onnx.helper.make_tensor("layer1_weight",
                    data_type=onnx.TensorProto.FLOAT, dims=(384, 48),  vals=howl_weight_1.flatten())
    linear_params2 = onnx.helper.make_tensor("layer2_weight",
                    data_type=onnx.TensorProto.FLOAT, dims=(144, 48),  vals=howl_weight_2.flatten())
    linear_params3 = onnx.helper.make_tensor("layer3_weight",
                    data_type=onnx.TensorProto.FLOAT, dims=(144, 128), vals=howl_weight_3.flatten())

    linear_bias1 = onnx.helper.make_tensor("layer1_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(48,),  vals=howl_bias_1.flatten())
    linear_bias2 = onnx.helper.make_tensor("layer2_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(48,),  vals=howl_bias_2.flatten())
    linear_bias3 = onnx.helper.make_tensor("layer3_bias",
                    data_type=onnx.TensorProto.FLOAT, dims=(128,), vals=howl_bias_3.flatten())

    
    G = [8192, 2048, 1024]
    node1=onnx.helper.make_node('Tdnn', inputs=['in', 'layer1_weight'],
                                outputs=['tdnn1'],
                                time_offsets=offsets_layer1,
                                bias_params=linear_bias1,
                                scale_params=G[0],
                                name="tdnn_0"
                                )

    relu_node1 = onnx.helper.make_node('Relu', inputs=['tdnn1'],
                                                outputs=['relu_out1'])

    node2=onnx.helper.make_node('Tdnn', inputs=['relu_out1', 'layer2_weight'],
                                outputs=['tdnn2'],
                                time_offsets=offsets_layer2,
                                bias_params=linear_bias2,
                                scale_params=G[1],
                                name="tdnn_1"
                                )

    relu_node2 = onnx.helper.make_node('Relu', ['tdnn2'], ['relu_out2'])

    node3=onnx.helper.make_node('Tdnn', inputs=['relu_out2', 'layer3_weight'],
                                outputs=['out'],
                                time_offsets=offsets_layer3,
                                bias_params=linear_bias3,
                                scale_params=G[2],
                                name="tdnn_2"
                                )


    in_shape = (1, 128)
    out_shape = (1, 128)
    initializer=[offsets_layer1, offsets_layer2, offsets_layer3,
            linear_params1, linear_params2, linear_params3,
            linear_bias1, linear_bias2, linear_bias3]

    graph = onnx.helper.make_graph([node1, relu_node1, node2, relu_node2, node3], "howl",
            inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT,
                                                        list(in_shape))],
            outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,
                                                        list(out_shape))],
            initializer=initializer
            )

    model = helper.make_model(graph, producer_name='howl')
    with open('%s/howl_net_int8.onnx'%modelpath, "wb") as of:
        of.write(model.SerializeToString())
    

    print("generate howl_dnn.onnx sucessfully!")

def test_model(modelpath):
    file_name = '%s/howl_net_int8.onnx'%modelpath
    onnx_model = onnx.load(file_name)
    target="npu"
    ctx="npu"
    config = './model/pipeline/sparse_matrix_model/opt_sparse.protobuf'
    tvm_out = get_tvm_output(onnx_model, config)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("py modelpath")
    #     exit(0)
    # else:
    #     modelpath = sys.argv[1]
    modelpath = './model/pipeline/sparse_matrix_model//params/'
    generate_tdnn_model(modelpath)
    test_model(modelpath)
