import numpy as np
import onnx
from onnx import helper, TensorProto
import os
import datetime
import witin
from witin import *

root_dir = './model/'
np.random.seed(10)


def mk_conv1d_node(name, inp, kernel, bias, time_offset, G, actv_type):
    conv_out = name+"_out"
    node = helper.make_node('Tdnn', inputs=[inp, kernel],
                            outputs=[conv_out],
                            time_offsets=time_offset,
                            bias_params=bias,
                            scale_params=G,
                            name=name)
    if actv_type == "RELU":
        out_name = name+'_relu'
        node_actv = helper.make_node('Relu', [conv_out], [out_name])
        return [node, node_actv], out_name
    elif actv_type == "NONE" or actv_type == "PASS":
        out_name = conv_out
        return [node], out_name
    else:
        print("ERR actv type:", actv_type)
        exit(0)


def mk_dnn_node(name, inp_s, kernel_s, bias_s, G, actv_type):
    dnn_out = name+'_scale_out'
    node_dnn = helper.make_node('Gemm', inputs=[inp_s, kernel_s, bias_s],
                                outputs=[name+'_dnn_out'], name=name)
    node_mul = helper.make_node('Scale', [name+'_dnn_out'], [dnn_out],
                                scale=1/G)
    if actv_type == "RELU":
        out_name = name+'_relu'
        node_actv = helper.make_node('Relu', [dnn_out], [out_name])
        return [node_dnn, node_mul, node_actv], out_name
    elif actv_type == "NONE" or actv_type == "PASS":
        out_name = dnn_out
        return [node_dnn, node_mul], out_name
    else:
        print("ERR actv type:", actv_type)
        exit(0)


def create_TC_resnet_onnx(base_path):

    grp0_a_0 = np.random.randint(-128, 127, size=(128, 64),dtype=np.int32).astype(np.float32)
    grp0_a_1 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp0_a_2 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp0_b_0 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp0_b_1 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)

    grp1_a_0 = np.random.randint(-128, 127, size=(64, 64),dtype=np.int32).astype(np.float32)
    grp1_a_1 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp1_a_2 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp1_b_0 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp1_b_1 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)

    grp2_a_0 = np.random.randint(-128, 127, size=(64, 64),dtype=np.int32).astype(np.float32)
    grp2_a_1 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp2_a_2 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp2_b_0 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp2_b_1 = np.random.randint(-128, 127, size=(320, 64),dtype=np.int32).astype(np.float32)
    grp2_b_2 = np.random.randint(-128, 127, size=(64, 128),dtype=np.int32).astype(np.float32)

    offset_grp0_a_0_ = helper.make_tensor("offset_grp0_a_0", data_type=TensorProto.FLOAT, dims=(1,), vals=[0])
    offset_grp0_a_1_ = helper.make_tensor("offset_grp0_a_1", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp0_a_2_ = helper.make_tensor("offset_grp0_a_2", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp0_b_0_ = helper.make_tensor("offset_grp0_b_0", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp0_b_1_ = helper.make_tensor("offset_grp0_b_1", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])

    offset_grp1_a_0_ = helper.make_tensor("offset_grp1_a_0", data_type=TensorProto.FLOAT, dims=(1,), vals=[0])
    offset_grp1_a_1_ = helper.make_tensor("offset_grp1_a_1", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp1_a_2_ = helper.make_tensor("offset_grp1_a_2", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp1_b_0_ = helper.make_tensor("offset_grp1_b_0", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp1_b_1_ = helper.make_tensor("offset_grp1_b_1", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])

    offset_grp2_a_0_ = helper.make_tensor("offset_grp2_a_0", data_type=TensorProto.FLOAT, dims=(1,), vals=[0])
    offset_grp2_a_1_ = helper.make_tensor("offset_grp2_a_1", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp2_a_2_ = helper.make_tensor("offset_grp2_a_2", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp2_b_0_ = helper.make_tensor("offset_grp2_b_0", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])
    offset_grp2_b_1_ = helper.make_tensor("offset_grp2_b_1", data_type=TensorProto.FLOAT, dims=(5,), vals=[-4,-3,-2, -1, 0])

    conv_grp0_a_0_ = helper.make_tensor("conv_grp0_a_0", data_type=TensorProto.FLOAT, dims=np.shape(grp0_a_0),vals=grp0_a_0.flatten())
    conv_grp0_a_1_ = helper.make_tensor("conv_grp0_a_1", data_type=TensorProto.FLOAT, dims=np.shape(grp0_a_1),vals=grp0_a_1.flatten())
    conv_grp0_a_2_ = helper.make_tensor("conv_grp0_a_2", data_type=TensorProto.FLOAT, dims=np.shape(grp0_a_2),vals=grp0_a_2.flatten())
    conv_grp0_b_0_ = helper.make_tensor("conv_grp0_b_0", data_type=TensorProto.FLOAT, dims=np.shape(grp0_b_0),vals=grp0_b_0.flatten())
    conv_grp0_b_1_ = helper.make_tensor("conv_grp0_b_1", data_type=TensorProto.FLOAT, dims=np.shape(grp0_b_1),vals=grp0_b_1.flatten())

    conv_grp1_a_0_ = helper.make_tensor("conv_grp1_a_0", data_type=TensorProto.FLOAT, dims=np.shape(grp1_a_0),vals=grp1_a_0.flatten())
    conv_grp1_a_1_ = helper.make_tensor("conv_grp1_a_1", data_type=TensorProto.FLOAT, dims=np.shape(grp1_a_1),vals=grp1_a_1.flatten())
    conv_grp1_a_2_ = helper.make_tensor("conv_grp1_a_2", data_type=TensorProto.FLOAT, dims=np.shape(grp1_a_2),vals=grp1_a_2.flatten())
    conv_grp1_b_0_ = helper.make_tensor("conv_grp1_b_0", data_type=TensorProto.FLOAT, dims=np.shape(grp1_b_0),vals=grp1_b_0.flatten())
    conv_grp1_b_1_ = helper.make_tensor("conv_grp1_b_1", data_type=TensorProto.FLOAT, dims=np.shape(grp1_b_1),vals=grp1_b_1.flatten())

    conv_grp2_a_0_ = helper.make_tensor("conv_grp2_a_0", data_type=TensorProto.FLOAT, dims=np.shape(grp2_a_0),vals=grp2_a_0.flatten())
    conv_grp2_a_1_ = helper.make_tensor("conv_grp2_a_1", data_type=TensorProto.FLOAT, dims=np.shape(grp2_a_1),vals=grp2_a_1.flatten())
    conv_grp2_a_2_ = helper.make_tensor("conv_grp2_a_2", data_type=TensorProto.FLOAT, dims=np.shape(grp2_a_2),vals=grp2_a_2.flatten())
    conv_grp2_b_0_ = helper.make_tensor("conv_grp2_b_0", data_type=TensorProto.FLOAT, dims=np.shape(grp2_b_0),vals=grp2_b_0.flatten())
    conv_grp2_b_1_ = helper.make_tensor("conv_grp2_b_1", data_type=TensorProto.FLOAT, dims=np.shape(grp2_b_1),vals=grp2_b_1.flatten())
    conv_grp2_b_2_ = helper.make_tensor("conv_grp2_b_2", data_type=TensorProto.FLOAT, dims=np.shape(grp2_b_2),vals=grp2_b_2.flatten())

    b_grp0_a_0_ = helper.make_tensor("b_grp0_a_0", data_type=TensorProto.FLOAT, dims=(np.shape(grp0_a_0)[-1],), vals=np.zeros(np.shape(grp0_a_0)[-1],dtype=np.float32).flatten())
    b_grp0_a_1_ = helper.make_tensor("b_grp0_a_1", data_type=TensorProto.FLOAT, dims=(np.shape(grp0_a_1)[-1],), vals=np.zeros(np.shape(grp0_a_1)[-1],dtype=np.float32).flatten())
    b_grp0_a_2_ = helper.make_tensor("b_grp0_a_2", data_type=TensorProto.FLOAT, dims=(np.shape(grp0_a_2)[-1],), vals=np.zeros(np.shape(grp0_a_2)[-1],dtype=np.float32).flatten())
    b_grp0_b_0_ = helper.make_tensor("b_grp0_b_0", data_type=TensorProto.FLOAT, dims=(np.shape(grp0_b_0)[-1],), vals=np.zeros(np.shape(grp0_b_0)[-1],dtype=np.float32).flatten())
    b_grp0_b_1_ = helper.make_tensor("b_grp0_b_1", data_type=TensorProto.FLOAT, dims=(np.shape(grp0_b_1)[-1],), vals=np.zeros(np.shape(grp0_b_1)[-1],dtype=np.float32).flatten())

    b_grp1_a_0_ = helper.make_tensor("b_grp1_a_0", data_type=TensorProto.FLOAT, dims=(np.shape(grp1_a_0)[-1],), vals=np.zeros(np.shape(grp1_a_0)[-1],dtype=np.float32).flatten())
    b_grp1_a_1_ = helper.make_tensor("b_grp1_a_1", data_type=TensorProto.FLOAT, dims=(np.shape(grp1_a_1)[-1],), vals=np.zeros(np.shape(grp1_a_1)[-1],dtype=np.float32).flatten())
    b_grp1_a_2_ = helper.make_tensor("b_grp1_a_2", data_type=TensorProto.FLOAT, dims=(np.shape(grp1_a_2)[-1],), vals=np.zeros(np.shape(grp1_a_2)[-1],dtype=np.float32).flatten())
    b_grp1_b_0_ = helper.make_tensor("b_grp1_b_0", data_type=TensorProto.FLOAT, dims=(np.shape(grp1_b_0)[-1],), vals=np.zeros(np.shape(grp1_b_0)[-1],dtype=np.float32).flatten())
    b_grp1_b_1_ = helper.make_tensor("b_grp1_b_1", data_type=TensorProto.FLOAT, dims=(np.shape(grp1_b_1)[-1],), vals=np.zeros(np.shape(grp1_b_1)[-1],dtype=np.float32).flatten())

    b_grp2_a_0_ = helper.make_tensor("b_grp2_a_0", data_type=TensorProto.FLOAT, dims=(np.shape(grp2_a_0)[-1],), vals=np.zeros(np.shape(grp2_a_0)[-1],dtype=np.float32).flatten())
    b_grp2_a_1_ = helper.make_tensor("b_grp2_a_1", data_type=TensorProto.FLOAT, dims=(np.shape(grp2_a_1)[-1],), vals=np.zeros(np.shape(grp2_a_1)[-1],dtype=np.float32).flatten())
    b_grp2_a_2_ = helper.make_tensor("b_grp2_a_2", data_type=TensorProto.FLOAT, dims=(np.shape(grp2_a_2)[-1],), vals=np.zeros(np.shape(grp2_a_2)[-1],dtype=np.float32).flatten())
    b_grp2_b_0_ = helper.make_tensor("b_grp2_b_0", data_type=TensorProto.FLOAT, dims=(np.shape(grp2_b_0)[-1],), vals=np.zeros(np.shape(grp2_b_0)[-1],dtype=np.float32).flatten())
    b_grp2_b_1_ = helper.make_tensor("b_grp2_b_1", data_type=TensorProto.FLOAT, dims=(np.shape(grp2_b_1)[-1],), vals=np.zeros(np.shape(grp2_b_1)[-1],dtype=np.float32).flatten())
    b_grp2_b_2_ = helper.make_tensor("b_grp2_b_2", data_type=TensorProto.FLOAT, dims=(np.shape(grp2_b_2)[-1],), vals=np.zeros(np.shape(grp2_b_2)[-1],dtype=np.float32).flatten())

    node_grp0_a_0,out_grp0_a_0 = mk_dnn_node("node_grp0_a_0","in1","conv_grp0_a_0","b_grp0_a_0",512,"RELU")
    node_grp0_a_1,out_grp0_a_1 = mk_conv1d_node("node_grp0_a_1","in2","conv_grp0_a_1",b_grp0_a_1_,offset_grp0_a_1_,512,"RELU")
    node_grp0_a_2,out_grp0_a_2 = mk_conv1d_node("node_grp0_a_2",out_grp0_a_1,"conv_grp0_a_2",b_grp0_a_2_,offset_grp0_a_2_,1024,"NONE")
    node_grp0_a_add_ = helper.make_node('Add', [out_grp0_a_0, out_grp0_a_2], ['node_grp0_a_add'])
    node_grp0_a_relu_ = helper.make_node('Relu', ['node_grp0_a_add'], ['node_grp0_a_relu'])
    node_grp0_b_0,out_grp0_b_0 = mk_conv1d_node("node_grp0_b_0","node_grp0_a_relu",  "conv_grp0_b_0",b_grp0_b_0_,offset_grp0_b_0_,1024,"RELU")
    node_grp0_b_1,out_grp0_b_1 = mk_conv1d_node("node_grp0_b_1",out_grp0_b_0,  "conv_grp0_b_1",b_grp0_b_1_,offset_grp0_b_1_,1024,"NONE")
    node_grp0_b_add_ = helper.make_node('Add', ["node_grp0_a_relu", out_grp0_b_1], ['node_grp0_b_add'])
    node_grp0_b_relu_ = helper.make_node('Relu', ['node_grp0_b_add'], ['node_grp0_b_relu'])
    all_nodes_grp0 = node_grp0_a_0+node_grp0_a_1+node_grp0_a_2+[node_grp0_a_add_,node_grp0_a_relu_]+node_grp0_b_0+node_grp0_b_1+[node_grp0_b_add_,node_grp0_b_relu_]

    node_grp1_a_0,out_grp1_a_0 = mk_dnn_node("node_grp1_a_0",'node_grp0_b_relu',"conv_grp1_a_0","b_grp1_a_0",4096,"RELU")
    node_grp1_a_1,out_grp1_a_1 = mk_conv1d_node("node_grp1_a_1",'node_grp0_b_relu',"conv_grp1_a_1",b_grp1_a_1_,offset_grp1_a_1_,2048,"RELU")
    node_grp1_a_2,out_grp1_a_2 = mk_conv1d_node("node_grp1_a_2",out_grp1_a_1,"conv_grp1_a_2",b_grp1_a_2_,offset_grp1_a_2_,2048,"NONE")
    node_grp1_a_add_ = helper.make_node('Add', [out_grp1_a_0, out_grp1_a_2], ['node_grp1_a_add'])
    node_grp1_a_relu_ = helper.make_node('Relu', ['node_grp1_a_add'], ['node_grp1_a_relu'])
    node_grp1_b_0,out_grp1_b_0 = mk_conv1d_node("node_grp1_b_0","node_grp1_a_relu",  "conv_grp1_b_0",b_grp1_b_0_,offset_grp1_b_0_,1024,"RELU")
    node_grp1_b_1,out_grp1_b_1 = mk_conv1d_node("node_grp1_b_1",out_grp1_b_0,  "conv_grp1_b_1",b_grp1_b_1_,offset_grp1_b_1_,1024,"NONE")
    node_grp1_b_add_ = helper.make_node('Add', ["node_grp1_a_relu", out_grp1_b_1], ['node_grp1_b_add'])
    node_grp1_b_relu_ = helper.make_node('Relu', ['node_grp1_b_add'], ['node_grp1_b_relu'])
    all_nodes_grp1 = node_grp1_a_0+node_grp1_a_1+node_grp1_a_2+[node_grp1_a_add_,node_grp1_a_relu_]+node_grp1_b_0+node_grp1_b_1+[node_grp1_b_add_,node_grp1_b_relu_]

    node_grp2_a_0,out_grp2_a_0 = mk_dnn_node("node_grp2_a_0",'node_grp1_b_relu',"conv_grp2_a_0","b_grp2_a_0",4096,"RELU")
    node_grp2_a_1,out_grp2_a_1 = mk_conv1d_node("node_grp2_a_1",'node_grp1_b_relu',"conv_grp2_a_1",b_grp2_a_1_,offset_grp2_a_1_,2048,"RELU")
    node_grp2_a_2,out_grp2_a_2 = mk_conv1d_node("node_grp2_a_2",out_grp2_a_1,"conv_grp2_a_2",b_grp2_a_2_,offset_grp2_a_2_,2048,"NONE")
    node_grp2_a_add_ = helper.make_node('Add', [out_grp2_a_0, out_grp2_a_2], ['node_grp2_a_add'])
    node_grp2_a_relu_ = helper.make_node('Relu', ['node_grp2_a_add'], ['node_grp2_a_relu'])
    node_grp2_b_0,out_grp2_b_0 = mk_conv1d_node("node_grp2_b_0","node_grp2_a_relu",  "conv_grp2_b_0",b_grp2_b_0_,offset_grp2_b_0_,1024,"RELU")
    node_grp2_b_1,out_grp2_b_1 = mk_conv1d_node("node_grp2_b_1",out_grp2_b_0,  "conv_grp2_b_1",b_grp2_b_1_,offset_grp2_b_1_,1024,"NONE") #!!!!out1
    node_grp2_b_2,out_grp2_b_2 = mk_dnn_node("node_grp2_b_2",'node_grp2_a_relu',"conv_grp2_b_2","b_grp2_b_2",4096,"NONE")#!!!!out2

    all_nodes_grp2 = node_grp2_a_0+node_grp2_a_1+node_grp2_a_2+[node_grp2_a_add_,node_grp2_a_relu_]+node_grp2_b_0+node_grp2_b_1+node_grp2_b_2

    all_nodes = []
    all_nodes = all_nodes+all_nodes_grp0+all_nodes_grp1+all_nodes_grp2

    in_node_name1 = 'in1'
    in_node_name2 = 'in2'
    out_node_name1 = out_grp2_b_1
    out_node_name2 = out_grp2_b_2

    in_shape1 = (1, 128)
    in_shape2 = (1, 64)
    out_shape1 = (1, 64)
    out_shape2 = (1, 128)

    initializer=[
        offset_grp0_a_0_,offset_grp0_a_1_,offset_grp0_a_2_,offset_grp0_b_0_,offset_grp0_b_1_,
        offset_grp1_a_0_,offset_grp1_a_1_,offset_grp1_a_2_,offset_grp1_b_0_,offset_grp1_b_1_,
        offset_grp2_a_0_,offset_grp2_a_1_,offset_grp2_a_2_,offset_grp2_b_0_,offset_grp2_b_1_,
        conv_grp0_a_0_,conv_grp0_a_1_,conv_grp0_a_2_,conv_grp0_b_0_,conv_grp0_b_1_,
        conv_grp1_a_0_,conv_grp1_a_1_,conv_grp1_a_2_,conv_grp1_b_0_,conv_grp1_b_1_,
        conv_grp2_a_0_,conv_grp2_a_1_,conv_grp2_a_2_,conv_grp2_b_0_,conv_grp2_b_1_,conv_grp2_b_2_,
        b_grp0_a_0_,b_grp0_a_1_,b_grp0_a_2_,b_grp0_b_0_,b_grp0_b_1_,
        b_grp1_a_0_,b_grp1_a_1_,b_grp1_a_2_,b_grp1_b_0_,b_grp1_b_1_,
        b_grp2_a_0_,b_grp2_a_1_,b_grp2_a_2_,b_grp2_b_0_,b_grp2_b_1_,b_grp2_b_2_,
        ]

    graph = helper.make_graph(all_nodes, "TC_resnet_bb04p1_multiInOut",
            inputs=[helper.make_tensor_value_info(in_node_name1, TensorProto.FLOAT, list(in_shape1)),
                    helper.make_tensor_value_info(in_node_name2, TensorProto.FLOAT, list(in_shape2))],
            outputs=[helper.make_tensor_value_info(out_node_name1, TensorProto.FLOAT,list(out_shape1)),
                    helper.make_tensor_value_info(out_node_name2, TensorProto.FLOAT,list(out_shape2))],
            initializer=initializer
            )
    model = helper.make_model(graph,
                              producer_name='TC_resnet_bb04p1_multiInOut')
    with open(root_dir + 'TC_resnet_bb04p1_multiInOut.onnx', "wb") as of:
        of.write(model.SerializeToString())

    print("generate TC_resnet_bb04p1_multiInOut.onnx sucessfully!")


def test_model():
    mods = []
    pms = []
    input_dts = []
    data1 = np.round(np.random.rand(150, 128)*255).astype("float32")
    data2 = np.round(np.random.rand(150, 320)*255).astype("float32")

    file_name = root_dir + 'TC_resnet_bb04p1_multiInOut.onnx'
    onnx_model = onnx.load(file_name)
    shape_dict = {}
    shape_dict['in1'] = (1, 128)
    shape_dict['in2'] = (1, 64)
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    input_dt = {}
    input_dt['in1'] = witin.nd.array(data1)
    input_dt['in2'] = witin.nd.array(data2)
    mods.append(mod)
    pms.append(params)
    input_dts.append(input_dt)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mods, target='npu',
                        target_host='npu', params=pms,
                        input_data=input_dts,
                        chip = "BB04P1",
                        output_dir=build_dir,
                        optimize_method_config = ""
                        )
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input('in1', witin.nd.array(data1))
    m.set_input('in2', witin.nd.array(data2))
    # execute
    m.run()
    output = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    print(output[0].shape)
    print(output[1].shape)


def test_forward_multi_input_output():
    create_TC_resnet_onnx(root_dir)
    test_model()


if __name__ == '__main__':
    test_forward_multi_input_output()


