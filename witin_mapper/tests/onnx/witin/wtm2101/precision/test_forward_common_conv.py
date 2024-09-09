import onnx
import numpy as np
import torch
import witin
from witin import *


def load_model(model_path, num_layer):
    net_w = []
    net_b = []

    for i in range(num_layer):
        name_w = model_path + '//w' + str(i) + '.npy'
        name_b = model_path + '//b' + str(i) + '.npy'

        w = np.load(name_w).astype(np.float32)
        b = np.load(name_b).astype(np.float32)

        net_w.append(w)
        net_b.append(b)

    return net_w, net_b

def make_conv2d_node(inp, weight, bias, node_name,kernel_shape, strides, pads, G_scale, actv_type):
    
    make_cnn_node = []    
    node_cnn = onnx.helper.make_node("Conv",inputs=[inp,weight, bias], outputs=[node_name + "-output"],
                                             kernel_shape=kernel_shape, strides=strides, pads=pads, name=node_name)

    node_scale = onnx.helper.make_node("Scale", [node_name + "-output"], [node_name + "-scale-output"], scale=1.0/G_scale)
    make_cnn_node = [node_cnn] + [node_scale]
    out_name = node_name + "-scale-output"

    if actv_type == "RELU":
        
        node_actv = onnx.helper.make_node('Relu', [node_name + "-scale-output"], [node_name + "-relu-output"])
        out_name = node_name + "-relu-output"
        make_cnn_node = make_cnn_node + [node_actv]
        
        return make_cnn_node, out_name
    elif actv_type == "NONE" or actv_type == "PASS":
        # out_name = conv_out
        return make_cnn_node, out_name
    else:
        print("ERR actv type:",actv_type)
        exit(0)

    # return make_cnn_node, 
    

def generate_model():
    layer_weight, layer_bias = load_model('./model/pipeline/cnn_model/params/', 5)
    G_scale = [512, 256, 256, 256, 1024]
    in_shape = [1, 1, 128, 128]
    out_shape = [1, 48, 4, 4]

    kernel_shape = [
        [5, 5],
        [5, 5],
        [3, 3],
        [3, 3],
        [3, 3]
    ]
    pads = [
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]

    strides = [
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
    ]
    nodes_list =[]
    initializers_list = []
    cnn_out_name = 'in'

    for i in range(5):

        key = 'conv' + str(i + 1)
        weight_key = key + ".weight"
        bias_key = key + ".bias"
        weight_tensor = layer_weight[i]
        bias_tensor = layer_bias[i]

        
        layer_w = onnx.helper.make_tensor(weight_key, data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(weight_tensor), vals=weight_tensor.flatten())
        layer_b = onnx.helper.make_tensor(bias_key, data_type=onnx.TensorProto.FLOAT,
                                              dims=np.shape(bias_tensor), vals=bias_tensor.flatten())
        
        if(i<4):
            cnn_node, cnn_out_name = make_conv2d_node(cnn_out_name, weight_key,bias_key, key, kernel_shape[i],strides[i],pads[i],G_scale[i],'RELU')
        else:
            cnn_node, cnn_out_name = make_conv2d_node(cnn_out_name, weight_key,bias_key, key, kernel_shape[i],strides[i],pads[i],G_scale[i],'NONE')
        
        nodes_list  = nodes_list + cnn_node
        initializers_list = initializers_list + [layer_w, layer_b]


    graph = onnx.helper.make_graph(nodes_list,"conv-net", 
                inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(in_shape))],
                outputs=[onnx.helper.make_tensor_value_info(cnn_out_name, onnx.TensorProto.FLOAT, list(out_shape))],
                initializer=initializers_list)

    model = onnx.helper.make_model(graph, producer_name="conv-cnn-model")
    with open('./model/pipeline/cnn_model/conv-cnn-model.onnx', "wb") as f:
        f.write(model.SerializeToString())
    print("Generate conv-cnn-model sucessfully!")

       


def run_models():
    target = 'npu'
    target_host = 'npu'

    input_shape_map = [1, 1, 128, 128]

    shape_dict = {}
    input_dt = {}

    
    onnx_model = onnx.load("./model/pipeline/cnn_model/conv-cnn-model.onnx")
    shape_dict['in'] = input_shape_map
    
    mod, params = witin_frontend.frontend.from_onnx(onnx_model, shape_dict)
    data = torch.load('./model/pipeline/cnn_model/input/calib.pt_input.pt')
    
    data = np.transpose(data,(0,3,2,1))
    data = data[:20,:,:,:]
    # data = data.astype('float32')
    input_dt['input_data'] = witin.nd.array(data)
    opt_config = "./model/pipeline/cnn_model/opti/bset_opti.protobuf"

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
    
     # # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)

    m.set_input(0, witin.nd.array(data))
    # m.set_input(1, witin.nd.array(data_conv_right))
    # m.set_input(2, witin.nd.array(data1))
    # m.set_input(3, witin.nd.array(data2))

    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    print(witin_output_list[0].shape)

        
            
if __name__ == '__main__':
    generate_model()
    run_models()
