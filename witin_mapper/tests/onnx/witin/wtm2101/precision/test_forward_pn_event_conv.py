
import onnx
import numpy as np
import witin
from witin import *
import os

np.random.seed(0)

onnx_dir = './model/pipeline/events/'
if not os.path.exists(onnx_dir):
    os.mkdir(onnx_dir)


def generate_model():

    # layer 1
    conv1_w = np.random.randint(-127,127,size=(2, 2, 3, 3), dtype='int8').astype('float32')
    conv1_b = np.random.randint(-127,127,size=(2), dtype='int8').astype('float32')
    conv1_b = conv1_b * 128

    conv2_w = np.random.randint(-127,127,size=(2, 4, 3, 3), dtype='int8').astype('float32')
    conv2_b = np.random.randint(-127,127,size=(2), dtype='int8').astype('float32')
    conv2_b = conv2_b * 128


    conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv1_w),
                                                  vals=conv1_w.flatten())
    conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor',
                                                  data_type=onnx.TensorProto.FLOAT,
                                                  dims=np.shape(conv1_b),
                                                  vals=conv1_b.flatten())

    conv1_node = onnx.helper.make_node("Conv",
                                            inputs=['in', 'conv1_w_tensor', 'conv1_b_tensor'],
                                            outputs=['conv1'],
                                            kernel_shape=[3, 3],
                                            strides=[1, 1],
                                            pads=[1, 1, 1, 1],
                                            name="conv1")
    conv1_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv1'],
                                                  outputs=['conv1_scale'],
                                                  scale=1.0 / 1024)
    conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['out'])

    node1 = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
    initializer1 = [conv1_w_tensor, conv1_b_tensor]   

   
    
    nodes = node1 
    initializers =  initializer1 

    input_shape = (1, 2, 32, 32)
    output_shape = (1, 4, 32, 32)

    dccrn_graph = onnx.helper.make_graph(
        nodes,
        "pn_event_model",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input_shape))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape))],
        initializer=initializers)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='pn_event_model')

    with open(onnx_dir + '/pn_event_conv_model.onnx', "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate pn_event_model sucessfully!")


def run_model():
    generate_model()
    mods = []
    params = []
    input_datas = []

    add_model = onnx.load(onnx_dir + "pn_event_conv_model.onnx")
    
    shape_dict_add = {}
    shape_dict_add['in'] = (1, 2,32, 32)  #NCHW

    mod_add, param_add = witin_frontend.frontend.from_onnx(add_model, shape_dict_add)
    mods.append(mod_add)
    params.append(param_add)

    data_add =  np.random.randint(-128,127,size=(100, 32, 32, 2), dtype='int8').astype('float32')
    
    input_dt_add = {}
    input_dt_add['in'] = witin.nd.array(data_add)

    input_datas.append(input_dt_add)

    opt_config = './model/pipeline/events/pn_opti_conv.protobuf'
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    #build
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mods,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=params,
                                                               input_data=input_datas,
                                                               chip='BB04P1',
                                                               output_dir=build_dir,
                                                               optimize_method_config=opt_config)
    # # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)

    m.set_input(0, witin.nd.array(data_add))
    
    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    print(witin_output_list[0].shape)




if __name__ == "__main__":
    generate_model()
    run_model()