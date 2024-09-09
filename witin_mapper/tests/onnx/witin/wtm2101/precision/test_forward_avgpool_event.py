
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

    #layer 1
    conv1_w = np.random.randint(-127,127,size=(1, 4, 3, 3), dtype='int8').astype('float32')
    conv1_b = np.random.randint(-127,127,size=(1), dtype='int8').astype('float32')
    conv1_b = conv1_b * 128

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
                                            pads=[0, 0, 0, 0],
                                            name="conv_left_prescale_1")
    conv1_scale_node = onnx.helper.make_node("Scale",
                                                  inputs=['conv1'],
                                                  outputs=['conv1_scale'],
                                                  scale=1.0 / 1024)
    conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['conv1_relu_op'])

    node1 = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
    initializer1 = [conv1_w_tensor, conv1_b_tensor]

    ave_pooling_node = onnx.helper.make_node(
        "AveragePool",
        inputs=["conv1_relu_op"],
        outputs=["out"],
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],  #[top,left,bottom,right]
        strides=[1, 1],
        scale_in=0.0625,
        scale_out=0.0625)

    
    nodes = node1 + [ave_pooling_node]
    initializers =  initializer1

    input_shape = (1, 4, 18, 18)
    output_shape = (1, 1, 2, 2)

    dccrn_graph = onnx.helper.make_graph(
        nodes,
        "avgpool_event_model",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input_shape))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape))],
        initializer=initializers)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='avgpool_event_model')

    with open(onnx_dir + '/avgpool_event_model.onnx', "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate avgpool_event_model sucessfully!")


def run_model():
    generate_model()
    
    mods = []
    params = []
    input_datas = []
    avg_pool_model = onnx.load(onnx_dir + "avgpool_event_model.onnx")
    

    shape_dict_avg_pool = {}
    shape_dict_avg_pool['in'] = (1, 4, 18, 18)  #NCHW

    mod_avg_pool, param_avg_pool = witin_frontend.frontend.from_onnx(avg_pool_model, shape_dict_avg_pool)
    data_avg_pool =  np.random.randint(0,127,size=(100, 4,18, 18), dtype='int8').astype('float32')
    
    data_avg_pool = np.transpose(data_avg_pool,(0,3,2,1)) #NWHC
    input_dt_avg_pool = {}
    input_dt_avg_pool['in'] = witin.nd.array(data_avg_pool)

    mods.append(mod_avg_pool)
    params.append(param_avg_pool)
    input_datas.append(input_dt_avg_pool)

    
    
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
                                                               optimize_method_config='')
    # # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)

    m.set_input(0, witin.nd.array(data_avg_pool))
    # m.set_input(1, witin.nd.array(data_conv_right))
    # m.set_input(2, witin.nd.array(data1))
    # m.set_input(3, witin.nd.array(data2))

    m.run()
    witin_output_list = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    print(witin_output_list[0].shape)




if __name__ == "__main__":
    run_model()