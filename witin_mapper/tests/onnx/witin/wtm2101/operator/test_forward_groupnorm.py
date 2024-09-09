import onnx
import numpy as np
import witin
from witin import *
import os

np.random.seed(0)

onnx_dir = './model/inner/'
if not os.path.exists(onnx_dir):
    os.mkdir(onnx_dir)

def generate_model():
    # layer 1
    conv1_w = np.random.randint(-127,127,size=(4, 3, 3, 3), dtype='int8').astype('float32')
    conv1_b = np.random.randint(-127,127,size=(4), dtype='int8').astype('float32')
    conv1_b = conv1_b * 128

    conv2_w = np.random.randint(-127,127,size=(4, 4, 3, 3), dtype='int8').astype('float32')
    conv2_b = np.random.randint(-127,127,size=(4), dtype='int8').astype('float32')
    conv2_b = conv2_b * 128

    conv1_w_tensor = onnx.helper.make_tensor('conv1_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_w),
                                             vals=conv1_w.flatten())
    conv1_b_tensor = onnx.helper.make_tensor('conv1_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv1_b),
                                             vals=conv1_b.flatten())

    conv2_w_tensor = onnx.helper.make_tensor('conv2_w_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_w),
                                             vals=conv2_w.flatten())
    conv2_b_tensor = onnx.helper.make_tensor('conv2_b_tensor',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=np.shape(conv2_b),
                                             vals=conv2_b.flatten())

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
    conv1_relu_node = onnx.helper.make_node('Relu', ['conv1_scale'], ['conv1_relu'])

    node1 = [conv1_node] + [conv1_scale_node] + [conv1_relu_node]
    initializer1 = [conv1_w_tensor, conv1_b_tensor]
    
    alpha = np.array([0.5, 0.6, 0.25, 0.8])
    beta = np.array([0.75, 0.3, 0.65, 0.5])
    alpha_tensor = onnx.helper.make_tensor('alpha_tensor',
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=np.shape(alpha),
                                           vals=alpha.flatten())
    beta_tensor = onnx.helper.make_tensor('beta_tensor',
                                           data_type=onnx.TensorProto.FLOAT,
                                           dims=np.shape(beta),
                                           vals=beta.flatten())
    group_norm_node = onnx.helper.make_node("GroupNorm",
                                            inputs=["conv1_relu", 'alpha_tensor', 'beta_tensor'],
                                            outputs=["group_norm_out"],
                                            num_groups=2,
                                            num_channels=4,
                                            scale_out=0.5,
                                            name="groupNorm1")
    group_norm_initializer = [alpha_tensor, beta_tensor]
    
    group_norm_relu_node = onnx.helper.make_node('Relu', ['group_norm_out'], ['group_norm_relu'])

    conv2_node = onnx.helper.make_node("Conv",
                                       inputs=['group_norm_relu', 'conv2_w_tensor', 'conv2_b_tensor'],
                                       outputs=['conv2'],
                                       kernel_shape=[3, 3],
                                       strides=[1, 1],
                                       pads=[1, 1, 1, 1],
                                       name="conv2")
    conv2_scale_node = onnx.helper.make_node("Scale",
                                             inputs=['conv2'],
                                             outputs=['conv2_scale'],
                                             scale=1.0 / 1024)
    conv2_relu_node = onnx.helper.make_node('Relu', ['conv2_scale'], ['out'])

    node2 = [conv2_node] + [conv2_scale_node] + [conv2_relu_node]
    initializer2 = [conv2_w_tensor, conv2_b_tensor]

    nodes = node1 + [group_norm_node] + [group_norm_relu_node] + node2 
    initializers = initializer1 + group_norm_initializer + initializer2

    input1_shape = (1, 3, 28, 28)
    output_shape = (1, 4, 28, 28)

    dccrn_graph = onnx.helper.make_graph(
        nodes,
        "group_norm_test",
        inputs=[onnx.helper.make_tensor_value_info("in", onnx.TensorProto.FLOAT, list(input1_shape))],
        outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, list(output_shape))],
        initializer=initializers)
    dccrn_model = onnx.helper.make_model(dccrn_graph, producer_name='group_norm_test')

    with open(onnx_dir + '/group_norm_test.onnx', "wb") as f:
        f.write(dccrn_model.SerializeToString())
    print("Generate group_norm_test sucessfully!")


def run_model():
    onnx_model = onnx.load(onnx_dir + "group_norm_test.onnx")
    shape_dict_add = {}
    shape_dict_add['in'] = (1, 3, 28, 28)  #NCHW
    mod, param = witin_frontend.frontend.from_onnx(onnx_model, shape_dict_add)    
    data =  np.random.randint(0,127,size=(11, 28, 28, 3), dtype='int8').astype('float32')
    input_dt = {}
    input_dt['in'] = witin.nd.array(data)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr

    # build
    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(mod,
                                                               target='npu',
                                                               target_host='npu',
                                                               params=param,
                                                               input_data=input_dt,
                                                               chip='BB04P1',
                                                               output_dir=build_dir,
                                                               optimize_method_config="")
    # execute
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    m.set_input("in", witin.nd.array(data))
    m.run()

def test_forward_groupNorm():
    generate_model()
    run_model()

if __name__ == "__main__":
    test_forward_groupNorm()
