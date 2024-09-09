import numpy as np
import onnx
from onnx import helper, TensorProto, mapping
import pytest
import os
import datetime
import witin
from witin import *

target='npu'
target_host='npu'

os.system('mkdir -p ./model/test_conv2d')
os.system('rm ./model/test_conv2d/*')
os.system("rm ./output/output_test_onnx_op_conv2d* -rf")

def generate_conv2d_graph(cnn_input, weight, stride, padding):
    #prepare data
    weight_params=np.random.randint(-128, 127, size=weight, dtype=np.int32).astype(np.float32)
    #turn to tensor
    weight_input_tensor = helper.make_tensor('weight_input_tensor', data_type=onnx.TensorProto.FLOAT, dims=weight, vals=weight_params.flatten())

    #bias需要扩大，故*128

    bias_params=128 * np.random.randint(-127, 127, size=(weight[0],), dtype=np.int32).astype(np.float32)

    #turn to tensor
    bias_input_tensor = helper.make_tensor('bias_input_tensor', data_type=onnx.TensorProto.FLOAT, dims=(weight[0],), vals=bias_params.flatten())

    #prepare node
    onnx_conv2d = helper.make_node('Conv',
                                  inputs = ['conv2d_in', 'weight_input_tensor', 'bias_input_tensor'],
                                  outputs = ['conv2d_out'],
                                  strides = stride,
                                  kernel_shape = [weight[2], weight[3]],
                                  pads = padding,
                                  name = 'onnx_conv2d')

    scale_conv2d = helper.make_node('Scale',
                                    inputs=['conv2d_out'],
                                    outputs=['scale_conv2d_out'],
                                    scale = 0.0009765625)
    
    relu_conv2d = helper.make_node('Relu',
                                   inputs=['scale_conv2d_out'],
                                   #inputs=['conv2d_out'],
                                   outputs=['out'])
    #make graph -> add node to onnx framework

    in_shape=cnn_input
    o_shape_h=(cnn_input[2] - weight[2] + padding[0] +padding[2]) / stride[0] + 1
    o_shape_w=(cnn_input[3] - weight[3] + padding[1] + padding[3]) / stride[1] + 1
    out_shape=(cnn_input[0], weight[0], int(o_shape_h), int(o_shape_w))
    print(f"o_shape_h is {o_shape_h}, o_shape_w is {o_shape_w}")
    
    input=[helper.make_tensor_value_info('conv2d_in', onnx.TensorProto.FLOAT, list(in_shape))]
    output=[helper.make_tensor_value_info('out', onnx.TensorProto.FLOAT, list(out_shape))]
    #depends on how many nodes inserts into frame.
    nodes=[onnx_conv2d, scale_conv2d, relu_conv2d]
    #nodes=[onnx_conv2d]
    onnx_graph = helper.make_graph(nodes,
                                   name="onnx_conv2d_case",
                                   inputs=input,
                                   outputs=output,
                                   initializer=[weight_input_tensor, bias_input_tensor])
    
    onnx_conv2d_model = helper.make_model(onnx_graph, producer_name='cnn')
    #save model to file
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S_%f")
    model_path="./model/test_conv2d/conv2d" + timestampStr + ".onnx"
    print(model_path)
    with open(model_path, 'wb') as of:
        of.write(onnx_conv2d_model.SerializeToString())
    return model_path, timestampStr

#normal cases
c_input_weight_p=[((1,16,17,10),(4,16,6,9)),((1,16,17,10),(16,16,6,9)),((1,16,17,10),(1,16,6,9)),((1,16,17,10),(33,16,6,9)),
                  ((1,32,15,10),(32,32,3,2)),((1,32,15,10),(5,32,3,2)),((1,32,15,10),(33,32,3,2)),((1,32,15,10),(1,32,3,2)),
                  ((1,32,15,10),(32,32,3,3)),((1,32,15,10),(5,32,4,4)),((1,32,15,10),(33,32,5,5)),((1,32,15,10),(1,32,6,6)),
	              ((1,16,15,28),(4,16,6,9)),((1,16,15,28),(16,16,6,9)),((1,16,15,28),(1,16,6,9)),((1,16,15,28),(33,16,6,9)),]
                  
stride_p=((1,1),(2,2),(3,3),(7,7),(8,8),(13,13),(15,15))
padding_p=((0,0,0,0),(1,1,1,1),(0,2,0,0),(0,0,3,0),(0,0,0,10),(5,0,0,0),(2,2,1,1),(4,23,2,3),(2,4,6,8),(1,3,5,7),(0,0,2,3),(0,2,2,0),(8,0,3,0))

@pytest.mark.parametrize("i_w", c_input_weight_p)
@pytest.mark.parametrize("stride", stride_p)
@pytest.mark.parametrize("padding", padding_p)
@pytest.mark.test_regression_usual
def test_onnx_conv2d_regression(i_w, stride, padding):
    cnn_input=i_w[0]
    weight=i_w[1]
    print(f"test case are cnn_input:{cnn_input},weight:{weight},stride:{stride},padding:{padding}")
    #in size < 1024*64. weight size < 896
    if (cnn_input[1]*cnn_input[2]*cnn_input[3] < 2**10*64 and weight[1]*weight[2]*weight[3] < 896):
        #kernel need same, in > out
        if (cnn_input[2] >= weight[2] and cnn_input[3] >= weight[3]):
        #stride_H!=stride_W, must 1, and same i_h, f_h or same i_w, f_w
            o_shape_h=(cnn_input[2] - weight[2] + padding[0] +padding[2]) // stride[0] + 1
            o_shape_w=(cnn_input[3] - weight[3] + padding[1] + padding[3]) // stride[1] + 1
            if (cnn_input[1]*cnn_input[2]*cnn_input[3]+weight[0]*o_shape_h*o_shape_w > 128*1024):
                pytest.xfail("SKAM is too big")
            if (weight[0]*o_shape_h*o_shape_w > 64*1024):
                pytest.xfail("output channel memory is too big")
            if (stride[0]==stride[1]):
                onnx_conv2d_run(cnn_input, weight, stride, padding)
            elif(stride[0] == 1 and cnn_input[2] == weight[2]):
                onnx_conv2d_run(cnn_input, weight, stride, padding)
            elif (stride[1] == 1 and cnn_input[3] == weight[3]):
                onnx_conv2d_run(cnn_input, weight, stride, padding)
            else:
                pytest.xfail(reason='stride different')
        else:
            pytest.xfail(reason='f_w or f_h is bigger than i_w or i_h')
        
    else:
        pytest.xfail(reason='Unreasonable case')

# diff_stride cases
params=[((1,16,7,9),(16,16,7,9),(4,1),(0,0,0,0)),
        ((1,18,7,4),(12,18,5,4),(4,1),(0,0,0,0)),
        ((1,18,7,4),(12,18,6,4),(8,1),(0,0,0,0)),
        ((1,18,7,4),(30,18,4,4),(8,1),(0,0,0,0)),
        ((1,16,7,9),(16,16,7,8),(1,3),(0,0,0,0)),
        ((1,18,7,4),(12,18,7,3),(1,3),(0,0,0,0)),
        ((1,18,7,4),(12,18,7,4),(1,4),(0,0,0,0)),
        ((1,18,7,4),(30,18,7,2),(1,3),(0,0,0,0)),]

@pytest.mark.diff_stride
@pytest.mark.parametrize("cnn_input,weight,stride,padding", params)
def test_onnx_conv2d_diffStride(cnn_input, weight, stride, padding):
    print(f"test case are cnn_input:{cnn_input},weight:{weight},stride:{stride},padding:{padding}")
    #in size < 1024*64. weight size < 896
    if (cnn_input[1]*cnn_input[2]*cnn_input[3] < 2**10*64 and weight[1]*weight[2]*weight[3] < 896):
        #kernel need same, in > out
        if (cnn_input[1] == weight[1] ):
            if (cnn_input[2] >= weight[2] and cnn_input[3] >= weight[3]):
            #stride_H!=stride_W, must 1, and same i_h, f_h or same i_w, f_w
                if(stride[0] == 1 and cnn_input[2] == weight[2]):
                    onnx_conv2d_run(cnn_input, weight, stride, padding)
                elif (stride[1] == 1 and cnn_input[3] == weight[3]):
                    # begin to test
                    onnx_conv2d_run(cnn_input, weight, stride, padding)
            else:
                pytest.xfail(reason='f_w or f_h is bigger than i_w or i_h')
        else:
            pytest.xfail(reason='Unreasonable case')
    else:
        pytest.xfail(reason='Unreasonable case')


#smoke cases
cnn_params_smoke = [
 # same HW、KHKW, stride, padding
    ((1,3,7,7), (3,3,3,3), (1,1), (0,0,0,0)),
    
    # ((1,3,27,27), (3,3,5,5), (3,3), (0,1,1,0)),
    # ((1,3,27,27), (32,3,7,7), (5,5), (0,3,3,0)),
    # ((1,5,64,64), (16,5,13,13), (2,2), (0,5,5,0)),
    # ((1,5,64,64), (16,5,7,7), (7,7), (0,2,2,0)),
    # ((1,3,64,64), (32,3,15,15), (3,3), (0,3,3,0)),
    # ((1,5,114,114), (32,5,13,13), (5,5), (0,7,7,0)),

    # # diff HW KW KH stride padding
    # ((1,3,32,32),(16,3,5,3),(4,4),(0,0,0,0)),
    # ((1,3,32,32),(16,3,3,5),(4,4),(0,0,0,0)),
    # ((1,3,32,32),(16,3,3,5),(4,4),(0,1,1,0)),
    # ((1,3,32,32),(16,3,3,5),(4,4),(1,0,0,1)),
    # ((1,3,32,32),(16,3,3,5),(4,4),(1,1,1,1)),
    # ((1,3,32,16),(16,3,3,3),(4,4),(1,1,1,1)),
    # ((1,3,16,32),(16,3,3,3),(4,4),(1,1,1,1)),
    # ((1,3,32,16),(16,3,3,3),(4,4),(1,0,0,1)),
    # ((1,3,16,32),(16,3,3,3),(4,4),(0,1,1,0)),
    # ((1,3,16,3),(16,3,3,3),(4,1),(0,0,0,0)),
    # ((1,3,16,3),(16,3,5,3),(4,1),(0,0,0,0)),

    # ((1,32,3,128),(56,32,3,7),(1,4),(0,3,0,1)),
    # ((1,32,16,58),(32,32,4,5),(4,4),(1,2,0,1)),
    # ((1,32,16,58),(32,32,4,5),(4,4),(1,2,2,1)),
    # ((1,3,36,5),(16,3,5,5),(4,1),(2,0,2,0)),

    # ((1,50,38,3),(3,50,5,3),(5,1),(2,0,1,0)),
    # ((1,3,38,64),(3,3,5,3),(5,5),(2,1,1,0)),

    # ((1,72,16,2),(112,72,5,2),(2,1),(0,0,2,0)),
    # ((1,18,16,32),(112,18,5,2),(2,2),(0,0,2,1)),
    # ((1,112,8,2),(112,112,3,2),(2,2),(0,0,1,0)),
    # ((1,16,64,16),(1,16,7,7),(2,2),(0,1,1,0)),  
]

@pytest.mark.smoke

@pytest.mark.parametrize("cnn_input,weight,stride,padding", cnn_params_smoke)
def test_cnn_model_exception_from_params(cnn_input,weight,stride,padding):
    print(f"test case are cnn_input:{cnn_input},weight:{weight},stride:{stride},padding:{padding}")
    onnx_conv2d_run(cnn_input, weight, stride, padding)


def onnx_conv2d_run(cnn_input, weight, stride, padding):
    print("*************")
    print(f"cnn_input={cnn_input}")
    print(f"weight={weight}")
    print(f"stride={stride}")
    print(f"padding={padding}")
    print("******************")

    model_path, timestampStr = generate_conv2d_Graph(cnn_input, weight, stride, padding)
    onnx_conv2d_model=onnx.load(model_path, timestampStr)

    # create data
    data = np.round(np.random.rand(11,cnn_input[1],cnn_input[2],cnn_input[3])*255).astype("float32")
    print(data)
    #test_onnx_conv2d

    shape_dict={}
    shape_dict['conv2d_in']=cnn_input
    #trans model to mod, params
    mod, params = witin_frontend.frontend.from_onnx(onnx_conv2d_model, shape_dict)
    input_dt = {}
    input_dt['input_data'] = witin.nd.array(data)

    #optimize file
    opt_config = ''
    #output file   
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestampStr
 
    #build
    with witin.transform.PassContext(opt_level=3):
        _,_,_,npu_graph = witin_frontend.build_module.build(mod,
                                                            target='npu',
                                                            target_host='npu',
                                                            params=params,
                                                            input_data=input_dt,
                                                            chip='BB04P1',
                                                            output_dir=build_dir,
                                                            optimize_method_config=opt_config)

    #build runtime engine
    from tvm.contrib import npu_graph_runtime
    m=npu_graph_runtime.create(npu_graph, 'BB04P1', build_dir)
    m.set_input('in1', witin.nd.array(data))
    m.run()


cnn_params=[
    ((1,16,27,27), (3,16,5,5), (3,3), (0,1,1,0)),
    ((1,89,98,7), (3,89,3,2), (4,4), (1,1,1,1)),

    #big channle
    ((1,32,10,2),(1,32,4,5), (3,3),(1,10,1,1)),
    ((1,64,2,2),(3,64,2,2),(2,2),(2,4,2,3)),
    ((1,64,1,4),(4,64,1,3),(10,10), (6,3,1,1)),
    ((1,1024,1,1),(4,1024,1,1),(2,2), (2,5,6,1)),

    #big channle_out
    ((1,16,27,27), (30,16,5,5), (3,3), (0,1,1,0)),

    #different stride
    ((1,16,27,27), (3,16,5,27), (5,1), (0,0,0,0)),

    ((1,16,27,27), (3,16,27,8), (1,4), (0,1,0,0)),
    #padding
]

@pytest.mark.repeat(10)
def test_onnx_conv2d_repeat():
    count=0
    def set_conv_inputDim():
        while True:
            channel=np.random.randint(1, 64)
            h=np.random.randint(1, 102)
            w=np.random.randint(1, 102)
            cnn_input_dim=(1, channel, h, w)
            if (channel * h * w < 65535):
                print("CNN_INPUT_DIM is OK")
                return cnn_input_dim
            else:
                print("CNN_INPUT_DIM is bigger than 65535, continue")
                continue

    def set_conv_weight_dim(channel, h, w):
        while True:
            output=np.random.randint(1,8)
            input=channel
            f_h=np.random.randint(1, h+1)
            f_w=np.random.randint(1, w+1)
            if (input*f_h*f_w < 896):
                weight_dim=(output, input, f_h, f_w)
                return weight_dim
    while count < 5:
        cnn_input=set_conv_inputDim()
        weight=set_conv_weight_dim(cnn_input[1],cnn_input[2],cnn_input[3])
        stride=np.random.randint(1,16)
        stride=(stride, stride)
        padding=(np.random.randint(0,32),np.random.randint(0,32),np.random.randint(0,32),np.random.randint(0,32))

        o_shape_h=(cnn_input[2] - weight[2] + padding[0] +padding[2]) // stride[0] + 1
        o_shape_w=(cnn_input[3] - weight[3] + padding[1] + padding[3]) // stride[1] + 1
        print(f"o_shape_h is {o_shape_h}, o_shape_w is {o_shape_w}")
        if (cnn_input[1]*cnn_input[2]*cnn_input[3]+weight[0]*o_shape_h*o_shape_w > 65535):
            print("SKAM is too big")
            continue
        if (weight[0]*o_shape_h*o_shape_w > 65535):
            print("output channel memory is too big")
        else:
            print(f"test case are cnn_input:{cnn_input},weight:{weight},stride:{stride},padding:{padding}")
            count=count+1
            onnx_conv2d_run(cnn_input, weight, stride, padding)

if __name__ == '__main__':
    #cnn_input=(1, 7, 30, 140)
    #weight=(3, 7, 3, 39)
    #stride=(5, 5)
    #padding=(7, 17, 29, 16)
    cnn_input=(1, 4, 10,10)
    weight=(10, 4, 7, 10)
    stride=(1, 1)
    padding=(8, 0, 3, 0)

    onnx_conv2d_run(cnn_input, weight, stride, padding)
    #test OPT
    #onnx_conv2d_repeated(20)

