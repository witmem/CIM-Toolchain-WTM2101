# CIM Tool Chain -WITMEM 语音芯片工具链用户使用说明

[***\*1\**** ***\*环境配置\****	](#_Toc30969 )

[1.1 简介	](#_Toc15889 )

[1.2 前端支持	](#_Toc20228 )

[1.3 版本信息	](#_Toc20781 )

[1.4 安装方式	](#_Toc27287 )

[1.4.1 Docker方式	](#_Toc16853 )

[***\*2\**** ***\*工程说明\****	](#_Toc2074 )

[***\*3\**** ***\*使用\*******\*流程\****	](#_Toc28158 )

[3.1 模型导入	](#_Toc31789 )

[3.2 编译build	](#_Toc5182 )

[3.3 运行run	](#_Toc15772 )

[3.4 与tensorflow结果进行比对	](#_Toc126 )

[3.5 与pytorch结果进行比对	](#_Toc32723 )

[3.6 多网络	](#_Toc23316 )

[3.7 优化策略	](#_Toc27137 )

[3.7.1 输入数据优化	](#_Toc5597 )

[3.7.2 权重数据优化	](#_Toc6223 )

[3.7.3 权重数据复制	](#_Toc3864 )

[3.7.4 稀疏矩阵优化	](#_Toc16341 )

[3.7.5 输出放大优化	](#_Toc4 )

[3.7.6 卷积优化	](#_Toc21301 )

[3.8 特殊配置环境变量	](#_Toc15767 )

[3.8.1 Array分配	](#_Toc13568 )

[3.8.2 开启FIFO	](#_Toc17352 )

[3.8.3 设置校准数据数量	](#_Toc11859 )

[***\*4\**** ***\*Onnx模型\****	](#_Toc2110 )

[4.1 创建Tensor	](#_Toc21323 )

[4.2 onnx算子	](#_Toc14180 )

[4.2.1 Gemm	](#_Toc3147 )

[4.2.2 Tdnn	](#_Toc22320 )

[4.2.3 Lstm	](#_Toc29216 )

[4.2.4 Gru	](#_Toc25894 )

[4.2.5 Scale	](#_Toc23682 )

[4.2.6 Relu	](#_Toc29014 )

[4.2.7 ActLut	](#_Toc15761 )

[4.2.8 Mul	](#_Toc1640 )

[4.2.9 Add	](#_Toc31289 )

[4.2.10 Averagepool	](#_Toc28406 )

[4.2.11 Slice	](#_Toc11933 )

[4.2.12 Concat	](#_Toc10946 )

[4.2.13 Conv	](#_Toc1786 )

[4.2.14 ConvTranspose	](#_Toc14984 )

[4.2.15 Gru2Array	](#_Toc22702 )

[4.3 创建graph	](#_Toc24446 )

[4.4 保存graph	](#_Toc17462 )

[4.5 导入模型	](#_Toc5061 )



# 1  **环境配置**

## 1.1  **简介**

本文主要介绍语音芯片工具链witin_mapper的使用。该工具链主要针对各种神经网络设计，用于将神经网络运算map到WTM1001/WTM2101芯片上，生成相应的配置文件和烧录文件。其中烧录文件需使用专门的工具烧写到WTM1001和WTM2101的ARRAY阵列中，配置文件集成到sdk中使用。

## 1.2  **前端支持**

前端转换支持tensorflow，onnx，pytorch深度学习框架。

## 1.3  **版本信息**

Ubuntu:18.04.5

Cmake 3.16（18.04默认是3.10版本，需升级）

## 1.4  **安装方式**

### 1.4.1  **Docker方式**

方式一：

预先在windows系统或者Liunx系统下安装docker环境。使用Dockerfile构建镜像

1构建镜像：docker build -t witin/witin_tc_wtm2101_vX.X.X .（Dockerfile目录下运行）

2.查看镜像：docker images

3.创建容器：

docker run -it --name xxx witin_tc witin_toolchain_wtm2101:vX.X.X /bin/bash

4.查看容器：docker ps -a

5重新进入容器：1)docker start container_id ; 2)docker attach container_id 

6退出容器：exit

7将本地文件（或文件夹）上传至容器：

docker cp 本地文件路径 container_id:/workspace/witin_mapper

8将容器文件（或文件夹）下载至本地：

docker cp container_id:/workspace/witin_mapper 本地文件路径

\10. 使用witin mapper生成的网络的映射配置，一般存放于witin_mapper/output路径下，可以使用docker cp将其从docker容器内拷贝到本地，以供后续步骤的烧录工具和精度分析工具使用。 

 

方式二：

1.拉取镜像：

docker pull witin/witin_toolchain_wtm2101:vX.X.X

2.查看镜像：

docker images

3.创建容器：

docker run -it --name xxx witin/witin_toolchain_wtm2101:vX.X.X /bin/bash

4.查看容器：

docker ps -a

5.启动容器

docker start container_id

docker attach container_id

6.进入容器,默认进入workspace路径

7.如若不用，退出容器，执行exit即可。

说明：2024年6月份后，中国大陆境内使用方式2 获取docker镜像可能会失败。



# 2  **工程说明**

build：存放安装脚本，output输出文件目录以及WTM1001/WTM2101模拟器

model：存放示例的模型文件

python:  存放整个工程的python文件

tests：部分测试文件

run.py：利用python3命令执行部分tests文件中的测试文件

README.md：对整个witin_mapper工程进行说明



# 3  **使用流程**

## 3.1  **模型导入**

以tensorflow为例，调用接口函数如下：

1．首先调用tensorflow接口函数来读取pb文件得到graph_def

```
with gfile.FastGFile(model_path, ‘rb’) as f:

graph_def=tf.GraphDef()

graph_def.ParseFromString(f.read())

sess.graph.as_default()

tf.import_graph_def(graph_def, name=’’)
```

2．调用前端转换函数

witin_frontend.frontend.from_tensorflow(graph_def, shape_dict)

其中graph_def为读取pb文件得到的文件，shape_dict为输入节点的名字和形状，字典形式。调用该函数后得到mod和params两个参数，其中mod为图的结构中间表示，params为用到的参数。

## 3.2  **编译build**

该步骤通过编译得到WTM1001/WTM2101上运行的配置文件，然后将该配置文件集成到sdk然后下载到芯片中并运行，即可完成整个网络的在WTM1001/WTM2101芯片上的运行。

首先配置target为npu，target_host为npu，然后调用如下接口函数：

```
with witin.transform.PassContext(opt_level=3):

​    _, _, _, npu_graph = witin_frontend.build_module.build(

​      mod,

​      target='npu',

​      target_host='npu',

​      params=params,

​      input_data=input_dt,

​      chip="BB04P1",

​      optimize_method_config=optimize_method_config,

​      output_dir=output_dir,

​      array_distribute=array_distribute)
```

其中：

mod为2.1中模型导入后得到的mod表示

target和target_host为npu

params为模型导入后得到的params文件

input_data参数为witin阵列Array校准需要用到的输入数据，推荐至少为100组输入。

chip为使用的芯片版本。

optimize_method_config为优化配置文件，默认为空字符串””。

output_dir为输出文件的保存路径，默认为”./build/output”。

array_distribute为指定权重布局分配的文件，默认为空字符串””，当指定此文件时，会最小化build模型，只保留一些必要文件。

在调用build后，在工程的build的output文件夹下会生成如下几个文件和文件夹：

1．net_config.h/net_config.c文件，配置文件，需集成到witin的sdk中使用

2．map文件夹，存放烧写array需要用到的expected_in.bin，expected_out.bin，layers.txt以及map.csv文件。

3．simulator_input文件夹：存放模拟器仿真输入文件

4．simulator_output文件夹：存放运行仿真后的输出文件

5．params文件夹：存放各层映射的权重文件，用于debug

6．layer_debug文件夹：存放各层仿真器的输出输出文件，用于debug及精度调试工具。

## 3.3  **运行run**

该部分主要提供build之后在witin的模拟器进行运行的接口。

build完成后会在build的output文件下生成配置文件和烧写文件。同时会返回npu_graph到python端。该npu_graph可用于创建runtime的module，然后打注输入并运行，最后得到网络运行的结果。具体函数接口如下：

• m=npu_graph_runtime.create(npu_graph, “BB04P”)，创建运行时module m。

• m.set_input(‘in1’, wintin.nd.array(data))，其中data为np.array的列表，‘in1’为input placeholder的名字。

• m.run()，在模拟器上运行。

• m.get_output(0)，得到在模拟器上运行的结果。

## 3.4  **与tensorflow结果进行比对**

设置阈值threshold，调用compare_tensorflow_with_npu函数比较tensorflow的运行结果tf_out_data和witin npu的运行结果。

## 3.5  **与pytorch结果进行比对**

(1) 构建网络模型

```
class Net(nn.Module):

  def __init__(self):

​    super(Net, self).__init__()

​    self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1)

​    self.conv2 = nn.Conv2d(10, 14, kernel_size=5, stride=3)

​    self.fc = nn.Linear(896, 10)

  def forward(self, x):

​    in_size = x.size(0)

​    x = F.relu(self.conv1(x))

​    x = F.relu(self.conv2(x))

​    x = x.view(in_size, -1)  # flatten the tensor

​    x = self.fc(x)

​    return x
```

(2) 传入模型和输入，得到script_model。

```
input_shape = [1, 1, 28, 28]

shape_list = [(input_name, input_shape)]

model.load_state_dict(model_dict)

data = torch.rand(input_shape)

scripted_model = torch.jit.trace(model, data).eval()
```

(3) 调用前端转换函数，得到mod和params

```
mod, params = witin_frontend.frontend.from_pytorch(scripted_model, 										shape_list)
```

调用该函数后得到mod和params两个参数，其中mod为图的结构中间表示，params为用到的参数。

(4) 编译build

```
with witin.transform.PassContext(opt_level=3):

​    _, _, _, npu_graph = witin_frontend.build_module.build(

​                      mod,

​                      target=target,

​                      target_host=target_host,

​                      params=params,

​                      input_data=input_dt,

​                      chip=chip)
```

(5) 运行run，和模拟器出来的结果进行对比

```
m = npu_graph_runtime.create(npu_graph, chip)

m.set_input(input_name, witin.nd.array(data))

m.run()

witin_output_list = [m.get_output(i).asnumpy() for i in range(1)]
```



## 3.6  **多网络**

支持多网络，详细请参照工程：

witin_mapper/tests/onnx/witin/wtm2101/operator/test_forward_multinet.py

多网络set_input()的时候，要指定网络输入节点的名称或网络的序号,例如：

​		# 1.已知输入节点的名称

```
	m.set_input('fingerprint_input', witin.nd.array(data1))

​		m.set_input(fingerprint_input', witin.nd.array(data2))
```

​		# 2.不知道节点名称，按照Graph的顺序

```
	m.set_input(0, witin.nd.array(data1))

​		m.set_input(1, witin.nd.array(data2))
```

单网络可任意设置节点名称。

## 3.7  **优化策略**

为了优化array的计算精度，可以设置不同层的优化策略。

现在有支持三种优化策略：输入数据优化、权重数据优化、权重数据复制。

详细定义信息如下：

```
syntax="proto3";

enum SpMatMode{

​	SIM_MODE = 0;

​	MIX_MODE = 1;

​	DIG_MODE = 2;}

enum ArrayRegion {

​	ANY_REGION = 0;

​	UP_REGION = 1;

​	DOWN_REGION = 2;}

message LayerOptiConfig{

​	string name = 1;

​	message InputOpti{

​		int32 shift = 1; // 0 left 1 right

​		int32 num = 2;

​		int32 front_scale = 3;

​		int32 signed_input = 4;}

​	message WeightOpti{

​		int32 shift = 1;

​		int32 num = 2;

​		ArrayRegion region = 3;}

​	message DoubleWeightOpti{

​		int32 multiple = 1;}

​	message SparseMatrixOpti{

​		SpMatMode sp_mode = 1;

​		int32 shift = 2;}

​	message ArrayOutputOpti{

​		int32 magnify = 1;

​		int32 repeat = 2;}

​	message ConvOpti{

​		int32 multi_point = 1;

​		bool tdnn_mode = 2;

​		bool conv2DToConv1D_enable = 3;

​		int32 fout_burst_point = 4;}

​	repeated InputOpti inputOpti = 2;

​	repeated WeightOpti weightOpti = 3;

​	repeated DoubleWeightOpti doubleWeightOpti = 4;

​	repeated SparseMatrixOpti sparseMatrixOpti = 5;

​	repeated ArrayOutputOpti arrayOutputOpti = 6;

​	repeated ConvOpti convOpti = 7;}

message OptimizeConfig{

​	repeated LayerOptiConfig layerOptiConfig = 1;}
```

 

其中，name为要优化的层名，为字符串；InputOpti为输入数据优化；WeightOpti为权重数据优化；DoubleWeightOpti为权重数据复制；SparseMatrixOpti为稀疏矩阵模式，ArrayOutputOpti为输出放大优化模式，ConvOpti为卷积优化模型。

详细参考工程：

witin_mapper/tests/python/frontend/onnx/witin/wtm21011/optimizer/test_layer_optimize_method.py

在build中指定optimize_method_config参数，为优化策略配置文件。

### 3.7.1  **输入数据优化**

​	message InputOpti{

​		int32 shift = 1; 

​		int32 num = 2;

​		int32 front_scale = 3;

​		int32 signed_input = 4;

​	}

• 当需要此层输入数据数据扩大时：

1．设置shift为1时且num>1，会扩大本层输入的输入，扩大的倍数为2^num。注意：当此层为首层时，需要人工将数据扩大输入，非首层时，要求前一层必须有Relu。

2．设置front_scale>0时，会将本层的上层scale值缩小，缩小倍数为2^front_scale。注意：此优化不可用于首层。

• 当首层输入为有符号数int8时：

设置signed_input = N > 0。注意：对于有符号输入X，需要在程序输入的时候手动将输入改变：

1．X1 = X[0:N]

2．X2 = X[N:end] + 128

3．X3 = -X[0:N]

4．将X1,X2,X3拼接一起

5．X' = np.concatenate((data0, data1, data2), axis = 1)

6．将负数变为0

7．X'[X'<0] = 0

8．得到新的输入X'(uint8)

参考示例程序为：

witin_mapper/tests/python/frontend/onnx/witin/wtm21011/optimizer/test_layer_optimize_method_signed_in.py

注意：v001.000.024及以前版本，当首层输入数据为有符号数int8时，需要按照上述过程处理，v001.000.025及以后版本支持首层输入数据为有符号数int8类型。

### 3.7.2  **权重数据优化**

​	message WeightOpti{

​		int32 shift = 1;

​		int32 num = 2;

​		ArrayRegion region = 3;

​	}

当需要此层权重数据扩大时：

设置shift为1时且num>1，会扩大本层输入的输入，扩大的倍数为2^num。

例如，权重为2*3的矩阵[[1,2,3][4,5,6]]，设置

weightOpti {

   shift : 1

   num : 1

 }

权重变为[[2,4,6][8,10,12]]。

region为权重分配区域指定：

ANY_REGION = 0; // 任意区域

UP_REGION = 1; // 上半区

DOWN_REGION = 2; // 下半区

<img width="219" alt="image" src="https://github.com/user-attachments/assets/d5789f1f-331f-4d2d-818d-e7e4010b3f13">


### 3.7.3  **权重数据复制**

message DoubleWeightOpti{

​		int32 multiple = 1;

​	}

当需要对权重复制时使用：

设置multiple >1，会复制本层的权重，复制的倍数为2^ multiple。

例如，权重为2*3的矩阵[[1,2,3][4,5,6]]，设置

doubleWeightOpti {

   multiple : 1

}

权重变为4*3的矩阵[[1,2,3][4,5,6],[1,2,3][4,5,6]]。

### 3.7.4  **稀疏矩阵优化**

message SparseMatrixOpti{

​		SpMatMode sp_mode = 1;

}

enum SpMatMode{

​	SIM_MODE = 0;

​	MIX_MODE = 1;

​	DIG_MODE = 2;

}

稀疏矩阵适用于权重大小超过array存放限制，即[-275,275]之间时配置，可以将权重分为高8位和低8位，其中低8位使用array（NPU）进行计算，高8位使用CPU计算，最后相加可得最终结果。

由于大部分高位部分是0，所以是一个稀疏矩阵，但是当高位部分非0值较多时，可能会导致空间不足无法布置，mapper会给出错误。

此优化选项分为三种模式：

SIM_MODE为默认模式，纯NPU计算，权重范围为[-275,275]。

MIX_MODE为混合模式，低位使用NPU，高位使用CP，高低位皆为8bit。

DIG_MODE为数字模式，纯CPU计算，此模式只适用于kernel shape很小时的计算，以解决NPU模拟计算精度不足的问题，权重范围为[-128,127]。

### 3.7.5  **输出放大优化**

​	message ArrayOutputOpti{

​		int32 magnify = 1;

​		int32 repeat = 2;

​	}

magnify直接将当前层scale值缩小，方法为右移magnify位数。 

repeat 复制倍数，2^repeat次： 插入同样的op_conv算子再插入一个add_op和scale_op，将op与op_conv输出结果通过add_op相加然后通过scale_op除2，在round中可以使用后处理add结果移位实现除2，如下图所示。

<img width="231" alt="image" src="https://github.com/user-attachments/assets/a66e3793-9837-4aa3-9cc6-1a98ef353226">


### 3.7.6  **卷积优化**

​	message ConvOpti{

​		int32 multi_point = 1;

​		bool tdnn_mode = 2;

​		bool conv2DToConv1D_enable = 3;

​	}

此处是设定卷积的计算模式和多点并行数(加速计算)。

multi_point为多点并行个数，设置multi_point，需要将tdnn_mode或者conv2DToConv1D_enable为ture的情况下，而tdnn_mode和conv2DToConv1D_enable不能同时为ture。

tdnn_mode:DCCRN单独计算模式

conv2DToConv1D_enable：将2D卷积在W方向进行拆分，根据2D卷积计算原理，需要在W方向进行划窗，将W方向划窗的动作单独拆分为一层。

在WTM2101中，卷积有两种计算：普通卷积模式和tdnn卷积模式。

普通卷积模式是指卷积按照通常计算方法，即连续完成多次滑窗得到最终的输出feature map，如果加速多点计算，需将conv2DToConv1D_enable设置为ture，此时tdnn_mode必须为false。

而tdnn卷积模式是将W方向上赋予时间序列信息，并按照tdnn模式计算，多用于DCCRN网络中，同时需将tdnn_mode设置为ture，此时conv2DToConv1D_enable必须为false。

 

## 3.8  **特殊配置环境变量**

### 3.8.1  **Array分配**

因为硬件制作原因，有些芯片的前N列的Array计算会有较大误差，在烧写芯片测试后若发现可能为此方面的原因，可以设置环境变量：

export ARRAY_ALLOC_RESERVED_COLUMN=N

跳过前N列的Array分配。

在进行Array分配时，默认的搜索超时时间比较短，通常在10s~100s之间，具体跟权重块的数量相关，在网络权重块比较多的场景下，可能搜索不到最优解或无解，可以通过环境变量设置超时时长，设置如下：

export ARRAY_ALLOC_TIMEOUT_LIMITS = 300

### 3.8.2  **开启FIFO**

硬件支持fifo的使用，可以在TDNN的情况下提高数据输入的效率，不开启时需要一次输入TDNN的所需的所有帧数据，开启后可以只输入一帧数据，通过设置环境变量：

export WITIN_FIFO_EN = 1

开启。

**注意：开启此选项可能导致r****egfile****空间不足，若出现此问题，可尝试关闭。**

### 3.8.3  **设置校准数据数量**

校准文件expect_in/out用于板卡烧写测试模型的准确率，文件默认有100帧的校准数据，若需要增大、减小数据的个数，可以设置环境变量：

export WITIN_EXPECTED_NUM=N

会保存N帧的校准数据。



# 4  **Onnx模型**

## 4.1  **创建Tensor**

1．创建tensor数据，为int8数据(-128~127)，shape为[514,128]，最终的数据类型为float32。

```
params = np.random.randint(-128, 127, size = (514, 128), dtype = np.int32).astype(np.float32)
```

3．创建onnx tensor，参数分别为：tensor名称，数据类型，大小，值。

```
params_tensor = onnx.helper.make_tensor("params_tensor",

​                  data_type=onnx.TensorProto.FLOAT,

​                  dims=(514, 128),

​                  vals=params.flatten())
```



## 4.2  **onnx算子**

Onnx模型支持自定义算子。

常用算子有Gemm（矩阵乘、全连接层、DNN层），Tdnn（时延神经网络），Add（向量加），Relu（激活函数-relu），Scale（标量乘），LSTM（长短期记忆网络），GRU（门控循环单元），

ActLut（自定义激活函数查找表），Mul(向量乘)。

| ***\*常见算子列\*******\*表\**** |                    |                         |                |                    |                      |
| -------------------------------- | ------------------ | ----------------------- | -------------- | ------------------ | -------------------- |
| ***\*序号\****                   | ***\*算子名称\**** | ***\*作用\****          | ***\*序号\**** | ***\*算子名称\**** | ***\*作用\****       |
| 1                                | Gemm               | 矩阵乘、全连接层、DNN层 | 8              | GRU                | 门控循环单元         |
| 2                                | Tdnn               | 时延神经网络            | 9              | ActLut             | 自定义激活函数查找表 |
| 3                                | Add                | 向量加                  | 10             | Mul                | 向量乘               |
| 4                                | Relu               | 激活函数-relu           | 11             | LSTM               | 长短期记忆网络       |
| 5                                | Scale              | 标量乘                  | 12             | AveragePool        | 平均池化             |
| 6                                | Concat             | 对指定维度进行拼接      | 13             | Conv               | 卷积                 |
| 7                                | Slice              | 切片操作                | 14             | ConvTranspose      | 转置卷积             |

注：Tdnn、GRU、LSTM和ActLut算子为自定义算子，因此它们必须用onnx模型，同时也建议其他算子也使用onnx构建网络模型。

 

### 4.2.1  **Gemm**

一般矩阵乘法（General Matrix multiplication）。

```
node = onnx.helper.make_node('Gemm',

​                inputs=['input', 'params', 'bias'],

​                outputs=['output'],

​                name='dnn_node0')
```

​	其中，

​	inputs 		list[tensor]			为输入、权重、偏置

​	outputs 		list[tensor]			输出

​	name 			string					节点名称

**注意：Gemm后通常跟一个“scale”节点，代表G值。**

### 4.2.2  **T****dnn**

```
时间延迟网络（Time Delay Neural Network）。

node = onnx.helper.make_node('Tdnn',

​                inputs=['input', 'params'],

​                outputs=['output'],

​                time_offsets=offsets,

​                bias_params=bias,

​                scale_params=1024,

​                name='tdnn_node0')
```

其中，

​	inputs 				list[tensor]			输入、权重

​	outputs 				list[tensor]			输出

​	time_offsets 	tensor					时延信息

​	bias						tensor					偏置

​	scale_params		float						缩放值（G值）

​	name 					string					节点名称

### 4.2.3  **Lstm**

长短期记忆网络。

```
lstm6_node = onnx.helper.make_node(

​    'Lstm',

​    inputs=['conv5_reshape_out', 'lstm6_w_ifo_tensor', 'lstm6_w_c_tensor', 'lstm6_b_ifo_tensor', 'lstm6_b_c_tensor'],

​    scale_ioft=1024*2,

​    scale_ct=1024,

​    activate_type=['sigmoid', 'tanh', 'tanh'],

​    activate_table=act_table,

​    shift_bits=[0,0,-7, -7],

​    clean_ctht=10,

​    outputs=['lstm6_out'],

​    name="lstm6")
```

 

其中，

​	inputs 				list[tensor]		 输入、ioft权重、ctl权重、ioft偏置、ct偏置

​	scale_ioft 			float 					ioft部分G值

​	scale_ct 				float 					ct部分G值

​	activate_type		list[string]		激活表的类型，只能为sigmoid/tanh

​	activate_table 	tensor				激活表，每个激活表长度为256，LSTM有三个为[3,256]

shift_bits 				list[int]				 长度为4，

[0]为对ft*ct(16bit)结果的移位数值，

[1]为对it*ctl(16bit)结果的移位数值，

[2]为对ft*ct(16bit)+it*ctl(16bit)的结果ct(16bit)移位数值，

[3]为对ot(8bit)*tanh2_ct(8bit)的结果ht(16bit)移位数值，

不设置默认为[0,0,-7,-7]。

clean_ctht int 表示htct在执行该设置次数后置为0，当clean_ctht设置为0或者不设置，为不清除ht/ct.

​	outputs 				list[tensor]		 输出Tensor

​	name 					string				 节点名称

计算公式如下：

![image](https://github.com/user-attachments/assets/13daf8d9-39fe-47f5-8dd8-5ccb3687d432)

### 4.2.4  **G****ru**

门控循环单元。

```
node = onnx.helper.make_node(

​    'Gru',

​    inputs=['in', 'gru_zrt_params', 'gru_ht_params', 'gru_zrt_bias', 'gru_ht_bias'],

​    scale_zr=1024,

​    scale_ht=1024,

​    activate_type=['sigmoid', 'tanh'],

​    activate_table=act_table,

​    shift_bits=[-7, -7],

​    outputs=['out'],

​    name="gru_node1")
```

其中，

​	inputs 				list[tensor]		输入、zrt权重、ht权重、zrt偏置、ht偏置

​	scale_zr 				float 					zrt部分G值

​	scale_ht 				float					 ht 部分G值

​	activate_type		list[string]		激活表的类型，只能为sigmoid/tanh

​	activate_table 	tensor				激活表，每个激活表长度为256，Gru有两个为[2,256]

shift_bits				list[int]				长度为2，

[0]为对rt(8bit)*ht(8bit)的结果ct(16bit)移位，

[1]为对[(1-zt)*ht](16bit)+[zt*ht~](16bit)的结果ht(16bit)移位，

不设置默认为[-7,-7]。

​	outputs 				list[tensor]		输出Tensor

​	name 					string				节点名称

计算公式如下：

![image](https://github.com/user-attachments/assets/6e052771-95aa-49c5-8d69-4634b55334ae)


### 4.2.5  **Scale**

标量乘。

```
node = onnx.helper.make_node('Scale',

​                inputs=['in'],

​                outputs=['out'],

​                scale=0.0009765625)
```

其中，

​	inputs 		list[tensor]			输入

​	outputs 		list[tensor]			输出

​	scale 		float						缩放值

**注意：scale的值取值范围为[****-128,127] U (-1,1)****。当scale的值为(****-1,1)****时，s****cale****的值只支持2****^-n****，****n****为整数，n****<=13****：n****<8****时，为后处理乘法，使用移位进行计算。n>****=8****，n****<=13****时，若scale层跟在Gemm层、Conv****2d****层后，代表为上层Gemm****/Conv2d****的G值，即最小G值为2****56****，最大为8****192****；若不跟在Gemm层、Conv****2d****层后，则为单独的乘法计算。当scale的值取值范围为[****-128,127]****时，scale只能为其中的整数。**

### 4.2.6  **Relu**

激活函数。

```
node = onnx.helper.make_node('Relu', 

​               inputs = ['input'], 

​               outputs = ['output'])
```

其中，

​	inputs 		list[tensor]			输入

​	outputs 		list[tensor]			输出

### 4.2.7  **ActLut**

自定义激活查找表函数，通过int8的数据范围-128~127的映射表，映射为激活输出。

```
node = onnx.helper.make_node('ActLut',

​              	 inputs=['in'],

​               outputs=['out'],

​               act_type='sigmoid',

​               table_params=act_table)
```

其中，

​	inputs 				list[tensor]			输入

​	outputs 				list[tensor]			输出

​	act_type				string					激活表类型，可选为sigmoid、tanh

​	table_params		tensor					激活表tensor，为[1,256]大小

**注意：查找表的对应方式为：**		**源输入[0~127] 对应查找表下标0~127，源输入[-128~-1]对应查找表下标128~255，即：输入[-128~127]的二进制转变为无符号形式为[128~255,0~127]，对应为查找表下标。**

### 4.2.8  **Mul**

乘

```
node = onnx.helper.make_node('Mul',

  														inputs=['in', 'mul_x'],

  														outputs=['out'],

)
```

其中，

​	inputs 				list[tensor]			输入，左乘数，右乘数

​	outputs 				list[tensor]			输出

注意：如果左乘数和右乘数都为向量时，它们的shape要求相等；也支持

左乘数和右乘数其中一个是向量，另外一个是标量的情况。

### 4.2.9  **Add**

加

```
node = onnx.helper.make_node('Add', 

inputs = ['in', 'add_x'], 

shift_bit = -1,

outputs = ['out'])
```

其中，

​	inputs 				list[tensor]			 输入，左加数，右加数

shift_bit int 表示对add后的结果移位，负数表示右移，正数表示左移，默认0

​	outputs 				list[tensor]			 输出

注意：		如果左加数和右加数都为向量时，它们的shape要求相等；也支持

左加数和右加数其中一个是向量，另外一个是标量的情况。

 

### 4.2.10  **Averagepool**

平均池化。

```
left_ave_pooling_node = onnx.helper.make_node(

​    "AveragePool",

​    inputs=["conv16_left_add_relu"],

​    outputs=["out"],

​    kernel_shape=[35, 2],

​    pads=[0, 0, 0, 0],  #[top,left,bottom,right]

​    strides=[1, 1],

​    scale_in=0.0625,

​    scale_out=0.0625)
```

其中，

inputs list[tensor] 输入、权重、偏置

outputs list[tensor] 输出

kernel_shape list 池化核shape

pads list 填充

strides list 步长

scale_in float 输入缩放尺度，默认为1

scale_out float 输出缩放尺度，默认为1

### 4.2.11  **Slice**

切片操作。

```
conv3_1_left_slice_node = onnx.helper.make_node(

​      "Slice",

​      inputs=["gemm3_left_reshape_out", "conv3_1_left_slice_starts", "conv3_1_left_slice_ends", "conv3_1_left_slice_axes", "conv3_1_left_slice_steps"],

​      outputs=["conv3_1_left_slice_out"])
```

其中，

inputs list[tensor] 输入、切片起始索引、切片结束索引、切片指定轴、切片步长

outputs list[tensor] 输出

### 4.2.12  **Concat**

数据拼接。

```
concat_node = onnx.helper.make_node(

"Concat",

​    inputs=["concat_in1","concat_in2"],

​    outputs=["deconv7_concat"],

​    axis = 1)
```

其中，

inputs list[tensor] 输入：左拼接数，右拼接数

outputs list[tensor] 输出

axis int 指定拼接轴，目前WTM2101仅支持axis=1.

### 4.2.13  **Conv**

卷积。

```
conv3_node = onnx.helper.make_node("Conv",

​       inputs=['conv2_relu', 'conv3_w_tensor', 'conv3_b_tensor'],

​       outputs=['conv3'],

​       kernel_shape=[3, 2],

​       strides=[2, 1],

​       pads=[1, 1, 0, 0],

​       name="conv3")
```

其中，

inputs list[tensor] 输入、权重、偏置

outputs list[tensor 输出

kernel_shape list[int] 卷积核shape

strides list[int] 滑动步长

pads list[int] 填充

name string 节点名称

注意： 

1）在构建onnx模型权重维度是[O,I,H,W],经工具链内部转换，维度转换为[O,W,H,I],最终shape为[O,W*H*I]；

2）在工具链内部，卷积feature map维度为[N,W,H,C].

### 4.2.14  **ConvTranspose**

转置卷积。

```
deconv12_node = onnx.helper.make_node("ConvTranspose",

​     inputs=['deconv_input', 'deconv_weight', 'deconv_bias'],

​     outputs=['deconv_output'],

​     kernel_shape=[3, 2],

​     strides=[2, 1],

​     pads=[1, 0, 1, 0],

​     output_padding=[1, 0, 1, 0],

​     name="deconv_node")
```

其中，

inputs list[tensor] 输入、权重、偏置

outputs list[tensor 输出

kernel_shape list[int] 卷积核shape

strides list[int] 滑动步长

pads list[int] 填充

output_padding list[int] 输出填充

name string 节点名称

注意：

注意：

1）在构建onnx模型权重维度是[I,O,H,W],经工具链内部将权重转换为等价的Conv，转换后维度转换为[O',W',H',I'],最终shape为[O’,W’*H’*I’]；

2）在工具链内部，转置卷积的feature map维度为[N,W,H,C]；

3）在WTM2101工具链中，转置卷积主要应用在DCCRN场景中.

### 4.2.15  **Gru2Array**

新增双权重GRU部署方式，其接口如下：

```
gru_node1 = onnx.helper.make_node('Gru2Array',

​    inputs=['in', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],

​    input_offset=0,

​    hidden_shift_bit=-1,  #输入scle

​    scale_input=1024,

​    scale_hidden=1024,

​    scale_ones = 128,

​    activate_type=['sigmoid', 'tanh'],

​    activate_table=act_table,

​    shift_bits=[-7, -7, -2, -3], 

​    clean_ht=0,

​    outputs=['out'],

​    name="gru_node1")
```

 

inputs list[tensor]输入、input_zrn权重、hidden_zrn权重、input_zrn偏置、hidden_zrn偏置

input_offset：对输入的偏移量，暂未适配，现仅保留接口

hidden_shift_bit：对array计算后的hidden_zrn进行移位，“-”代表右移位

scale_input float 表示input_zrn部分G值：

scale_hidden float 表示hidden_zrn部分G值

activate_typelist[string]激活表的类型，只能为sigmoid/tanh

activate_table tensor激活表，每个激活表长度为256，或者1024，目前建议取长度为1024

shift_bitslist[int]长度为2，

[0]为对rt(8bit)*hn(8bit)的结果(16bit)移位，

[1]为对[(1-zt)*ht](16bit)+[zt*ht~](16bit)的结果ht(16bit)移位，

[2]为对(1-zt)*ht移位，如果不移位，必须设置为0

[3]为对zt*ht~进行移位，如果不移位，必须设置为0

不设置默认为[-7,-7,-7,-7],-7表示右移七位，-代表右移位。

outputs: list[tensor],输出Tensor

name string:节点名称

scale_ones：表示“1”和zt的缩放尺度，目前支持-128~128

clear_ht:gru执行如果次数后，将ht清零，等于0时，代表不清除ht

 

## 4.3  **创建graph**

节点列表

nodes = [node0, node1]

图的名称

name = 'model_name'

输入信息

inputs=[onnx.helper.make_tensor_value_info("in",onnx.TensorProto.FLOAT,list(in_shape))]

输出信息

outputs=[onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT,list(out_shape))]

初始化Tensor

```
initializer=[tensor1, tensor2, tensor3]

graph = onnx.helper.make_graph(nodes, 

name,

​                 inputs,

​                 outputs,

 										      initializer

 )
```

 

 

## 4.4  **保存graph**

```
model=helper.make_model(graph, producer_name=model_name)

with open(ROOT_DIR + model_name + '.onnx', "wb") as of:

​    of.write(model.SerializeToString())
```



## 4.5  **导入模型**

```
onnx_model = onnx.load(file_path)
```





















# 1  **常见问题和说明**

## 1.1  **Array分配error**

<img width="429" alt="image" src="https://github.com/user-attachments/assets/a0a2bc1e-cfb4-4718-9689-37c2ff54026f">


提示：

[ERROR]low_level_mapping/src/tensor/array_alloc_cp.cpp:169: Failed to allocation array memory

说明array分配超出限制，需重新调整网络结构或者优化方式。

## 1.2  **Regfile空间不足**

<img width="429" alt="image" src="https://github.com/user-attachments/assets/e4898c98-d441-4516-bbc3-ee3df7810c4b">

提示：

[ERROR]low_level_mapping/src/memory/mem.cpp:284: No enough memory to alloc, Need: 771

说明regfile资源不够使用，需重新调整网络结构。

## 1.3  **SRAM空间不足**

<img width="449" alt="image" src="https://github.com/user-attachments/assets/ab01b084-b00a-43ac-9380-633241de82a6">

提示：

Bitfield ERROR: json:[paras7, pen ding_rd_addr, ] name:pending_rd_addr lsb:16 len:16 value:67200

说明：SRAM资源不足，需重新调整网络结构。

## 1.4  **输入类型错误**

<img width="457" alt="image" src="https://github.com/user-attachments/assets/d6cc772f-a695-4030-b7be-038ece297732">

提示：

ValueError: input data only support float32, but now is float64, please use input_data.astype('float32')

说明：输入数据类型有无，应使用input_data.astype('float32')。
