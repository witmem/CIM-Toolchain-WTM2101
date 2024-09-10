# Witin Toolchain

This folder contains Witin compiler and runtime.

## V0.3.0: 2020.11.23

功能和使用说明：
1、上接口转换
	1）支持自定义组图和直接输入pb文件，两种方式均可以跟witin npu的结果进行比对。
	2）自定义组图时候需将scale， round, clip等算子加入到图中。具体见witin_730_custom.py
	3）使用sigmoid算子时候，为保持tensorflow和npu的计算结果一致，tensorflow图中需使用
		tf.round(127/(1+tf.exp(-x/8)))计算代替sigmoid，npu图中正常使用sigmoid函数即可。
	4）目前支持relu, sigmoid激活函数，不支持tanh激活函数

2、集成了NB01P模拟器。
	1）模拟器存放于witin_sim/路径下
	2）提供了仿真运行的module，可进行打注输入，运行，并返回结果，
		分别调用module的set_input, run, get_output等接口
	3）可进行与tensorflow结果进行对比
		具体参照witin_730.py脚本
	4）模拟器运行的结果存放于output/simulator_output中

3、多网络的支持
	1）支持多网络的功能，可同时映射多个网络到NB01P芯片中
	2）witin_mapper网络映射完成后，会生成array和regfile的占用情况文件，存放于
		output/memory/memUseageFile.data文件中
	3）witin_mapper将首先从output/memory/memInitFile.data中读取数据，用户初始化memory
	4）使用的仅需将memInitFile.data中内容替换为memUseageFile.data中的内容

4、profile
	1）该工具链会生成权重和bias等的分析文件，存放于output/profile/文件中
		该文本文件可用于后期witin进行相应的debug

5、烧写工具配置文件
	1）存放于output/map/文件夹中

6、工程文件
	1）在output/目录下产生array_round_config.h文件，
		该文件用于集成到keil工程中用于npu的配置，以完成计算。

## v0.3.1

1. modify phy.csv to map.csv

## v0.3.2

1.generate array_layer_config.h

## v0.5.0

1.support tdnn
	see example tests/onnx/witin/test_forward.py
2.runtime debug function
	after run in cmodel, see output/simulator_output/input_x_output_round_y
	x is Nth input,
	y is layer num
3.multi net support
	1）支持多网络的功能，可同时映射多个网络到WTM1001芯片中
	2）witin_mapper网络映射完成后，会生成array和regfile的占用情况文件，存放于
		output/memory/array_archive_save.txt和output/memory/memUseageFile.data中
	3）witin_mapper将首先从array_archive_init.txt和memInitFile.data中读取数据，用户初始化memory
	4）使用的仅需将memInitFile.data中内容替换为memUseageFile.data中的内容,并用array_archive_init.txt替换为array_archive_save.txt中的内容
4、完善dnn的例程，tdnn+dnn
	参见tests/onnx/witin/test_forward_tdnn_dnn.py

## v0.5.3

1.tdnn+dnn例程（build+run）
	tests/onnx/witin/test_forward_tdnn_dnn.py
2.tdnn例程(build)
	tests/onnx/witin/test_forward_04.py
3.增加安装脚本
	build/install.sh

## v0.6.0

1.support multi chip
  chips: NB01P NB01PP BB04P BB04P1
  (1).build
  witin_frontend.build_module.build(mod, target=target,
                        target_host=target_host, params=params,
                        input_data=input_dt,
                        chip = "NB01P"
                        )
  (2).create graph
  m = npu_graph_runtime.create(npu_graph, "NB01P")

2.examples
	tests/onnx/witin/wtm1001/test_forward_tdnn_dnn.py
	tests/onnx/witin/wtm2101/test_forward_tdnn_dnn.py

## v0.7.0

1.support multi net
  examples: python3 tests/onnx/witin/wtm100/test_forward_tdnn_vad_fuse.py
  TODO: multi net runtime

## v0.8.0

1.support conv2d, elementwise add/mul, const add/mul, reshape operators
2.support tdnn/dnn/cnn input is int8 Tensor
3.support add+relu+tdnn structure
4.generate array additional config : in output/array_additional_config dir
5.update array alloc algorithm , use google ortool instead

## v0.8.1

1. support branch Scale operation , left and right shift 1....8

## v0.8.2

1. support runtime  run multi frame input
2. add bias split check

## v0.8.3

1. 增加层的优化选项，示例程序
	tests/onnx/witin/wtm2101/test_layer_optimize_method.py
   	build时指定路径 optimize_method_config

2. 优化方法文件书写格式示例 model/huawei_model/optimize_config.protobuf
	layerOptiConfig {
		name : "TdnnOpNode2"    //节点名称，生成ONNX node时指定name=""
		inputOpti {     //输入扩大
			shift : 0   //是否开启
			num : 1     //移位数，>0向左移位，<0向右移位，现在只支持1
		}
		weightOpti {    //权重扩大
			shift : 1   //是否开启
			num : 1     //移位数，>0向左移位，<0向右移位
		}
		doubleWeightOpti {  //权重复制
			multiple : 0    //只能为2^n个，即1时为复制1份，2时复制为4份
		}
		sparseMatrixOpti {  //稀疏矩阵，当前不可用
			enable : 0
		}
	}
3. 层优化模拟器现仅支持BB04P1
4. protoc 版本为 libprotoc 3.7.1，可通过 protoc --version查看

## v0.8.4

1. 优化方法,修复inputOpti的配置，注意：
	1).第一层输入设置扩大，需要将net_input的值手动扩大，G值配置后会自动扩大
	2).若输入扩大的上节点为分支，则分支的另一输入也需要同等扩大
	3).输入扩大的的上节点必须有relu
2. 修复结果对比余弦相似度的问题。
3. 增加生成 INPUT_UPDATE_NUM_NET0 和 INPUT_FIFO_EN_NET0 和 DIFF_SRAM_ADDR 宏

## v0.8.5

1. 更新bias，最大为16*128*255
2. 修复了校准数据的问题
2. log信息更为准确

## v0.8.6

1. 修复解析配置文件出错
2. 修复加入优化且输入为signed时bias计算错误的问题
3. 加入huawei_sh_CBG识别网络的优化策略测试
   witin_mapper/tests/onnx/witin/wtm2101/test_forward_hua_shibie_all.py

## v0.8.7

1. 增加fifo使能，通过环境变量设置 export WITIN_FIFO_EN=1 开启，
   可能会导致空间不足，此时关闭export WITIN_FIFO_EN=0即可，默认不开启。
2. 修复校准与实际运行std斜率不匹配的问题。

## v0.8.8

1. 修改Array计算除G值，使用floor向下取整
2. 修改移位为直接移位

## v0.8.8R1

1、InputOpti中增加front_scale项，当一层inputOpti优化开启，num=1且其front_scale为1时，将通过改变上一层g值的方式（右移front_sacle位）将本层的输入扩大一倍。
2、网络第一层bias可扩大4倍
3、增加相应的测试case，见test_layer_optimize_method.py

## v0.8.8R2

1. 增加降噪网络的支持，测试程序见
	witin_mapper/tests/onnx/witin/wtm2101/test_forward_huawei_hz_e2e_decoder_1_relu.py
2. 模拟器与功能仿真完全对齐。

## v0.8.8R3

1. 修改功能仿真和模拟器Array运算、后处理移位与硬件对齐
2. 增加功能仿真与TensorFlow结果对比测试例
   witin_mapper/tests/python/frontend/tensorflow/wtm2101_custom/test_func_sim_output_equal_tf.py
3. 增加异常返回：DNN节点使用doubleweightopti，但前一节点为TDNN且输出为多帧时报错。

## v0.8.8R4

1. 对于网络输入为有符号数时，对第一层添加signed_input:1优化，注意需要手动将原始输入[x]拼接为[x,-x]，测试程序见
	witin_mapper/tests/onnx/witin/wtm2101/test_layer_optimize_method.py
	'optimize_method_config_frist_layer_signed_in'测试例。

## v0.8.8R5

1. 修复第一层为TDNN时，开FIFO模式同时开启doubleWeight优化时计算不正确的问题。
2. 添加分配Array时跳过前N列的开关，通过设置环境变量：export ARRAY_ALLOC_RESERVED_COLUMN=N 开启，
   例如export ARRAY_ALLOC_RESERVED_COLUMN=128为跳过前128列

## v0.8.8R6

1. 修复TDNN前为多帧eltwise/active输出时计算错误的问题，测试例子为：
	witin_mapper/tests/python/frontend/tensorflow/wtm21011_custom/test_func_sim_output_equal_tf_case2.py
2. 在输入可释放的情况下复用active op的regfile输入空间；优化分配空间时的错误提示。

## v0.8.8R7

1. regfile空间不足时输出详细的每层占用的空间，方便裁剪网络。
2. 优化策略的层名“name”配置错误时，输出错误信息。

## v0.8.8R8

1. 	添加自定义激活算子。
  	生成onnx模型节点格式为
   	onnx.helper.make_node('ActLut', ['in'], ['out'], act_type = 'sigmoid' , table_params = table1)
	其中，
		ActLut   	 为onnx自定义激活节点类型；
		['in']   	 输入名称；
		['out']  	 输出名称；
		act_type 	 自定义激活算子占据的激活查找表名称，可选为sigmoid、tanh,其他类型将报错；
		table_params 自定义激活查找表，注意，查找表的对应方式为：
		 			 源输入[0~127] 对应查找表下标0~127，源输入[-128~-1]对应查找表下标128~255,
					 即：输入[-128~127]的二进制转变为无符号形式为[128~255,0~127]，对应为查找表下标。
	生成模型及测试例为：
    tests/onnx/witin/wtm2101/test_forward_acttable.py
2. Array分配失败时输出log:各层的名称及占用array的大小。
3. Build/Run计算效率优化。

## v0.8.8R9

1. 	多网络的支持，测试例见：
    	witin_mapper/tests/onnx/witin/wtm2101/test_forward_multinet.py
   	多网络set_input()的时候，要指定网络输入节点的名称或网络的序号,例如：
		# 1.已知输入节点的名称
		m.set_input('cnn_1', witin.nd.array(data1))
		m.set_input('encoder_dnn_1', witin.nd.array(data2))
		# 2.不知道节点名称，按照Graph的顺序
		# m.set_input(0, witin.nd.array(data1))
		# m.set_input(1, witin.nd.array(data2))
   	单网络可任意设置。

2. 修改输出接口，m.get_output(i)为第i个输出节点的多帧输出，不再为1个输出的第i帧。

## v0.8.8R10

1.	生成校准数据时，随机选取其中一些帧生成校准文件(expected_in/out.bin)，同时保留所有帧生成的校准数据文件(expected_out_complete.bin)。
2.	array分配失败时，输出每一层占据array的位置信息。
3. 	修复多网络下，regfile空间申请异常的问题。


## v0.8.9R1

1. 	添加输入为int8时，部分输入采用扩充为[x,-x]方法、部分输入采用+128的方法的优化接口，测试例为：
		witin_mapper/tests/onnx/witin/wtm2101/test_layer_optimize_method_signed_in.py
	如果网络输入为int8时，启用此优化，并根据array大小选择设置适合的signed_input = N(N > 0, N < input_size)。
    需要对int8输入数据进行处理输入到网络中，原始输入X(int8)需要改变为3部分:
        X1 = X[0:N]
        X2 = X[N:end] + 128
        X3 = -X[0:N]
    将X1,X2,X3拼接一起
        X' = np.concatenate((data0, data1, data2), axis = 1)
    将负数变为0
        X'[X'<0] = 0
    得到新的输入X'(uint8)

2. 	修复在 weightopt 和 signed_input 同时开启时，计算顺序调整，对于权值-128：
		以前： [-128] ==> [-128,127] ==> * 2 ==> [-256,254]
		修改为:[-128] ==> * 2 ==> [-256] ==> [-256,255]

3. 	添加检查： signed_input 开启时，doubleweight 和 WITIN_FIFO_EN 不能开启(在0.8.9R2弃用)
4. 	修复：多网络build时可以设置不同网络不同帧的输入，注意：此时的expected_out_complete.bin不可用于校准，因为他保留所有数据但不同网络有不同帧。
5.  自动得到校准输入的起始帧，输出在random_idx.txt
6.	设置校准文件的长度 export WITIN_EXPECTED_NUM=N ,默认为100帧校准数据。

## v0.8.9R2

1.	修复优化选项冲突：
	修复 signed_input 开启时，doubleweight 和 WITIN_FIFO_EN 可以同时开启。
	新增 首层为 TDNN 时可以开启 signed_input，但只支持全部输入为正负输入的设置，
		注意此时 WITIN_FIFO_EN 开启时，设置的 signed_input 数值应为1帧大小，未开启时 signed_input 为全部帧大小
2.	新增生成每层的输入输出对比文件，用于调试。
3.	新增支持model权重参数可支持数值范围为int16。

## v0.9.0

1.  增加对网络多输入多输出的支持(BB04P1)，测试例为：
		witin_mapper/tests/onnx/witin/wtm2101/test_forward_multi_input_output.py
	多输入build时，输入字典的key应当为真实节点名称。
2. 	修复某些情况下，分支间regfile空间分配不同的问题
3. 	修复输入优化节点前为分支节点时，检查逻辑错误的问题
4. 	新增对LSTM算子的支持，测试例为：
		witin_mapper/tests/onnx/witin/wtm2101/test_lstm.py
	生成onnx节点的接口：
		# 生成所需的激活表，LSTM为3个，每个激活表的长度为256
	    act_table = np.random.randint(-50, 50, size=(3, 256), dtype=np.int32).astype(np.float32)
    	act_table = onnx.helper.make_tensor("act_table",
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=(3, 256),
                                        vals=act_table.flatten())
		onnx节点，参数从头到尾分别为：
			type			Lstm
			inputs			输入数据，权重ioft参数，权重ct参数，ioft偏置，ct偏置
			scale_ioft		权重ioft的G值
			scale_ct		权重ct的G值
			activate_type	激活表的类型
			activate_table	激活表
		 	shift_bits		对ft*ct(16bit)+it*ctl(16bit)的结果(16bit)移位数值，对ot(8bit)*tanh_ct(8bit)的结果(16bit)移位数值
			outputs			输出
			name			节点名称
		lstm_node1 = onnx.helper.make_node(
			'Lstm',
			inputs=['in', 'lstm_params1', 'lstm_params2', 'lstm_bias1', 'lstm_bias2'],
			scale_ioft=1024,
			scale_ct=512,
			activate_type=['sigmoid', 'tanh', 'tanh'],
			activate_table=act_table,
			shift_bits=[-7, -7],
			outputs=['lstm_out'],
			name="lstm_node1")
	注意：当前版本LSTM需要的激活表数量>2，所以工具生成的配置不可直接使用，需要部署时加入CPU切图的操作。

5.	新增对GRU算子的支持，测试例为：
		单算子	witin_mapper/tests/onnx/witin/wtm2101/test_gru.py
		图		witin_mapper/tests/onnx/witin/wtm2101/test_gru_nodes.py
	生成onnx节点的接口：
		# 生成所需的激活表，LSTM为3个，每个激活表的长度为256
	    act_table = np.random.randint(-50, 50, size=(2, 256), dtype=np.int32).astype(np.float32)
		act_table = onnx.helper.make_tensor("act_table",
											data_type=onnx.TensorProto.FLOAT,
											dims=(2, 256),
											vals=act_table.flatten())
		onnx节点，参数从头到尾分别为：
			type			Gru
			inputs			输入数据，权重zr参数，权重ht参数，zr偏置，ht偏置
			scale_ioft		权重zr的G值
			scale_ct		权重ht的G值
			activate_type	激活表的类型
			activate_table	激活表
		 	shift_bits		对rt(8bit)*ht(8bit)的结果(16bit)移位数值，对[(1-zt)*ht](16bit)+[zt*ht~](16bit)的结果(16bit)移位数值
			outputs			输出
			name			节点名称
		gru_node1=onnx.helper.make_node('Gru', inputs=['in', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],
                                    scale_zr=1024, scale_ht = 1024, activate_type = ['sigmoid','tanh'],
                                    activate_table=act_table,
									shift_bits=[-7, -7],
                                    outputs=['out'], name="gru_node1")

6.	修复：pytorch模型conv2d转化时，padding参数符合硬件配置要求。
7.	新增：生成layer_debug文件，保存所有round的输入输出，用于调试。


## v0.9.0R1

1.	新增：模拟器run支持GRU和CNN模式。
		注意：模拟器CNN模式暂不支持开启double weights 优化，暂不支持padding的conv层，暂不支持中间插入后处理层。

2. 	修改：scalar OP，右乘数不在使用向量而使用标量，可以减小的Regfile使用空间。
		注意：scalar 标量值取值范围为[-128,127]，为int8整数；在(-1,1)之间时，数值只支持2^-n,n<8。

3.	修复：生成精度调试工具所需文件及配置的问题。
		注意：首层为TDNN网络时，build输入为单帧数据。

4.	修复：weights取值范围为[-275,275]，超过时截断。

5.	修复：CNN模式下配置出错的问题。

## v000.090.016

1	添加优化选项自动搜索功能。
		根据搜索到的各个算子的参数，分析各算子可以采用的优化选项，
			遍历所有算子的优化选项，用尽可能少的运行次数，把所有算子的所有的可行优化选项跑完，得到所有优化选项对应的output和优化策略。
		目前仅支持DNN算子，TDNN算子在运行模拟器的时候还存在bug，目前分析应该是模拟器的bug，待进一步定位分析。

## v0.9.0R1.7

1，添加优化选项自动搜索功能，根据搜索到的各个算子的参数，分析各算子可以采用的优化选项，遍历所有算子的优化选项，用尽可能少的运行次数，把所有算子的所有的可行优化选项跑完，得到所有优化选项对应的output和优化策略
2, 目前仅支持DNN算子，TDNN算子在运行模拟器的时候还存在bug，目前分析应该是模拟器的bug，待进一步定位分析

## v000.090.018

1, 修复array_additional.h中，在多次调用build的场景下，部分配置会被重复添加的bug

## v000.090.019

1, 修复regfile_init.h中，初始化数据总是占用8K的bug

## v000.090.020

1, 添加自动分析搜索优化选项功能对conv的支持，并添加对应测试例

## v000.091.001

发布日期：2022/08
1.	新增：mapper指定输出文件路径接口output_dir：
		witin_frontend.build_module.build(..., ..., output_dir="./output")
		未指定保存路径或 output_dir="" 默认保存在 "./output" 文件夹下。

2. 	新增：支持多线程运行：
		参考例程在witin_mapper/tests/onnx/witin/wtm2101/test_multi_thread.py。
		通过指定输出路径即可相互不依赖的同时build模型。

3.	新增：最小build的接口:
		witin_frontend.build_module.build(..., ..., array_distribute="output_dir/map/layers.txt")
		array_distribute 为需要输入完整build时的array分布（output_dir/map/layers.txt)文件。
	此模式下只保留：
		map/map.csv,layers.txt,expected_in.bin,expected_out.bin;
		array_additional_config/array_additional_data.h;
		params/*
		BoardConfig.json
	其他无关的文件不再保存及计算。
	适用于修改了部分参数但网络结构未改变的情况且需要最小化build模型时间时使用。

4.	修改：稀疏矩阵优化选项：
		sparseMatrixOpti {
			sp_mode : SIM_MODE/MIX_MODE/DIG_MODE
		}
	SIM_MODE: 普通array模拟计算模式，默认。
	MIX_MODE：将权重分为低位和高位部分，使用array计算低位，使用数字计算高位。
	DIG_MODE：使用数字计算，只适用于kernel shape 很小的情况。
	
5.	新增：Onnx中Mul算子，支持8bit输入乘后移位为8bit时的数量。
		node_mul = onnx.helper.make_node('Multi', inputs=['in1', 'in2'], outputs=['out'], shift_bit=-8)
		或
		node_mul = onnx.helper.make_node('Mul', inputs=['in1', 'in2'], outputs=['out'], shift_bit=-8)
		onnx节点，参数从头到尾分别为：
			type			Multi/Mul
			inputs			输入数据
			outputs			输出数据
			shift_bit		输出移位，大于0为左移，小于0为右移
			
6.	新增：Onnx中Add算子，支持8bit输入加后移位为8bit时的数量。
		node_add = onnx.helper.make_node('Added', inputs=['in1', 'in2'], outputs=['out'], shift_bit=-8)
		或
		node_add = onnx.helper.make_node('Add', inputs=['n1', 'in2'], outputs=['out'], shift_bit=-8)
		onnx节点，参数从头到尾分别为：
			type			Add/Added
			inputs			输入数据
			outputs			输出数据
			shift_bit		输出移位，大于0为左移，小于0为右移

7.	修改：LSTM、GRU的shift_bits字段为默认为[-7,-7]，代表默认右移7位

8.	新增：RPT-tool工具，在build/rpttool下，为json生成C代码时使用，使用命令如下：
		./rpttool json文件 输出路径 [是否为补偿]

9.	新增：conv1d支持多点并行计算，默认开启

10. 新增：优化选项 inputOpti.signed_input = -1 时，使用全部输入数据+128的方式。

11. 新增：优化选项 convOpti.multi_point, 为0时自动计算可进行的最大多点并行，大于0时指定并行的点数。
			当并行点数不能被输出整除时，会自动缩减为输出的最大整数倍。
			当指定的点数超过最大可并行点数时，会自动缩减为最大可并行点数。

12. 新增：环境变量 export WITIN_ADDITIONAL_DATA_USE_AVG=1 ，补偿数据使用所有帧的平均数。

## v001.000.001

1. 支持dccrn网络

## v001.000.005

1. 修复更新round memory config 的逻辑

## v001.000.014
1. 补偿方法更新
2. 修复expect数据选取
3. 修复查表数据大小端

## v001.000.015
1. 支持输出文件加密
2. 支持事件debug输入输出文件

## v001.000.016
1. 修复scale层在cnn时复用输入输出空间

## v001.000.017
1. 修复lstm输入与ht空间不连续时复制输入时，地址错误的问题
2. 修复空间分配的问题

## v001.000.018
1. 新增：支持avgpool算子，使用mcu进行计算
	onnx.helper.make_node(
			"AveragePool",
			inputs=["in"],
			outputs=["out"],
			kernel_shape=[8, 2],
			pads=[0, 0, 0, 0],  #[top,left,bottom,right]
			strides=[1, 1],
			scale_in=0.0625,
			scale_out=0.0625)
	onnx节点，参数从头到尾分别为：
				type			AveragePool
				inputs			输入数据
				outputs			输出数据
				kernel_shape    池化窗口，[kh,kw]
				pads 			边补0，[top,left,bottom,right]
				strides 		池化步长m,[sh,sw]
				scale_in		输入量化参数
				scale_out		输出量化参数

2. 支持 mv/conv2d + scale 融合到mv/conv

## v001.000.019
1. 修复：scale融合后配置和debug数据不正确的问题
2. 修复：某些文件名含有特殊字符导致在windows下不正常的问题
3. 新增：检查输入数据类型必须为float32

## v001.000.020
1. 修复debug数据保存时，某些情况下多行一帧的问题。

## v001.000.021
1. 修复conv/dnn + scale 移位错误的问题

## v001.000.022
1. 修复补偿配置生成错误的问题
2. 添加输入节点，设置输入数据时使用真实节点名称

## v001.000.023
1. 默认补偿数据使用2帧不同数据
2. Lstm添加每过clean_ctht帧数据清除ht/ct数据接口，只在功能仿真生成debug数据时使用，不生成硬件配置。
	clean_ctht设置为0或者不设置，为不清除ht/ct.
	测试：witin_mapper/tests/onnx/witin/wtm2101/test_lstm.py::test_lstm_case3_clean_ctht
	lstm_node1 = onnx.helper.make_node(
			'Lstm',
			inputs=['in', 'lstm_params1', 'lstm_params2', 'lstm_bias1', 'lstm_bias2'],
			scale_ioft=1024,
			scale_ct=512,
			activate_type=['sigmoid', 'tanh', 'tanh'],
			activate_table=act_table,
			shift_bits=[-7, -7],
			clean_ctht=10,
			outputs=['lstm_out'],
			name="lstm_node1")

3. 修复：conv2d首层添加double weights优化，但是必须手动将输入[N,W,H,C]变为[N,W,H,2*C],在channel方向进行复制。
4. 修复：在某些情况下，优化文件未设置某些优化时，段错误的问题。

## v001.000.024 
1. 修复：tdnn多输入时，配置生成错误的问题
2. 修复：多节点共用输入节点时，空间重复使用的问题。
3. 新增：对LSTM内部某个array单独进行复制，然后对输出相加除2。
   参考例程：witin_mapper/tests/onnx/witin/wtm2101/test_lstm.py::test_model_case2
   优化文件编写：
		layerOptiConfig {
			name : "lstm_node1"
			arrayOutputOpti{
				repeat : 0
			}
			arrayOutputOpti{
				repeat : 2
			}
		}
	其中，第一个arrayOutputOpti为LSTM第一个array复制（当前关闭），第二个arrayOutputOpti为LSTM第二个array复制（当前复制）。
	repeat只支持为2。设置为0或1时为不进行复制。

## v001.000.025
1. 修复：LSTM节点layer_debug.json中，第二个array+mul，mul的输入数据路径不正确的问题。
2. 修复：conv tdnn模式某些情况下配置生成不正确的问题。
3. 支持首节点自动插入正负优化。
	注意：对于超大网络可能造成内存不足，此时应当修改onnx模型文件，使其首层为PN后的weights、bias和输入。
4. 支持首节点的自动插入*2优化。
	注意：只支持dnn/tdnn/conv2d/deconv2d首节点自动插入，其他节点请手动*2输入。
5. 增加优化提示信息，对于添加输入优化后，需要手动对输入数据进行修改的，mapper会在log中进行提示。

## v001.000.026
1. 精度分析工具实测pipeline case

## v001.000.027
1. 添加自动优化,并定义
def npu_graph_build(mods, params, inputs, optimize_config):
    target = 'npu'
    target_host = 'npu'
    date_time = datetime.datetime.now()
    timestamp = date_time.strftime("%d_%b_%Y_%H_%M_%S_%f")
    build_dir = "./output/output_" + os.path.basename(__file__).split(".")[0] + timestamp
    optimize_path = ""
    if optimize_config and optimize_config.ByteSize() > 0:
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        optimize_path = os.path.join(build_dir, "optimize_dccrn.protobuf")
        with open(optimize_path, "w") as f:
            txt_opti = google.protobuf.text_format.MessageToString(optimize_config)
            f.write(txt_opti)

    with witin.transform.PassContext(opt_level=3):
        _, _, _, npu_graph = witin_frontend.build_module.build(
            mods, target=target, target_host=target_host, params=params,
            input_data=inputs, optimize_method_config=optimize_path,
            output_dir=build_dir, chip="BB04P1")
    assert npu_graph is not None
    ######################################################################
    # create npu module with npu_graph and run graph
    ######################################################################
    from tvm.contrib import npu_graph_runtime
    m = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
    for input_dict in inputs:
        for key, data in input_dict.items():
            m.set_input(key, data)

    return m

参考示例:
tests/onnx/witin/wtm2101/pipeline/test_auto_optimizer_tc_resnet_cnn.py
tests/onnx/witin/wtm2101/pipeline/test_auto_optimizer_dccrn_skip_connect.py
tests/onnx/witin/wtm2101/pipeline/test_forward_auto_opti_denoise.py

## v001.000.028
1. 添加自动优化,并定义softmax和sum_norm event
2. 修复PN时间的“net input”字段bug
参考示例:
tests/onnx/witin/wtm2101/operator/test_forward_softmax.py
tests/onnx/witin/wtm2101/operator/test_forward_sumnorm.py

## v001.000.029
1. 修复了网络输入节点被网络输出node接收时的timie_offset字段设置问题
参考示例：
ests/python/frontend/onnx/witin/wtm2101/customer/test_forward_pku_attention_softmax.py

## v001.000.030
1.在map路径下新增name.txt，以及map.json中新增“name”字段内容，定义规则：
“name”字段是长度为8的字符串，包含8个字符：[0:7]
如下为各个字符的含义：
[0:1]:占用两个字符，表示从2023年1月1日起到当前时间的月数目，存放的是阿拉伯数字转换为字符串的格式。
[2]:占用一个字符，表示当前月份的天数，可以表示1~31。用A~Z表示1~26，a~z表示27~31。
[3]:占用一个字符，表示当前的小时数，可以表示0~23。用A~X表示0~23
[4:5]:占用两个字符，表示当前的分数，可以表示0~59。用00~59的阿拉伯数字转换后的字符串表示。
[6:7]:占用两个字符，表示当前的秒数，可以表示0~59。用00~59的阿拉伯数字转换后的字符串表示。
例如：
假定当前系统时间是2023年6月27日11时22分32秒，则转换后的“name”字段的字符串为：06aK2232，其中：
06:表示从2023年1月1日起到当前时间的总月数是6个月；
a:表示本月的27日；
K:表示当天的11时；
22:表示11时的22分；
32:表示22分的32秒。

## v001.000.031
1.round_config长度做一个限制，长度设置成246， 留点空间给补偿和其他设置用

## v001.000.032
1.修复补偿配置，调整package_len

## v001.000.033
1.新增当dnn的输出channel>896时，支持输出拆分
参考示例：tests/onnx/witin/wtm2101/operator/test_gemm_array_extend.py
2.Matmul算子实现：计算两个矩阵的乘法。
矩阵乘法常见的有两种情况：
1）两个输入featureMap都是变量：在wtm2101中的支持需要在MCU上通过event来实现
2）第一个输入是变量，第二个输入时常量：在wtm2101中的支持需要在MPU上通过conv2d来实现。
 参考示例：tests/onnx/witin/wtm2101/operator/test_forward_matmul.py
3.GroupNorm event实现
参考示例：tests/onnx/witin/wtm2101/operator/test_forward_groupnorm.py
4."INPUT_NEGTIVE_OPT"调整为"INPUT_NEGATIVE_OPT"

## v001.000.034
1.修复多输入地址覆盖问题
2.增加用户接口：
用户接口有两个脚本user_json_generate.py和npu_build.py，其中
user_json_generate.py用来生成用户接口json,而npu_build.py调用用户接口json.
用户在user_json_generate.py直接修改user_json_format = User_Json_Format(models=...,shape_dicts=...,input_datas=...,optimize_config=...,
nets_type=...,build_dir=...)内容即可，在witin_mapper跟目录下按照如下命令执行：
python tests/python/frontend/onnx/witin/wtm2101/npu_build.py --json_file 所生成用户json,例如
python tests/python/frontend/onnx/witin/wtm2101/npu_build.py --json_file model/user_interface/params_mulnet_mulinput.json
用户接口json格式说明：包含"models"、"shape_dicts"、"input_datas"、"optimize_config"、
"nets_type"和"build_dir"，这几个字段都必须包含。
其中：
models是指定onnx模型的路径，可以是一个onnx模型，也可以多个；例如：
"models": [
"./model/user_interface/multi_net_multi_input/hw_new_cnn_net_left.onnx",
"./model/user_interface/multi_net_multi_input/hw_new_cnn_net_right.onnx",
"./model/user_interface/multi_net_multi_input/hw_new_gemm_net.onnx"
],
"shape_dicts"是由onnx模型的输入节点和输入shap构成的字典，支持多输入和多个网络的输入，例如：
"shape_dicts": [
{
"left_in": [
1,
8,
280,
12
]
},
{
"right_in": [
1,
1,
64,
12
]
},
{
"in1": [
1,
768
],
"in2": [
1,
768
]
}
],
"input_datas"指的是输入数据，可支持多个网络的输入和多输入；例如：
1）数据格式要求为.npy;
2) DCCRN网络输入要求维度为[1,C,H,W],
3) 卷积网络的输入要求维度为[N,C,H,W],(N>100)
"input_datas": [
"./model/user_interface/multi_net_multi_input/conv_left.npy",
"./model/user_interface/multi_net_multi_input/conv_right.npy",
"./model/user_interface/multi_net_multi_input/gemm_left.npy",
"./model/user_interface/multi_net_multi_input/gemm_right.npy"
],
总而言之，"models"、"shape_dicts"和"input_datas"顺序要对应起来。
"optimize_config"指优化文件；例如，"./model/user_interface/multi_net_multi_input/hua_dw_tc_opti.protobuf"
"nets_type"：指定网络类型，

如果首层是卷积，需指定CONV;
如果是DCCRN网络，需要指定首层卷积核的kw参数，例如"DCCRN_2",2在此就表示kw，只有DCCRN会报错;
如果是TDNN网络，需要指定首层time-offset，例如"TDNN_3",3在此就表示time-offset，只有TDNN会报错。
"nets_type"数量要求与"models"数量和顺序保持一致。
"build_dir"：指输出路径，例如："build_dir": "test_cnn_resnet"

示例：详见model/user_interface/下
3.修复多网络case中每个网络补偿配置生成时没有清空补偿激活表的问题
4.修复GRU精度测试示例

## v001.000.035
作为特定版本发布

## v001.000.036
1.支持transpose event
2.支持batchnorm_1d event
构图过程：
linear1_bn_node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['linear1_scale_out', 'linear1_scale', 'linear1_bias', 
                'linear1_input_mean', 'linear1_input_var'],
        outputs=['linear1_bn_out'],
        epsilon=1e-5,
        scale_in=0.125,
        scale_out=0.125
    )
参考示例：
tests/python/frontend/onnx/witin/wtm2101/operator/test_fordward_batchnorm.py

## v001.000.037
支持GRU,接口新支持如下：
gru_node1 = onnx.helper.make_node('Gru',
                                      inputs=['relu_out', 'gru_weight1', 'gru_weight2', 'gru_weight3', 'gru_bias1', 'gru_bias2', 'gru_bias3'],
                                      scale_zr=1024,
                                      scale_ht=1024,
                                      scale_in=1024,
                                      scale_ones = 127,
                                      activate_type=['sigmoid', 'tanh'],
                                      activate_table=act_table,
                                      shift_bits=[-7, -7],
                                      clean_ht=0,
                                      outputs=['out'],
                                      name="gru_node1")

其中：
inputs list[tensor]输入、zrt权重、hn权重、in权重、zrt偏置、hn偏置、in偏置
scale_zr float 表示zrt部分G值：
scale_ht float 表示hn部分G值
scale_in float 表示in部分G值
activate_typelist[string]激活表的类型，只能为sigmoid/tanh
activate_table tensor激活表，每个激活表长度为256，或者1024，目前建议取长度为1024
shift_bitslist[int]长度为2，
[0]为对rt(8bit)*hn(8bit)的结果(16bit)移位，
[1]为对[(1-zt)*ht](16bit)+[zt*ht~](16bit)的结果ht(16bit)移位，
不设置默认为[-7,-7],-7表示右移七位，-代表右移位。
outputs list[tensor]输出Tensor
name string节点名称
scale_ones：表示“1”和zt的缩放尺度，目前固定取值为127
clear_ht:gru执行如果次数后，将ht清零，等于0时，代表不清除ht

## v001.000.038
1.优化batchnorm 

## v001.000.039
1.修复gru pn优化

## v001.000.040
1.修复gru,支持scale_ones=128

## v001.000.041
1.增加floatScale算子，具体用法如下：
floatScale_node = onnx.helper.make_node("FloatScale",
										inputs=['floatscale_input'],
										outputs=['floatscale_output'],
										scale=scale_value,
										clip_bits=clip_bit_num)


其中，单节点的输入和单节点的输出，scale为浮点scale，clip_bit为截断bit位数。	

forward过程：
out =  (int)(in * float_scale + 0.5) //需要加 0.5，以达到四舍五入的目的
out = out.clip(min=-2^clip_bits , max=2^clip_bits -1) //clip_bits， 就是截断的bit数

## v001.000.042
1.修复floatScale算子

## v001.000.043
1.新增ReduceMean算子，具体用法如下：
reduceMean_node = onnx.helper.make_node("ReduceMean", 
										inputs=["conv1_relu_out"], 
										axes=[1], 
										keepdims=True, 
										outputs=["reduceMean_out"])
参考示例：tests/onnx/witin/wtm2101/operator/reduceMean/test_reduceMean_forward.py
2.修复Maxpool算子

## v001.000.044
1.调整ReduceMean算子

## v001.000.045
1.调整weight复制优化
2.新增ClearLowBits,具体用法如下：
floatScale_node = onnx.helper.make_node("ClearLowBits",
										inputs=['input'],
										outputs=['output'],
										bits=3)
python实现code:
def hard_clean_bits(x, bit):
	x = int(2**(bits-1))
	out = torch.where(x>0, x+v, x-v)
	out = torch.trunc(out>>bits) << bits
	return out

## v001.000.046
1.增加GRU优化

## v001.001.000
1.新增双权重GRU部署方式，其接口如下：
gru_node1 = onnx.helper.make_node('Gru2Array',
        inputs=['in', 'gru_params1', 'gru_params2', 'gru_bias1', 'gru_bias2'],
        input_offset=0,
        hidden_shift_bit=-1,   #输入scle
        scale_input=1024,
        scale_hidden=1024,
        scale_ones = 128,
        activate_type=['sigmoid', 'tanh'],
        activate_table=act_table,
        shift_bits=[-7, -7, -2, -3], 
        clean_ht=0,
        outputs=['out'],
        name="gru_node1")

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

## v001.001.002
优化gru的地址重复分配，可以重复使用


## v001.001.003
1.适配gru2array的PN优化，inputX2优化,weightX2优化，并在模拟器上验证
2.适配gru2array+reshape+conv结构，并在模拟器上进行验证
3.增加tdnn_mode=true时，不适用于kw=1的情况，这种情况下请使用conv mode

## v001.001.004
gru2array:
1.优化array空间分配
2.增加event:ht从SRAM拷贝到regfile和ht从regfile拷贝到SRAM
DNN:
修复dnn作为网络最后一层的输出地址覆盖输入地址的情况

## v001.001.005
优化gru2array的round结构

## v001.001.006
适配双gru2array的网络结构

## v001.001.007
支持dnn/conv输出通道大于array边界（大于1024，实际可用896，因为需避开前128列）时的自动拆分，增加concat输出配置
