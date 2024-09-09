# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Minimum npu graph runtime that executes graph containing TVM PackedFunc."""
import numpy as np
import tvm._ffi

from tvm.rpc import _ffi_api as _rpc_ffi_api
from tvm.rpc import base as rpc_base
from tvm._ffi.base import string_types
from tvm._ffi.runtime_ctypes import TVMContext
from .. import nd as _nd
import subprocess


def create(graph_, chip=None, libmod=None, ctx=None, output_dir="./output/"):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_r : wtGraph

    libmod : tvm.runtime.Module
        default to None

    ctx : TVMContext or list of TVMContext
        The context to deploy the module. It can be local or remote when there
        is only one TVMContext. Otherwise, the first context in the list will
        be used as this purpose. All context should be given for heterogeneous
        execution.

    Returns
    -------
    graph_module :  NpuGraphModule
        Runtime graph module that can be used to execute the graph.
    """
    if output_dir[-1] != "/":
        output_dir = output_dir + "/"
    fcreate = tvm.get_global_func("tvm.npu_graph_runtime.create")
    modules = fcreate(graph_, chip, output_dir)
    npu_modules = []
    for m in modules:
        npu_modules.append(NpuGraphModule(m))
    if len(npu_modules) == 1:
        return npu_modules[0]
    else:
        return npu_modules


def create_from_json(json_path, libmod=None, ctx=None):
    """Create a runtime executor module given a json config file and module.

    Parameters
    ----------
    json_path : json_path

    libmod : tvm.runtime.Module
        default to None

    ctx : TVMContext or list of TVMContext
        The context to deploy the module. It can be local or remote when there
        is only one TVMContext. Otherwise, the first context in the list will
        be used as this purpose. All context should be given for heterogeneous
        execution.

    Returns
    -------
    graph_module :  NpuGraphModule
        Runtime graph module that can be used to execute the graph.
    """
    fcreate = tvm.get_global_func("tvm.npu_graph_runtime.create_from_json")
    return NpuGraphModule(fcreate(json_path))


def runCmd(cmd):
    res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sout, serr = res.communicate()  #该方法和子进程交互，返回一个包含 输出和错误的元组，如果对应参数没有设置的，则无法返回
    return res.returncode, sout, serr, res.pid  #可获得返回码、输出、错误、进程号；


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a).astype(np.float32)
    vector_b = np.mat(vector_b).astype(np.float32)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        #the out is all equal 0
        if np.sum(vector_a) == np.sum(vector_b):
            return 1
        else:
            return 0
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def distance(a, b):
    v1 = np.sqrt(np.sum((np.int32(b) - np.int32(a))**2))
    v2 = np.sqrt(np.sum(1e-5 + np.int32(b)**2))
    v3 = v1 / v2
    ret = np.sum(v3)
    # print("compare distance is:%.4f"%(ret))
    return ret


def _judge_input_is_uint(param):
    if isinstance(param, np.ndarray):
        param = _nd.array(param)
    param = param.asnumpy()
    min_data = param.min()
    max_data = param.max()
    # int8/int16
    if (min_data < 0):
        if (min_data < -32768 or max_data > 32767):
            raise ValueError(
                "input data can not < -32768 or > 32767, now input min is {},please make sure you input data".format(
                    min_data))
    else: # uint8
        if (max_data > 255):
            raise ValueError("input data can not > 255, now input max is {}".format(max_data))


class NpuGraphModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.

    Attributes
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.
    """

    def __init__(self, module, chip=None):
        self.module = module
        self.chip = chip
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._func_run = module["FuncRun"]
        self._get_op_array_inputs = module["GetOpArrayInputs"]
        self._get_op_array_outputs = module["GetOpArrayOutputs"]
        self._get_op_initializer = module["GetOpInitializer"]
        self._get_op_scale = module["GetOpScale"]
        self._get_num_array_inputs = module["GetOpArrayInputNb"]
        self._get_num_array_outputs = module["GetOpArrayOutputNb"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_num_inputs = module["get_num_inputs"]
        self._set_frame = module["set_frame"]
        #self._set_chip = module["set_chip"]
        self._get_output_dir = module["get_output_dir"]
        # In our project, integers are converted to floating-point as input for runtime sometimes, 
        # so it's necessary to know the storage data type of the input ndarrays.
        self._set_input_storage_type = module["SetInputStorageType"]

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        if key is not None:
            _judge_input_is_uint(value)
            self._set_input_storage_type(key, value.dtype)
            self._set_frame(key, value.shape[0])
            v = self._get_input(key)
            if v is None:
                raise RuntimeError("Could not find '%s' in graph's inputs" % key)
            v.copyfrom(value)

        if params:
            # upload big arrays first to avoid memory issue in rpc mode
            keys = list(params.keys())
            keys.sort(key=lambda x: -np.prod(params[x].shape))
            for k in keys:
                # TODO(zhiics) Skip the weights for submodule in a better way.
                # We should use MetadataModule for initialization and remove
                # params from set_input
                val = self._get_input(k)
                if val:
                    self._get_input(k).copyfrom(params[k])

    def run(self, threshold=0.99, **input_dict):
        """Run forward execution of the graph

        Parameters
        ----------
        threshold: cos similarity, default valueis 0.99
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()
        """
            运行结束后,比较功能仿真器的结果和cmodel的运行结果
            方法：余弦相似度
        """
        import os
        import math
        output_dir = self._get_output_dir()
        count = 0
        for root, dirs, files in os.walk(output_dir + "/function_sim_output"):
            for each in dirs:
                count += 1

        for net_idx in range(count):
            func_sim_output_path = output_dir + "/function_sim_output/net" + str(net_idx) + "/"
            cmodel_output_prefix = output_dir + "/simulator_output/net" + str(
                net_idx) + "/layer_debug/"
            func_sim_output_files = os.listdir(func_sim_output_path)
            func_sim_output_files = sorted(func_sim_output_files)

            for i, val in enumerate(func_sim_output_files):
                path1 = func_sim_output_path + val
                path2 = cmodel_output_prefix + val
                round_data1 = []
                round_data2 = []
                with open(path1, 'r+') as rwf1:
                    line = rwf1.readline()
                    line = line[:-1]  #remove "\n"
                    data = line.split(" ")
                    round_data1 = [int(i) for i in data]
                    round_data1 = np.array(round_data1)
                with open(path2, 'r+') as rwf2:
                    line = rwf2.readline()
                    line = line[:-1]  #remove and " "
                    data = line.split(" ")
                    round_data2 = [int(i) for i in data]
                    round_data2 = np.array(round_data2)

                max = np.max(round_data1)
                min = np.min(round_data1)
                max2 = np.max(round_data2)
                min2 = np.min(round_data2)
                if max < max2:
                    max = max2
                if min > min2:
                    min = min2
                data_type = "int8"
                if (max > 255 or min < -256):
                    data_type = "int16"
                    threshold = 0.95 if threshold > 0.95 else threshold

                if data_type == "int8":
                    similarity1 = cos_sim(round_data1.astype(np.uint8), round_data2.astype(np.uint8))
                    similarity2 = cos_sim(round_data1.astype(np.int8), round_data2.astype(np.int8))
                else:
                    similarity1 = cos_sim(round_data1.astype(np.uint16), round_data2.astype(np.uint16))
                    similarity2 = cos_sim(round_data1.astype(np.int16), round_data2.astype(np.int16))

                if similarity1 > similarity2:
                    similarity = similarity1
                else:
                    similarity = similarity2
                input_num = str(int(val[18:22], 10))
                round_num = str(int(val[-6:-4], 10))
                if math.isnan(similarity):
                    raise ValueError("similarity is nan, input %s, round %s" %
                                     (input_num, round_num))
                # print("input %s, round %s, similarity is  '%f' " % (input_num, round_num, similarity))
                if similarity < threshold:
                    print(round_data1)
                    print(round_data2)
                    diff = (round_data1 - round_data2).astype(np.int8)
                    print("diff\n", diff)
                    raise ValueError("net %d, input %s, round %s, similarity is  '%f' " %
                                     (net_idx, input_num, round_num, similarity))
            print("func simulation result is equal to cmodel result!")

    def func_run(self, **input_dict):
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._func_run()

    def get_num_array_inputs(self, op_name):
        """Get op's array inputs number

        Returns
        -------
        count : int
            The number of array inputs.
        """
        return self._get_num_array_inputs(op_name)


    def get_num_array_outputs(self, op_name):
        """Get op's array outputs number

        Returns
        -------
        count : int
            The number of array outputs.
        """
        return self._get_num_array_outputs(op_name)

    def get_op_array_inputs(self, op_name, index):
        """Get op's array inputs

        Returns
        -------
        return value: ndarray
            The op's array inputs
        """
        return self._get_op_array_inputs(op_name, index)

    def get_op_array_outputs(self, op_name, index):
        """Get op's specified array outputs

        Returns
        -------
        return value: ndarray
            The op's array outputs.
        """
        return self._get_op_array_outputs(op_name, index)

    def get_op_initializer(self, op_name, index):
        """Get op's specified initializer

        Returns
        -------
        return value: ndarray
            The op's initializer.
        """
        return self._get_op_initializer(op_name, index)

    def get_op_scale(self, op_name, index):
        """Get op's specified scale

        Returns
        -------
        return value: The op's scale.
        """
        return self._get_op_scale(op_name, index)

    def get_num_outputs(self):
        """Get the number of outputs from the graph

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_num_inputs(self):
        """Get the number of inputs to the graph

        Returns
        -------
        count : int
            The number of inputs.
        """
        return self._get_num_inputs()

    def get_input(self, index, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(index).copyto(out)
            return out

        return self._get_input(index)

    def get_output(self, index, out=None):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index

        out : NDArray
            The output array container
        """
        if out:
            self._get_output(index, out)
            return out

        return self._get_output(index)

    def debug_get_output(self, node, out):
        """Run graph up to node and get the output to out

        Parameters
        ----------
        node : int / str
            The node index or name

        out : NDArray
            The output array container
        """
        raise NotImplementedError("Please use debugger.debug_runtime as graph_runtime instead.")

    def __getitem__(self, key):
        """Get internal module function

        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.module[key]
