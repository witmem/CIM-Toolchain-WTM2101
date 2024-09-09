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
"""
Construct the necessary state for the TVM graph runtime
from a Relay expression.
"""
import warnings
import numpy as np

from tvm.ir import IRModule

from tvm.tir import expr as tvm_expr
from .. import nd as _nd, target as _target, autotvm
from ..contrib import graph_runtime as _graph_rt
from . import _build_module
from . import _build_module_npu
from . import ty as _ty
from . import expr as _expr
from . import function as _function
from .backend import graph_runtime_factory as _graph_runtime_factory
from .backend import interpreter as _interpreter
from .backend.vm import VMExecutor
import subprocess
import glob

import os
import sys
tvm_home = os.environ['TVM_HOME']
sys.path.append(tvm_home + '/python/tvm/relay/witin/chips/build/lib.linux-x86_64-3.6')

# import draw_array_space as _draw_array_space
# import get_weight_bias_hist_map as _get_weight_bias_hist_map
from tvm.relay.witin.chips_source.draw_array_space import draw_array_space
from tvm.relay.witin.chips_source.get_weight_bias_hist_map import witin_get_weight_bias_hist_map

import os
import shutil
import logging
logging.getLogger().setLevel(logging.INFO)

def _update_target(target):
    target = target if target else _target.Target.current()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")

    tgts = {}
    if isinstance(target, (str, _target.Target)):
        dev_type = tvm_expr.IntImm("int32", _nd.context(str(target)).device_type)
        tgts[dev_type] = _target.create(target)
    elif isinstance(target, dict):
        for dev, tgt in target.items():
            dev_type = tvm_expr.IntImm("int32", _nd.context(dev).device_type)
            tgts[dev_type] = _target.create(tgt)
    else:
        raise TypeError("target is expected to be str or " +
                "tvm.target.Target, but received " +
                "{}".format(type(target)))
    return tgts


def _convert_param_map(params):
    inputs = {}
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = _nd.array(param)
        inputs[name] = _expr.const(param)
    return inputs

def _judge_input_is_uint(params):
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = _nd.array(param)
        param = param.asnumpy()
        min_data = param.min()
        max_data = param.max()
        # int8
        if (min_data < 0):
            if (min_data < -128 or max_data > 127):
                raise ValueError("input data is int8, now input min is {},input max is {}".format(min_data,max_data))
        if (max_data > 255):
            raise ValueError("input data is uint8 can not > 255, now input max is {}".format(max_data))

class BuildModule(object):
    """Build an IR module to run on TVM graph runtime. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """
    def __init__(self):
        self.mod = _build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._optimize = self.mod["optimize"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]

    def build(self, mod, target=None, target_host=None, params=None):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IRModule to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
        device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        target_host : str or :any:`tvm.target.Target`, optional
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            to setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm intepreter is used.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        Returns
        -------
        graph_json : str
            The json string that can be accepted by graph runtime.

        mod : tvm.Module
            The module containing necessary libraries.

        params : dict
            The parameters of the final graph.
        """
        target = _update_target(target)
        # Setup the params.
        if params:
            self._set_params(params)
        # Build the IR module
        self._build(mod, target, target_host)
        # Get artifacts
        graph_json = self.get_json()
        mod = self.get_module()
        params = self.get_params()

        return graph_json, mod, params

    def optimize(self, mod, target=None, params=None):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IR module to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
        device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        Returns
        -------
        mod : :py:class:`~tvm.IRModule`
            The optimized relay module.

        params : dict
            The parameters of the final graph.
        """
        target = _update_target(target)

        # Setup the params.
        if params:
            self._set_params(params)
        mod = self._optimize(mod, target)
        # Get artifacts
        params = self.get_params()

        return mod, params


    def _set_params(self, params):
        self._set_params_func(_convert_param_map(params))

    def get_json(self):
        """Return the json file of the built program."""
        return self._get_graph_json()

    def get_module(self):
        """Return the built module."""
        return self._get_module()

    def get_params(self):
        """Return the updated weights."""
        params = self._get_params_func()
        ret = {}
        for key, value in params.items():
            ret[key] = value.data
        return ret


def runCmd(cmd) :
    res = subprocess.Popen(cmd, shell=True,  stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    sout ,serr = res.communicate() #该方法和子进程交互，返回一个包含 输出和错误的元组，如果对应参数没有设置的，则无法返回
    return res.returncode, sout, serr, res.pid #可获得返回码、输出、错误、进程号；


def setDir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        for clean_up in glob.glob(file_path + "/*"):
            if not clean_up.endswith('.protobuf'):
                if os.path.isdir(clean_up):
                    shutil.rmtree(clean_up)
                elif os.path.isfile(clean_up):
                    os.remove(clean_up)


class BuildModuleNpu(object):
    """Build an IR module to run on TVM graph runtime. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """
    def __init__(self):
        self.mod = _build_module_npu._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._build_mods = self.mod["build_mods"]
        self._optimize = self.mod["optimize"]
        self._set_params_func = self.mod["set_params"]
        self._set_net_params_func = self.mod["set_net_params"]
        self._get_params_func = self.mod["get_params"]
        self._set_calibration_input_data = self.mod["set_calibration_input_data"]
        self._set_net_calibration_input_data = self.mod["set_net_calibration_input_data"]
        self._get_npu_graph = self.mod["get_npu_graph"]
        self._set_npu_graph = self.mod["set_npu_graph"]
        self._set_sigmoid_quan_mode = self.mod["set_sigmoid_quan_mode"]
        self._set_optimize_method = self.mod["set_optimize_method"]
        self._set_output_dir = self.mod["set_output_dir"]
        self._set_array_distribute = self.mod["set_array_distribute"]

    def build(self,
              mod,
              target=None,
              target_host=None,
              params=None,
              input_data=None,
              chip="BB04P1",
              build_config=None,
              optimize_method_config="",
              output_dir="./output/",
              array_distribute=""):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IRModule to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
        device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        target_host : str or :any:`tvm.target.Target`, optional
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            to setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm intepreter is used.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        input_data： input data, dict of str to NDArray
            input data for witin array calibration
        build_config: build config for witin unique

        Returns
        -------
        graph_json : str
            The json string that can be accepted by graph runtime.

        mod : tvm.Module
            The module containing necessary libraries.

        params : dict
            The parameters of the final graph.
        """
        target = _update_target(target)
        # clear and re-create
        setDir(output_dir)
        setDir(output_dir + "map")
        if array_distribute == "":
            setDir(output_dir + "profile")
            setDir(output_dir + "memory")
            setDir(output_dir + "simulator_input")
            setDir(output_dir + "simulator_input_txt")
            setDir(output_dir + "simulator_output")
            setDir(output_dir + "function_sim_output")
        # Setup the params.
        if params:
            self._set_net_params(params)
        # Setup the input data.
        net_input_list = []
        if isinstance(input_data, list):
            for idata in input_data:
                if idata:
                    net_input_list.append(_convert_param_map(idata))
                    # _judge_input_is_uint(idata)
        self._set_net_calibration_input_data(net_input_list)
        self._set_optimize_method(optimize_method_config)
        self._set_output_dir(output_dir)
        self._set_array_distribute(array_distribute)
        # if build_config is not None:
        #     if build_config['calibration_data'] is not None:
        #        self._set_calibration_input_data(_convert_param_map(input_data))
        #     if build_config['sg_data'] is not None:
        #        self._set_sigmoid_quan_mode(_convert_param_map(build_config['sg_data']))
        # Build the IR module
        # multi mod
        stat = self._build_mods(mod, target, target_host, chip)
        if stat < 0:
            logging.info("Failed to alloc array & bias")
            return None, None, None, None
        draw_array_space(output_dir + "map/layers.txt", output_dir + "map/")
        #self._build(mod, target, target_host, chip)
        # Get artifacts
        graph_json = []
        mod = []
        params = []
        npu_graph = self._get_npu_graph()

        if array_distribute == "":
            # Generate numerical distribution of weight parameters
            witin_get_weight_bias_hist_map(output_dir + "params", output_dir + "map/weight_hist")

        return graph_json, mod, params, npu_graph

    def optimize(self, mod, target=None, params=None):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IR module to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
        device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        Returns
        -------
        mod : :py:class:`~tvm.IRModule`
            The optimized relay module.

        params : dict
            The parameters of the final graph.
        """
        target = _update_target(target)

        # Setup the params.
        if params:
            self._set_params(params)
        mod = self._optimize(mod, target)
        # Get artifacts
        params = self.get_params()

        return mod, params

    def _set_net_params(self, params):
        params_list = []
        for net_params in params:
            pts = _convert_param_map(net_params)
            params_list.append(pts)
        self._set_net_params_func(params_list)

    def _set_params(self, params):
        self._set_params_func(_convert_param_map(params))

    def get_json(self):
        """Return the json file of the built program."""
        return self._get_graph_json()

    def get_module(self):
        """Return the built module."""
        return self._get_module()

    def get_params(self):
        """Return the updated weights."""
        params = self._get_params_func()
        ret = {}
        for key, value in params.items():
            ret[key] = value.data
        return ret


def build(mod,
          target=None,
          target_host=None,
          params=None,
          input_data=None,
          build_config=None,
          chip="BB04P1",
          mod_name='default',
          optimize_method_config="",
          output_dir="./output/",
          array_distribute=""):
    """Helper function that builds a Relay function to run on TVM graph
    runtime.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The IR module to build. Using relay.Function is deprecated.

    target : str, :any:`tvm.target.Target`, or dict of str(i.e. device/context
    name) to str/tvm.target.Target, optional
        For heterogeneous compilation, it is a dictionary indicating context to
        target mapping. For homogeneous compilation, it is a build target.

    target_host : str or :any:`tvm.target.Target`, optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    input_data: input data
        input data for witin array calibration

    build_config: build_config dict
        build_config for witin unique

    mod_name: Optional[str]
        The module name we will build

    Returns
    -------
    graph_json : str
        The json string that can be accepted by graph runtime.

    mod : tvm.Module
        The module containing necessary libraries.

    params : dict
        The parameters of the final graph.
    """
    mod_list = []
    params_list = []
    input_data_list = []
    if not isinstance(mod, list):
        mod_list.append(mod)
        params_list.append(params)
        input_data_list.append(input_data)
        #raise ValueError("Type of input parameter mod must be tvm.IRModule list")
    else:
        mod_list = mod
        params_list = params
        input_data_list = input_data
    #if not isinstance(mod, (IRModule, _function.Function)):
    #    raise ValueError("Type of input parameter mod must be tvm.IRModule")
    mods = []
    for md in mod_list:
        if isinstance(md, _function.Function):
            if params:
                md = bind_params_by_name(md, params)
            md = IRModule.from_expr(md)
            mods.append(md)
            warnings.warn(
                "Please use input parameter mod_list (tvm.IRModule) "
                "instead of deprecated parameter mod_list (tvm.relay.function.Function)",
                DeprecationWarning)
        else:
            mods.append(md)
    target_str = target
    target = _update_target(target)

    if isinstance(target_host, (str, _target.Target)):
        target_host = _target.create(target_host)
    elif target_host:
        raise ValueError("target host must be the type of str, " + "tvm.target.Target, or None")

    if output_dir[-1] != "/":
        output_dir = output_dir + "/"

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.util.EmptyContext()

    #BuildModuleNpu
    with tophub_context:
        if target_str == "npu":
            bld_mod_npu = BuildModuleNpu()
            graph_json, mod, params, npu_graph = bld_mod_npu.build(
                mods,
                target,
                target_host,
                params_list,
                input_data_list,
                build_config=build_config,
                chip=chip,
                optimize_method_config=optimize_method_config,
                output_dir=output_dir,
                array_distribute=array_distribute)
            cmd = "cp  ./build/BoardConfig.json ./output/"
            res = runCmd(cmd)
            cmd = "cp  ./build/params.dat ./output/"
            res = runCmd(cmd)
            return graph_json, mod, params, npu_graph
        else:
            bld_mod = BuildModule()
            graph_json, mod, params = bld_mod.build(mod, target, target_host, params)
            mod = _graph_runtime_factory.GraphRuntimeFactoryModule(graph_json, mod, mod_name, params)
            return mod


def optimize(mod, target=None, params=None):
    """Helper function that optimizes a Relay module.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The module to build. Using relay.Function is deprecated.

    target : str, :any:`tvm.target.Target`, or dict of str(i.e. device/context
    name) to str/tvm.target.Target, optional
        For heterogeneous compilation, it is a dictionary indicating context to
        target mapping. For homogeneous compilation, it is a build target.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    Returns
    -------
    mod : :py:class:`~tvm.IRModule`
        The optimized relay module.

    params : dict
        The parameters of the final graph.
    """
    if not isinstance(mod, (IRModule, _function.Function)):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    if isinstance(mod, _function.Function):
        if params:
            mod = bind_params_by_name(mod, params)
        mod = IRModule.from_expr(mod)
        warnings.warn(
                "Please use input parameter mod (tvm.IRModule) "
                "instead of deprecated parameter func (tvm.relay.function.Function)",
                DeprecationWarning)

    target = _update_target(target)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.util.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        mod, params = bld_mod.optimize(mod, target, params)
    return mod, params


def bind_params_by_name(func, params):
    """Bind params to function by name.
    This could be useful when assembling custom Relay optimization
    passes that involve constant folding.

    Parameters
    ----------
    func : relay.Function
        The function to bind parameters to.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    Returns
    -------
    func : relay.Function
        The function with parameters bound
    """
    inputs = _convert_param_map(params)
    return _build_module.BindParamsByName(func, inputs)


class GraphExecutor(_interpreter.Executor):
    """Wrapper around Executor interface.

    This executor is used for debug and testing purpoes.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The module to support the execution.

    ctx : :py:class:`TVMContext`
        The runtime context to run the code on.

    target : :py:class:`Target`
        The target option to build the function.
    """

    def __init__(self, mod, ctx, target):
        assert mod is not None
        self.mod = mod
        self.ctx = ctx
        self.target = target

    def _make_executor(self, expr=None):
        if expr:
            self.mod["main"] = expr
        ret_type = self.mod["main"].checked_type.ret_type
        if _ty.is_dynamic(ret_type):
            raise ValueError("Graph Runtime only supports static graphs, got output type",
                    ret_type)
        num_outputs = len(ret_type.fields) if isinstance(ret_type, _ty.TupleType) else 1
        mod = build(self.mod, target=self.target)
        gmodule = _graph_rt.GraphModule(mod['default'](self.ctx))

        def _graph_wrapper(*args, **kwargs):
            args = self._convert_args(self.mod["main"], args, kwargs)
            # Create map of inputs.
            for i, arg in enumerate(args):
                gmodule.set_input(i, arg)
            # Run the module, and fetch the output.
            gmodule.run()
            # make a copy so multiple invocation won't hurt perf.
            if num_outputs == 1:
                return gmodule.get_output(0).copyto(_nd.cpu(0))
            outputs = []
            for i in range(num_outputs):
                outputs.append(gmodule.get_output(i).copyto(_nd.cpu(0)))
            return outputs

        return _graph_wrapper


def create_executor(kind="debug",
        mod=None,
        ctx=None,
        target="llvm"):
    """Factory function to create an executor.

    Parameters
    ----------
    kind : str
        The type of executor

    mod : :py:class:`~tvm.IRModule`
        The Relay module containing collection of functions

    ctx : :py:class:`tvmContext`
        The context to execute the code.

    target : :py:class:`tvm.Target`
        The corresponding context
    """
    if mod is None:
        mod = IRModule()
    if ctx is not None:
        assert ctx.device_type == _nd.context(str(target), 0).device_type
    else:
        ctx = _nd.context(str(target), 0)

    if isinstance(target, str):
        target = _target.create(target)
    if kind == "debug":
        return _interpreter.Interpreter(mod, ctx, target)
    if kind == "graph":
        return GraphExecutor(mod, ctx, target)
    if kind == "vm":
        return VMExecutor(mod, ctx, target)
    raise RuntimeError("unknown execution strategy: {0}".format(kind))
