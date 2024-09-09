# p. Licensed to the Apache Software Foundation (ASF) under one
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
import numpy as np
import math
import os
from . import optiConfig_pb2
import logging
import tempfile
import google.protobuf
import google.protobuf.text_format
import tvm
import witin
import copy
from witin import *
from tvm.relay import build_module
from tvm.contrib import npu_graph_runtime
logging.getLogger().setLevel(logging.INFO)


class Optimizer:
    op_type_scale = "Scale"
    op_type_gemm = "Gemm"
    op_type_tdnn = "Tdnn"
    op_type_conv = "Conv"
    op_type_convT = "ConvTranspose"
    op_type_lstm = "Lstm"
    op_type_gru = "Gru"

    array_id_dict = {
        op_type_gemm: [[1], [2]],
        op_type_tdnn: [[1], [2]],
        op_type_conv: [[1], [2]],
        op_type_convT: [[1], [2]],
        op_type_lstm: [[1, 2], [3, 4]],
        op_type_gru: [[1, 2], [3, 4]],
    }
    g_up_limit = 8192
    g_down_limit = 256
    weight_limit = 256
    bias_limit = 16*275*128
    input_limit = 255
    output_limit = 128
    array_max_row = 896

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name


    def __init__(self, onnx_models, shape_dicts, inputs, manual_opt_conf):
        self.mods = []
        self.params = []
        for i in range(len(onnx_models)):
            mod, param = witin_frontend.frontend.from_onnx(onnx_models[i], shape_dicts[i])
            self.mods.append(mod)
            self.params.append(param)
        self.nodes = {}
        self.init_tensors = {}
        self.ops_array_params = {}
        self.ops_array_options = {}
        self.ops_optimize_params = []
        self.optimize_confs = []
        self.graphs = [model.graph for model in onnx_models]
        self.inputs = inputs
        self.branch_ops = []

        manual_opt = None
        if manual_opt_conf is not None:
            with open(manual_opt_conf, "r") as f:
                manual_opt = google.protobuf.text_format.Parse(f.read(), optiConfig_pb2.OptimizeConfig())

        with tempfile.TemporaryDirectory() as build_dir:
            npu_graph = self.npu_graph_build(self.mods, self.params, inputs, manual_opt, build_dir)
            assert npu_graph is not None
            runtime_model = npu_graph_runtime.create(npu_graph, "BB04P1", output_dir=build_dir)
            assert runtime_model is not None
            for input_dict in inputs:
                for key, data in input_dict.items():
                    runtime_model.set_input(key, data)
            runtime_model.func_run()

        input_users = {}
        for i in range(len(self.graphs)):
            for node in self.graphs[i].node:
                node_name = self._parse_value_proto(node)
                for op_input in node.input:
                    if op_input in input_users:
                        input_users[op_input].append(node_name)
                    else:
                        input_users[op_input] = [node_name]

            for init_tensor in self.graphs[i].initializer:
                self.init_tensors[init_tensor.name] = init_tensor
            for node in self.graphs[i].node:
                if node.name:
                    self.nodes[node.name] = node
            self.graph_analysys(self.graphs[i], runtime_model)

        self.branch_ops = [v for v in input_users.values() if len(v) > 1]

        self.op_analysys()
        self.show_op_optimizer_options()
        self.get_optimize_params()
        self.gen_optimize_confs(manual_opt)

    def get_branch_op(self, curr_op):
        branch = None
        for iter_branch in self.branch_ops:
            if curr_op in iter_branch:
                branch = copy.deepcopy(iter_branch)
                branch.remove(curr_op)
                return branch[0]

        return None

    def update_graphs_inputs(self, opt_conf, graphs_inputs):
        if not opt_conf.ByteSize():
            return graphs_inputs
        updated_graphs_inputs = []
        for graph_inputs in graphs_inputs:
            updated_graph_inputs = {}
            for k, v in graph_inputs.items():
                for conf in opt_conf.layerOptiConfig:
                    if conf.name != k:
                        continue
                    for input_opt in conf.inputOpti:
                        ctx = tvm.context("cpu", 0)
                        v = np.multiply(v.asnumpy(), 1 << input_opt.shift)
                        v = tvm.nd.array(v, ctx=ctx)
                updated_graph_inputs[k] = v
            updated_graphs_inputs.append(updated_graph_inputs)
        return updated_graphs_inputs

    def show_op_optimizer_options(self):
        with open("./output/auto_optimization_analysis_result.txt", "w") as f:
            for key, options in self.ops_array_options.items():
                f.write("Op: {0}\n".format(key))
                for array_id in range(len(options)):
                    option = options[array_id]
                    f.write("Array Index: {0}\n".format(array_id))
                    for item in option:
                        f.write("Input Magnify: {0} Weight Magnify: {1}\
                                Weight BroadCast: {2} Output Magnify: {3}\n"
                                .format(item[0], item[1], item[2], item[3]))

    def npu_graph_build(self, mods, params, inputs, optimize_conf, build_dir):
        target = 'npu'
        target_host = 'npu'
        with tempfile.TemporaryDirectory() as opt_dir:
            optimize_path = ""
            if optimize_conf and optimize_conf.IsInitialized():
                optimize_path = os.path.join(opt_dir, "optimize_method.pb")
                with open(optimize_path, "w") as f:
                    txt = google.protobuf.text_format.MessageToString(
                        optimize_conf)
                    f.write(txt)

            with witin.transform.PassContext(opt_level=3):
                _, _, _, npu_graph = build_module.build(
                    mods, target=target, target_host=target_host,
                    params=params, input_data=inputs,
                    output_dir=build_dir,
                    optimize_method_config=optimize_path, chip="BB04P1")
            return npu_graph

    def get_optimzer_conf_per_partition(self, ops_optimize_param, partition, manual_opt_conf, single_op):
        keys = list(ops_optimize_param.keys())
        curr_optimize_confs = []
        for part in partition:
            if manual_opt_conf is not None:
                optimize_conf = copy.deepcopy(manual_opt_conf)
            else:
                optimize_conf = optiConfig_pb2.OptimizeConfig()
            offset = part[0]
            step_size = part[1]
            for curr_id in range(offset, offset + step_size):
                op_param = ops_optimize_param[keys[curr_id]]
                branch_op = self.get_branch_op(keys[curr_id])
                logging.info("op: {0}, branch_op: {1}".format(keys[curr_id], branch_op))
                input_opti_shift = op_param[0][0]
                input_opti_num = op_param[0][0]
                if branch_op is not None:
                    if branch_op == '':
                        input_opti_shift = 0
                        input_opti_num = 0
                    else:
                        if branch_op not in ops_optimize_param:
                            input_opti_shift = 0
                            input_opti_num = 0
                        else:
                            branch_op_param = ops_optimize_param[branch_op]
                            if branch_op_param[0][0] != op_param[0][0]:
                                input_opti_shift = 0
                                input_opti_num = 0

                if any(op_param[0]) or (len(op_param) == 2 and any(op_param[1])):
                    layer_opti = None
                    for iter_layer_opti in optimize_conf.layerOptiConfig:
                        if iter_layer_opti.name == keys[curr_id]:
                            layer_opti = iter_layer_opti
                    if not layer_opti:
                        layer_opti = optimize_conf.layerOptiConfig.add()
                        layer_opti.name = keys[curr_id]
                    if len(layer_opti.inputOpti) > 0:
                        input_opti = layer_opti.inputOpti[0]
                    else:
                        input_opti = layer_opti.inputOpti.add()
                    input_opti.shift = input_opti_shift
                    input_opti.num = input_opti_num
                    for i in range(len(op_param)):
                        array_param = op_param[i]
                        if len(layer_opti.weightOpti) > i:
                            weight_opti = layer_opti.weightOpti[i]
                        else:
                            weight_opti = layer_opti.weightOpti.add()
                        weight_opti.shift = array_param[1]
                        weight_opti.num = array_param[1]
                        if len(layer_opti.doubleWeightOpti) > i:
                            dw_opti = layer_opti.doubleWeightOpti[i]
                        else:
                            dw_opti = layer_opti.doubleWeightOpti.add()
                        dw_opti.multiple = array_param[2]
                        if len(layer_opti.arrayOutputOpti) > i:
                            output_opti = layer_opti.arrayOutputOpti[i]
                        else:
                            output_opti = layer_opti.arrayOutputOpti.add()
                        output_opti.magnify = array_param[3]
            with tempfile.TemporaryDirectory() as build_dir:
                npu_graph = self.npu_graph_build(self.mods, self.params, self.inputs, optimize_conf, build_dir)
                if not npu_graph:
                    logging.info("Failure allocate partion: {0}".format(partition))
                    if not single_op:
                        return False
                    else:
                        continue
            curr_optimize_confs.append(optimize_conf)
        for optimize_conf in curr_optimize_confs:
            self.optimize_confs.append(optimize_conf)
        return True

    def get_optimzer_conf_per_group(self, ops_optimize_param, manual_opt_conf):
        keys = list(ops_optimize_param.keys())
        for times in range(1, len(keys)+1):
            partition = []
            offset = 0
            for i in range(times - 1):
                partition.append([offset, int(len(keys)/times)])
                offset += int(len(keys)/times)
            partition.append([offset, (len(keys) % times) + int(len(keys)/times)])
            if self.get_optimzer_conf_per_partition(ops_optimize_param, partition, manual_opt_conf, times == len(keys)):
                logging.info("Success partition: {0}".format(partition))
                return True
        return False

    def gen_optimize_confs(self, manual_opt_conf):
        for ops_optimize_param in self.ops_optimize_params:
            if not self.get_optimzer_conf_per_group(ops_optimize_param, manual_opt_conf):
                logging.error("Failed to test param: {0}".format(ops_optimize_param))
                return False
        return True

    def get_optimize_configs(self):
        return self.optimize_confs

    def get_max_op_policy_num(self):
        max_num = 0
        for key, op_array_options in self.ops_array_options.items():
            for array_options in op_array_options:
                max_num = max(max_num, len(array_options))
        return max_num

    def policy_iter(self, curr_id):
        ops_optimize_param = {}
        for key, op_array_options in self.ops_array_options.items():
            op_iter_option = []
            for array_options in op_array_options:
                array_option = array_options[0]
                if curr_id < len(array_options):
                    array_option = array_options[curr_id]
                op_iter_option.append(array_option)
            ops_optimize_param[key] = op_iter_option
        self.ops_optimize_params.append(ops_optimize_param)

    def get_optimize_params(self):
        max_num = self.get_max_op_policy_num()
        for curr_id in range(max_num):
            self.policy_iter(curr_id)

    def graph_analysys(self, graph, model):
        for node in graph.node:
            if node.op_type not in self.array_id_dict.keys():
                continue
            if not node.name:
                continue
            input_nb = model.get_num_array_inputs(node.name)
            logging.info("array_input_number----------{0}".format(input_nb))
            output_nb = model.get_num_array_outputs(node.name)
            logging.info("array_output_number---------{0}".format(output_nb))
            assert input_nb == output_nb
            array_id = self.array_id_dict[node.op_type][0]
            op_array_params = []
            for index in range(len(array_id)):
                index = index * input_nb - index
                weight = self.get_op_initializer(model, node.name, index)
                weight_row = weight.shape[0]
                weight_col = weight.shape[1]
                weight_max = self.get_tensor_abs_max(list(weight.flatten()))
                assert weight_max != 0
                bias = self.get_op_initializer(model, node.name, index + len(array_id))
                bias_max = self.get_tensor_abs_max(list(bias.flatten()))
                # bias_max is not allowed to be zero
                if bias_max == 0:
                    bias_max = 1
                g = self.get_op_scale(model, node.name, index)
                inputs = self.get_array_inputs(model, node.name, index)
                inputs_max = self.get_tensor_abs_max(inputs)
                assert inputs_max != 0
                outputs = self.get_array_outputs(model, node.name, index)
                outputs_max = self.get_tensor_abs_max(outputs)
                assert outputs_max != 0
                outputs = self.get_array_outputs(model, node.name, index)
                op_array_params.append(
                    [weight_max, bias_max, g, inputs_max, outputs_max,
                     weight_row, weight_col, node.op_type])
            self.ops_array_params[node.name] = op_array_params

        with open("./output/auto_optimization_layer_parameter.txt", "w") as f:
            for key, op_array_params in self.ops_array_params.items():
                f.write("Layer Name: {0}\n".format(key))
                for param in op_array_params:
                    f.write("Weight Max: {0} Bias Max: {1} G: {2} Input Max: {3} Output Max: {4} Weight Row: {5} Weight Col: {6} Op: {7}\n".format(
                        param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7]))

    def param_analysys(self, array_param):
        options = []
        weight_max = array_param[0]
        if not weight_max:
            logging.error("Weight max abs value should not be zero, otherwise there is no way to do weight magnify optimization")
            return None
        bias_max = array_param[1]
        g = array_param[2]
        input_max = array_param[3]
        if not input_max:
            logging.error("Input max abs value should not be zero, otherwise there is no way to do input magnify optimization")
            return None
        output_max = array_param[4]
        if not output_max:
            logging.error("Output max abs value should not be zero, otherwise there is no way to do output magnify optimization")
            return None
        weight_row = array_param[5]
        op_type = array_param[7]
        g_up_range = self.g_up_limit/g
        g_down_range = self.g_down_limit/g
        weight_range = self.weight_limit/weight_max
        if bias_max:
            bias_range = self.bias_limit/bias_max
        else:
            bias_range = weight_range
        input_range = int(math.log(self.input_limit/input_max, 2))
        #if op_type == self.op_type_lstm or op_type == self.op_type_gru:
        #    input_range = 0
        output_range = 0
        # Simulator doesn't support output enlarge now
        #if op_type == self.op_type_tdnn or op_type == self.op_type_gemm or op_type == self.op_type_conv:
        if op_type == self.op_type_tdnn or op_type == self.op_type_gemm:
            output_range = int(math.log(self.output_limit/output_max, 2))
        row_multi_range = 0
        # Conv doesn't support double weight optimize method now
        #if op_type == self.op_type_tdnn or op_type == self.op_type_gemm or op_type == self.op_type_conv:
        if op_type == self.op_type_tdnn or op_type == self.op_type_gemm:
            row_multi_range = int(math.log(g_up_range, 2))
        for multi_in in range(input_range + 1):
            for multi_weight in range(int(math.log(weight_range, 2)) + 1):
                for row_multi in range(row_multi_range + 1):
                    for multi_out in range(output_range + 1):
                        g_product = math.pow(2, multi_in + multi_weight + row_multi - multi_out)
                        bias_product = math.pow(2, multi_in + multi_weight + row_multi)
                        if g_product > g_up_range or g_product < g_down_range or bias_product > bias_range:
                            continue
                        if weight_row * math.pow(2, row_multi) > self.array_max_row:
                            continue
                        options.append([multi_in, multi_weight,
                                        row_multi, multi_out])
        return options

    def op_analysys(self):
        for key, op_array_params in self.ops_array_params.items():
            op_array_options = []
            for array_param in op_array_params:
                op_array_option = self.param_analysys(array_param)
                if not op_array_option:
                    logging.error("Please make sure that max abs value of input/output/weight is not zero")
                    return None
                op_array_options.append(op_array_option)
            self.ops_array_options[key] = op_array_options

    def output_user_ops(self, graph, op_output):
        user_ops = []
        for node in graph.node:
            if op_output in node.input:
                user_ops.append(node)
        return user_ops

    def get_array_inputs(self, model, op_name, index):
        inputs = model.get_op_array_inputs(
            op_name, index).asnumpy().flatten()
        return inputs

    def get_array_outputs(self, model, op_name, index):
        outputs = model.get_op_array_outputs(
            op_name, index).asnumpy().flatten()
        return outputs

    def get_op_initializer(self, model, op_name, index):
        initializer = model.get_op_initializer(
            op_name, index).asnumpy()
        return initializer

    def get_op_scale(self, model, op_name, index):
        scale = model.get_op_scale(op_name, index)
        return scale

    def get_array_output_dim(self, model, op_name, index):
        outputs = model.get_op_array_outputs(
            op_name, index).asnumpy()
        return outputs.shape

    def get_array_scale(self, graph, op_name, index):
        node = self.nodes[op_name]
        g_dict = {"scale_params": 0,
                  "scale_ct": 1,
                  "scale_ioft": 0,
                  "scale_zr": 0,
                  "scale_ht": 1}
        for attr in node.attribute:
            if attr.name in g_dict.keys() and g_dict[attr.name] == index:
                return attr.i

        for op_output in node.output:
            users = self.output_user_ops(graph, op_output)
            for user in users:
                if user.op_type == "Scale":
                    return int(1/user.attribute[0].f)

    def get_array_bias(self, op_name, index):
        node = self.nodes[op_name]
        for attr in node.attribute:
            if attr.name == "bias_params":
                return attr.t

        bias_indexs = self.array_id_dict[node.op_type][1]
        op_input = node.input[bias_indexs[index]]
        if op_input in self.init_tensors.keys():
            return self.init_tensors[op_input]

    def get_array_weight(self, op_name, index):
        node = self.nodes[op_name]
        weight_id = self.array_id_dict[node.op_type][0]
        op_input = node.input[weight_id[index]]
        if op_input in self.init_tensors.keys():
           return self.init_tensors[op_input]

    def get_tensor_abs_max(self, ndarray):
        abs_values = [abs(elem) for elem in ndarray]
        return max(abs_values)
