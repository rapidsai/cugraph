# Copyright (c) 2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from cugraph_equivariant.nn import FullyConnectedTensorProductConv
# import torch._C._onnx as _C_onnx
import threading
lock_sm = threading.Lock()

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)
from tp_plugin import FusedTensorProductPluginCreator
from tr_plugin import SegmentedTransposePluginCreator

# from polygraphy.logger import G_LOGGER
# G_LOGGER.module_severity = {'': G_LOGGER.EXTRA_VERBOSE}

input_names=["src_features", "edge_sh", "edge_emb", "edge_index", "num_dst_nodes", "src_scalars", "dst_scalars"]
output_names=["out"]

def test_onnx_export(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs
    my_input_names = input_names
    
    if src_scalars is None and dst_scalars is None:
        my_input_names = my_input_names[:-2]
    
    with torch.no_grad():
        expected = tp_conv(*inputs)
        
        torch.onnx.export(tp_conv,
                          inputs,
                          "a.onnx",
                          input_names=my_input_names,
                          output_names=output_names,
                          # operator_export_type=_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          # verbose=True
                          )

@pytest.mark.skip(reason="Only works with any single conftest option, crashes with any two. Reduce conftest to a single tp_conv to run this. TODO: fix that")
def test_trt(
        create_tp_conv_and_data,
):
    (tp_conv, inputs, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs

    with lock_sm:
        # Register plugin creator
        plg_registry = trt.get_plugin_registry()
        plg_registry.register_creator(FusedTensorProductPluginCreator(), "")
        plg_registry.register_creator(SegmentedTransposePluginCreator(), "")
        
        my_input_names = input_names
        
        if src_scalars is None and dst_scalars is None:
            my_input_names = my_input_names[:-2]
        torch.onnx.export(tp_conv,
                          inputs,
                          "a.onnx",
                          input_names=my_input_names,
                          output_names=output_names,
                          # operator_export_type=_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          # verbose=True
                          )

        
        # build engine
        build_engine = EngineFromNetwork(
            NetworkFromOnnxPath("a.onnx"), CreateConfig()
        )

        # Run
        with torch.no_grad(), TrtRunner(build_engine, "trt_runner") as runner:
            expected = tp_conv(*inputs)
            trt_inputs = {}
            my_input_names = input_names
            
            for inp, name in zip(inputs, input_names):
                if inp is not None:
                    trt_inputs[name]=inp
            # TODO: dynamic axes
            trt_inputs.pop("num_dst_nodes")
            
            outputs = runner.infer(trt_inputs)
            
            t_res = torch.as_tensor(outputs["out"], device="cuda")
            
            torch.testing.assert_close(t_res, expected)

        
def test_torch_compile(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs

    with torch.no_grad():
        expected = tp_conv(*inputs)
        tp_conv = torch.compile(tp_conv)
        compiled = tp_conv(*inputs)
        torch.testing.assert_close(compiled, expected)

@pytest.mark.skip(reason="Need registry/onnx extension to test this")
def test_dynamo_compile(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs

    with torch.no_grad():
        expected = tp_conv(*inputs)
        tp_ex = torch.onnx.dynamo_export(tp_conv, *inputs)
        
@pytest.mark.skip(reason="RuntimeError: _Map_base::at")
def test_jit(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs
    inputs = tuple([i for i in inputs if i is not None])
    with torch.no_grad():
        expected = tp_conv(*inputs)
        tp_conv = torch.jit.freeze(torch.jit.trace(tp_conv, inputs))
        torch.jit.save("test.ts")
        restored_conv = torch.jit.load("test.ts")
        compiled = restored_conv(*inputs)
        torch.testing.assert_close(compiled, expected)
