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
from conftest import compare_results

from cugraph_equivariant.nn import FullyConnectedTensorProductConv
# import torch._C._onnx as _C_onnx

import tensorrt as trt
import onnxruntime as ort
import onnxruntime_extensions

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

from polygraphy.backend.onnx import onnx_from_path, fold_constants, save_onnx
from polygraphy.backend.onnxrt import OnnxrtRunner, session_from_onnx

from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)

import pylibcugraphops.pytorch.onnx as cg_onnx

# from polygraphy.logger import G_LOGGER
# G_LOGGER.module_severity = {'': G_LOGGER.EXTRA_VERBOSE}

input_names=["src_features", "edge_sh", "edge_emb", "edge_index", "num_dst_nodes", "src_scalars", "dst_scalars"]
output_names=["out"]

def run_onnx(ex_name, inputs):
    from onnxruntime import InferenceSession
    onnx_runner = OnnxrtRunner(
        InferenceSession(ex_name, providers=["CUDAExecutionProvider"],
                         sess_options=cg_onnx.register_custom_ops_library())
    )
    trt_inputs = {}
    onnx_runner.activate()
    for inp, name in zip(inputs, input_names):
        if inp is not None:
            trt_inputs[name]=inp
    # TODO: dynamic axes
    trt_inputs.pop("num_dst_nodes")

    ret = onnx_runner.infer(trt_inputs)
    return torch.as_tensor(ret["out"], device="cuda")        


def test_onnx_export(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _, param) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs
    (dtype, batch_norm, e3nn_compat_mode, _) = param
    
    my_input_names = input_names
    
    if src_scalars is None and dst_scalars is None:
        my_input_names = my_input_names[:-2]
    ex_name = "a.onnx"
    with torch.no_grad():
        expected = tp_conv(*inputs)
        
        torch.onnx.export(tp_conv,
                          inputs,
                          ex_name,
                          input_names=my_input_names,
                          output_names=output_names,
                          # operator_export_type=_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          # verbose=True
                          )
        # onnxruntime only supports bfloat16 with opset>=20
        if dtype == torch.bfloat16:
            return
        res = run_onnx(ex_name, inputs)
        compare_results(res, expected)
         
def test_trt(
        create_tp_conv_and_data,
):
    (tp_conv, inputs, _, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs
    cg_onnx.register_trt_plugins()

    with torch.no_grad():
        my_input_names = input_names
    
        if src_scalars is None and dst_scalars is None:
            my_input_names = my_input_names[:-2]
        onnx_path = "a.onnx"
        torch.onnx.export(tp_conv,
                          inputs,
                          onnx_path,
                          input_names=my_input_names,
                          output_names=output_names,
                          # operator_export_type=_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          # verbose=True
                          )

        model_onnx = onnx_from_path(onnx_path)
        fold_constants(model_onnx)
        save_onnx(model_onnx, onnx_path)
        
        # build engine
        build_engine = EngineFromNetwork(
            NetworkFromOnnxPath("a.onnx"), CreateConfig()
        )

        # Run
        with TrtRunner(build_engine, "trt_runner") as runner:
            expected = tp_conv(*inputs)
            trt_inputs = {}
            for inp, name in zip(inputs, input_names):
                if inp is not None:
                    trt_inputs[name]=inp
            # TODO: dynamic axes
            trt_inputs.pop("num_dst_nodes")
            
            outputs = runner.infer(trt_inputs)
            
            t_res = torch.as_tensor(outputs["out"], device="cuda")
            
            compare_results(t_res, expected)

        
def test_torch_compile(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs

    with torch.no_grad():
        expected = tp_conv(*inputs)
        tp_conv = torch.compile(tp_conv)
        compiled = tp_conv(*inputs)
        # With Torch 2.2, compile() results differ with float32 as much as they do with bfloat16
        compare_results(compiled, expected, atol=6e-1, rtol=6e-1)


def test_dynamo_export(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _, param) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs
    (dtype, batch_norm, e3nn_compat_mode, _) = param
    
    if batch_norm:
        return;
    options = torch.onnx.ExportOptions(dynamic_shapes=True,
                                       onnx_registry=cg_onnx.register_custom_ops())

    ex_name="a.onnx"
    with torch.no_grad():
        expected = tp_conv(*inputs)

        tp_ex = torch.onnx.dynamo_export(tp_conv,
                                         *inputs,
                                         export_options=options
                                         )
        tp_ex.save(ex_name)
        # @pytest.mark.skip(reason="Unsupported model IR version: 10, max supported IR version: 9")
        return
        res = run_onnx(ex_name, inputs)
        compare_results(res, expected)
        
def test_jit(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs
    inputs = tuple([i for i in inputs if i is not None])
    with torch.no_grad():
        expected = tp_conv(*inputs)
        tp_conv = torch.jit.trace(tp_conv, inputs)
        torch.jit.save(tp_conv, "test.ts")
        restored_conv = torch.jit.load("test.ts")
        compiled = restored_conv(*inputs)
        compare_results(compiled, expected)

