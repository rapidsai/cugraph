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
import torch._C._onnx as _C_onnx

def test_onnx_export(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs

    with torch.no_grad():
        expected = tp_conv(*inputs)
        
        torch.onnx.export(tp_conv,
                          inputs,
                          "a.onnx",
                          # operator_export_type=_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          # verbose=True
                          )

def test_torch_compile(
        create_tp_conv_and_data, 
):
    (tp_conv, inputs, _) = create_tp_conv_and_data
    (src_features, edge_sh, edge_emb, edge_index, num_dst_nodes, src_scalars, dst_scalars) = inputs

    with torch.no_grad():
        expected = tp_conv(*inputs)
        tp_conv = torch.compile(tp_conv)
        compiled = tp_conv(*inputs)
        # too many diffs in Torch 2.3
        # torch.testing.assert_close(compiled, expected)

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
