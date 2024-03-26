#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx
import cupy as cp
import numpy as np
import tensorrt as trt

from polygraphy.json import to_json, from_json
import torch

class FusedTensorProductPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, fc=None):
        trt.IPluginV2DynamicExt.__init__(self)
        
        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_type = "tensor_product"
        self.plugin_version = "1"

        fc_dict = {}
        
        if fc is not None:
            for f in fc:
                fc_dict[f.name]=f.data
            self.path_csr_offsets = torch.as_tensor(fc_dict["path_csr_offsets"], device = "cuda")
            self.path_cg_values = torch.as_tensor(fc_dict["path_cg_values"], device = "cuda")
            self.path_offsets = torch.as_tensor(fc_dict["path_offsets"], device = "cuda")
            if len(self.path_offsets.shape)==1:
                self.path_offsets = self.path_offsets.reshape((self.path_cg_values.shape[0], -1))
            self.connection_mode = fc_dict["connection_mode"][:-1].tobytes().decode()
            # print ("connection_mode :", fc_dict["connection_mode"], self.connection_mode)
            self.stride_out = fc_dict["stride_out"]

            
    def get_output_datatype(self, index, input_types):
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder):
        output_dims = trt.DimsExprs(inputs[0])
        output_dims[len(output_dims) - 1] = exprBuilder.constant(self.stride_out)
        return output_dims

    def serialize(self):
        return to_json(self.__dict__)

    def configure_plugin(self, inp, out):
        pass

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 3 or num_inputs == 4
        assert pos < len(in_out)

        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be (b)float16 or float32
        if pos == 0:
            return desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF or desc.type == trt.DataType.BF16
        else:
        # should have the same type as the input[0]
            return in_out[0].type == desc.type

        assert False

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        inp_dtype = trt.nptype(input_desc[0].type)

        a_mem = [ cp.cuda.UnownedMemory(inputs[i], np.prod(input_desc[i].dims) * cp.dtype(inp_dtype).itemsize, self) for i in range(len(inputs)) ]

        c_mem = cp.cuda.UnownedMemory(
            outputs[0],
            np.prod(output_desc[0].dims) * cp.dtype(inp_dtype).itemsize,
            self,
        )

        a_ptr = [ cp.cuda.MemoryPointer(a, 0) for a in a_mem ] 
        c_ptr = cp.cuda.MemoryPointer(c_mem, 0)

        a_d = [ cp.ndarray(tuple(input_desc[i].dims), dtype=inp_dtype, memptr=a_ptr[i]) for i in range(len(inputs)) ]
        
        c_d = cp.ndarray((np.prod(output_desc[0].dims)), dtype=inp_dtype, memptr=c_ptr)

        a_t = [ torch.as_tensor(d, device='cuda') for d in a_d ]

        out = torch.ops.cgtp.tensor_product(a_t[0], a_t[1], a_t[2],
                                            a_t[3] if len(inputs)==4 else None,
                                            self.path_csr_offsets,
                                            self.path_offsets,
                                            self.path_cg_values,
                                            self.connection_mode,
                                            self.stride_out)
        cp.copyto(c_d, cp.reshape(cp.asarray(out), (-1,)))

        return 0

    def clone(self):
        cloned_plugin = FusedTensorProductPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


    def get_serialization_size(self):
         return len(to_json(self.__dict__))

    # def get_workspace_size(self, input_desc, output_desc):
    #     return 0
    
    # def destroy(self):
    #     pass

    # def terminate(self):
    #     pass

class FusedTensorProductPluginCreator(trt.IPluginCreator):
    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = "tensor_product"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [trt.PluginField("path_csr_offsets"),
             trt.PluginField("path_offsets"),
             trt.PluginField("path_cg_values"),
             trt.PluginField("connection_mode"),
             trt.PluginField("stride_out")
             ]
        )

    def create_plugin(self, name, fc):
        pl = FusedTensorProductPlugin(fc)
        return pl

    def deserialize_plugin(self, name, data):
        j = dict(from_json(data.decode("utf-8")))
        deserialized = FusedTensorProductPlugin()
        deserialized.__dict__.update(j)
        return deserialized

