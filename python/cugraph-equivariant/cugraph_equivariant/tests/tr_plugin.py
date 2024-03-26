
import onnx
import cupy as cp
import numpy as np
import tensorrt as trt

from polygraphy.json import to_json, from_json
import torch

class SegmentedTransposePlugin(trt.IPluginV2DynamicExt):
    def __init__(self, fc=None):
        trt.IPluginV2DynamicExt.__init__(self)
        
        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_type = "segmented_transpose"
        self.plugin_version = "1"

        fc_dict = {}        
        if fc is not None:
            for f in fc:
                fc_dict[f.name]=f.data
            self.flag = fc_dict["flag"]

            
    def get_output_datatype(self, index, input_types):
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder):
        output_dims = trt.DimsExprs(inputs[0])
        return output_dims

    def serialize(self):
        return to_json(self.__dict__)

    def configure_plugin(self, inp, out):
        pass

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 2
        assert pos < len(in_out)

        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be (b)float16 or float32
        if pos == 0:
            return desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF or desc.type == trt.DataType.BF16
        elif pos == 1:
            return desc.type == trt.DataType.INT32 or desc.type == trt.DataType.INT64
        else:
        # should have the same type as the input[0]
            return in_out[0].type == desc.type

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        a_mem = [ cp.cuda.UnownedMemory(inputs[i], np.prod(input_desc[i].dims) * cp.dtype(trt.nptype(input_desc[i].type)).itemsize, self) for i in range(len(inputs)) ]

        c_mem = cp.cuda.UnownedMemory(
            outputs[0],
            np.prod(output_desc[0].dims) * cp.dtype(trt.nptype(output_desc[0].type)).itemsize,
            self,
        )

        a_ptr = [ cp.cuda.MemoryPointer(a, 0) for a in a_mem ] 
        c_ptr = cp.cuda.MemoryPointer(c_mem, 0)

        a_d = [ cp.ndarray(tuple(input_desc[i].dims), dtype=trt.nptype(input_desc[i].type), memptr=a_ptr[i]) for i in range(len(inputs)) ]
        
        c_d = cp.ndarray((np.prod(output_desc[0].dims)), dtype=trt.nptype(output_desc[0].type), memptr=c_ptr)

        a_t = [ torch.as_tensor(d, device='cuda') for d in a_d ]

        out = torch.ops.cgtp.segmented_transpose(a_t[0], a_t[1], self.flag)

        cp.copyto(c_d, cp.reshape(cp.asarray(out), (-1,)))

        return 0

    def clone(self):
        cloned_plugin = SegmentedTransposePlugin()
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

class SegmentedTransposePluginCreator(trt.IPluginCreator):
    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = "segmented_transpose"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [trt.PluginField("flag"),
             ]
        )

    def create_plugin(self, name, fc):
        pl = SegmentedTransposePlugin(fc)
        return pl

    def deserialize_plugin(self, name, data):
        j = dict(from_json(data.decode("utf-8")))
        deserialized = SegmentedTransposePlugin()
        deserialized.__dict__.update(j)
        return deserialized
