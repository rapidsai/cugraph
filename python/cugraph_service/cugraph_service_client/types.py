# Copyright (c) 2022, NVIDIA CORPORATION.
#
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

import numpy

from gaas_client.gaas_thrift import spec

Value = spec.Value
GraphVertexEdgeID = spec.GraphVertexEdgeID
BatchedEgoGraphsResult = spec.BatchedEgoGraphsResult
Node2vecResult = spec.Node2vecResult
UniformNeighborSampleResult = spec.UniformNeighborSampleResult


class UnionWrapper:
    """
    Provides easy conversions between py objs and Thrift "unions".
    """
    def get_py_obj(self):
        not_members = set(["default_spec", "thrift_spec", "read", "write"])
        attrs = [a for a in dir(self.union)
                    if not(a.startswith("_")) and a not in not_members]
        for a in attrs:
            val = getattr(self.union, a)
            if val is not None:
                return val

        return None


class ValueWrapper(UnionWrapper):
    def __init__(self, val, val_name="value"):
        if isinstance(val, Value):
            self.union = val
        elif isinstance(val, int):
            if val < 4294967296:
                self.union = Value(int32_value=val)
            else:
                self.union = Value(int64_value=val)
        elif isinstance(val, numpy.int32):
            self.union = Value(int32_value=int(val))
        elif isinstance(val, numpy.int64):
            self.union = Value(int64_value=int(val))
        elif isinstance(val, str):
            self.union = Value(string_value=val)
        elif isinstance(val, bool):
            self.union = Value(bool_value=val)
        else:
            raise TypeError(f"{val_name} must be one of the "
                            "following types: [int, str, bool], got "
                            f"{type(val)}")


class GraphVertexEdgeIDWrapper(UnionWrapper):
    def __init__(self, val, val_name="id"):
        if isinstance(val, GraphVertexEdgeID):
            self.union = val
        elif isinstance(val, int):
            if val >= 4294967296:
                self.union = GraphVertexEdgeID(int64_id=val)
            else:
                self.union = GraphVertexEdgeID(int32_id=val)
        elif isinstance(val, list):
            # FIXME: this only check the first item, others could be larger
            if val[0] >= 4294967296:
                self.union = GraphVertexEdgeID(int64_ids=val)
            else:
                self.union = GraphVertexEdgeID(int32_ids=val)
        else:
            raise TypeError(f"{val_name} must be one of the "
                            "following types: [int, list<int>], got "
                            f"{type(val)}")
