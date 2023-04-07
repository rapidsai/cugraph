# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

# Optional modules: additional features are enabled if these are present
try:
    import numpy
except ModuleNotFoundError:
    numpy = None
try:
    import cupy
except ModuleNotFoundError:
    cupy = None

from cugraph_service_client.cugraph_service_thrift import spec

Value = spec.Value
GraphVertexEdgeID = spec.GraphVertexEdgeID
BatchedEgoGraphsResult = spec.BatchedEgoGraphsResult
Node2vecResult = spec.Node2vecResult
UniformNeighborSampleResult = spec.UniformNeighborSampleResult
Offsets = spec.Offsets


class UnionWrapper:
    """
    Provides easy conversions between py objs and Thrift "unions". This is used
    as a base class for the "*Wrapper" classes below. Together with the derived
    classes below, these objects allow the caller to go from py objects/Thrift
    unions to Thrift unions/py objects.
    """

    non_attrs = set(["default_spec", "thrift_spec", "read", "write"])


class ValueWrapper(UnionWrapper):
    """
    Provides an easy-to-use python object for abstracting Thrift "unions",
    allowing a python obj to be automatically mapped to the correct union
    field.
    """

    valid_types = ["int", "float", "str", "bool"]
    if numpy:
        valid_types += ["numpy.int8", "numpy.int32", "numpy.int64", "numpy.ndarray"]
    if cupy:
        valid_types += ["cupy.int8", "cupy.int32", "cupy.int64", "cupy.ndarray"]

    def __init__(self, val, val_name="value"):
        """
        Construct with a value supported by the Value "union". See
        cugraph_service_thrift.py

        val_name is used for better error messages only, and can be passed for
        including in the exception thrown if an invalid type is passed here.
        """
        if isinstance(val, Value):
            self.union = val
        elif isinstance(val, int):
            if val < 4294967296:
                self.union = Value(int32_value=val)
            else:
                self.union = Value(int64_value=val)
        elif isinstance(val, float):
            self.union = Value(double_value=val)
        elif (numpy and isinstance(val, (numpy.int8, numpy.int32))) or (
            cupy and isinstance(val, (cupy.int8, cupy.int32))
        ):
            self.union = Value(int32_value=int(val))
        elif (numpy and isinstance(val, numpy.int64)) or (
            cupy and isinstance(val, cupy.int64)
        ):
            self.union = Value(int64_value=int(val))
        elif (
            (numpy and isinstance(val, numpy.float32))
            or (cupy and isinstance(val, cupy.float32))
            or (numpy and isinstance(val, numpy.float64))
            or (cupy and isinstance(val, cupy.float64))
        ):
            self.union = Value(double_value=float(val))
        elif isinstance(val, str):
            self.union = Value(string_value=val)
        elif isinstance(val, bool):
            self.union = Value(bool_value=val)
        elif isinstance(val, (list, tuple)):
            self.union = Value(list_value=[ValueWrapper(i) for i in val])
        # FIXME: Assume ndarrays contain values Thrift can accept! Otherwise,
        # check and possibly convert ndarray dtypes.
        elif (numpy and isinstance(val, numpy.ndarray)) or (
            cupy and isinstance(val, cupy.ndarray)
        ):
            # self.union = Value(list_value=val.tolist())
            self.union = Value(list_value=[ValueWrapper(i) for i in val.tolist()])
        elif val is None:
            self.union = Value()
        else:
            raise TypeError(
                f"{val_name} must be one of the "
                f"following types: {self.valid_types}, got "
                f"{type(val)}"
            )

    def __getattr__(self, attr):
        """
        Retrieve all other attrs from the underlying Value object. This will
        essentially duck-type this ValueWrapper instance and allow it to be
        returned to Thrift and treated as a Value.
        """
        return getattr(self.union, attr)

    def get_py_obj(self):
        """
        Get the python object set in the union.
        """
        attrs = [
            a
            for a in dir(self.union)
            if not (a.startswith("_")) and a not in self.non_attrs
        ]
        # Much like a C union, only one field will be set. Return the first
        # non-None value encountered.
        for a in attrs:
            val = getattr(self.union, a)
            if val is not None:
                # Assume all lists are homogeneous. Check the first item to see
                # if it is a Value or ValueWrapper obj, and if so recurse.
                # FIXME: this might be slow, consider handling lists of numbers
                # differently
                if isinstance(val, list) and len(val) > 0:
                    if isinstance(val[0], Value):
                        return [ValueWrapper(i).get_py_obj() for i in val]
                    elif isinstance(val[0], ValueWrapper):
                        return [i.get_py_obj() for i in val]
                    else:
                        raise TypeError(
                            f"expected Value or ValueWrapper, got {type(val)}"
                        )
                else:
                    return val

        return None


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
            raise TypeError(
                f"{val_name} must be one of the "
                "following types: [int, list<int>], got "
                f"{type(val)}"
            )

    def get_py_obj(self):
        """
        Get the python object set in the union.
        """
        attrs = [
            a
            for a in dir(self.union)
            if not (a.startswith("_")) and a not in self.non_attrs
        ]

        # Much like a C union, only one field will be set. Return the first
        # non-None value encountered.
        for a in attrs:
            val = getattr(self.union, a)
            if val is not None:
                return val

        return None
