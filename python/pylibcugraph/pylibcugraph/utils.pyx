# Copyright (c) 2022, NVIDIA CORPORATION.
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

# Have cython use python 3 syntax
# cython: language_level = 3

import numpy

# FIXME: add tests for this
cdef assert_success(cugraph_error_code_t code,
                    cugraph_error_t* err,
                    api_name):
    if code != cugraph_error_code_t.CUGRAPH_SUCCESS:
        if code == cugraph_error_code_t.CUGRAPH_UNKNOWN_ERROR:
            code_str = "CUGRAPH_UNKNOWN_ERROR"
        elif code == cugraph_error_code_t.CUGRAPH_INVALID_HANDLE:
            code_str = "CUGRAPH_INVALID_HANDLE"
        elif code == cugraph_error_code_t.CUGRAPH_ALLOC_ERROR:
            code_str = "CUGRAPH_ALLOC_ERROR"
        elif code == cugraph_error_code_t.CUGRAPH_INVALID_INPUT:
            code_str = "CUGRAPH_INVALID_INPUT"
        elif code == cugraph_error_code_t.CUGRAPH_NOT_IMPLEMENTED:
            code_str = "CUGRAPH_NOT_IMPLEMENTED"
        elif code == cugraph_error_code_t.CUGRAPH_UNSUPPORTED_TYPE_COMBINATION:
            code_str = "CUGRAPH_UNSUPPORTED_TYPE_COMBINATION"
        else:
            code_str = "unknown error code"
        # FIXME: extract message using cugraph_error_message()
        # FIXME: If error_ptr has a value, free it using cugraph_error_free()
        raise RuntimeError(f"non-success value returned from {api_name}: {code_str}")


cdef assert_CAI_type(obj, var_name, allow_None=False):
    if allow_None:
        if obj is None:
            return
        msg = f"{var_name} must be None or support __cuda_array_interface__"
    else:
        msg = f"{var_name} does not support __cuda_array_interface__"

    if not(hasattr(obj, "__cuda_array_interface__")):
        raise TypeError(msg)


cdef get_numpy_type_from_c_type(data_type_id_t c_type):
    if c_type == data_type_id_t.INT32:
        return numpy.int32
    elif c_type == data_type_id_t.INT64:
        return numpy.int64
    elif c_type == data_type_id_t.FLOAT32:
        return numpy.float32
    elif c_type == data_type_id_t.FLOAT64:
        return numpy.float64
    else:
        raise RuntimeError("Internal error: got invalid data type enum value "
                           f"from C: {c_type}")
