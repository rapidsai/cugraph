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

from libc.stdint cimport uintptr_t

import numpy
import cupy

from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_size,
    cugraph_type_erased_device_array_view_type,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_copy,
    cugraph_type_erased_device_array_view_free,
)

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


cdef assert_AI_type(obj, var_name, allow_None=False):
    if allow_None:
        if obj is None:
            return
        msg = f"{var_name} must be None or support __array_interface__"
    else:
        msg = f"{var_name} does not support __array_interface__"

    if not(hasattr(obj, "__array_interface__")):
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


cdef get_c_type_from_numpy_type(numpy_type):
    if numpy_type == numpy.int32:
        return data_type_id_t.INT32
    elif numpy_type == numpy.int64:
        return data_type_id_t.INT64
    elif numpy_type == numpy.float32:
        return data_type_id_t.FLOAT32
    elif numpy_type == numpy.float64:
        return data_type_id_t.FLOAT64
    else:
        raise RuntimeError("Internal error: got invalid data type enum value "
                          f"from Numpy: {numpy_type}")


cdef copy_to_cupy_array(
   cugraph_resource_handle_t* c_resource_handle_ptr,
   cugraph_type_erased_device_array_view_t* device_array_view_ptr):
    """
    Copy the contents from a device array view as returned by various cugraph_*
    APIs to a new cupy device array, typically intended to be used as a return
    value from pylibcugraph APIs.
    """
    cdef c_type = cugraph_type_erased_device_array_view_type(
        device_array_view_ptr)
    array_size = cugraph_type_erased_device_array_view_size(
        device_array_view_ptr)

    cupy_array = cupy.zeros(
        array_size, dtype=get_numpy_type_from_c_type(c_type))

    cdef uintptr_t cupy_array_ptr = \
        cupy_array.__cuda_array_interface__["data"][0]

    cdef cugraph_type_erased_device_array_view_t* cupy_array_view_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cupy_array_ptr, array_size, c_type)

    cdef cugraph_error_t* error_ptr
    error_code = cugraph_type_erased_device_array_view_copy(
        c_resource_handle_ptr,
        cupy_array_view_ptr,
        device_array_view_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr,
                   "cugraph_type_erased_device_array_view_copy")

    cugraph_type_erased_device_array_view_free(device_array_view_ptr)

    return cupy_array
