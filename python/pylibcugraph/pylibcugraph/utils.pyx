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
    cugraph_type_erased_device_array_view_pointer,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_copy,
    cugraph_type_erased_device_array_view_free,
)

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_message,
    cugraph_error_free
)

# FIXME: add tests for this
cdef assert_success(cugraph_error_code_t code,
                    cugraph_error_t* err,
                    api_name):
    if code != cugraph_error_code_t.CUGRAPH_SUCCESS:
        c_error = cugraph_error_message(err)
        if isinstance(c_error, bytes):
            c_error = c_error.decode()
        else:
            c_error = str(c_error)

        cugraph_error_free(err)

        if code == cugraph_error_code_t.CUGRAPH_UNKNOWN_ERROR:
            code_str = "CUGRAPH_UNKNOWN_ERROR"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise RuntimeError(error_msg)
        elif code == cugraph_error_code_t.CUGRAPH_INVALID_HANDLE:
            code_str = "CUGRAPH_INVALID_HANDLE"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise ValueError(error_msg)
        elif code == cugraph_error_code_t.CUGRAPH_ALLOC_ERROR:
            code_str = "CUGRAPH_ALLOC_ERROR"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise MemoryError(error_msg)
        elif code == cugraph_error_code_t.CUGRAPH_INVALID_INPUT:
            code_str = "CUGRAPH_INVALID_INPUT"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise ValueError(error_msg)
        elif code == cugraph_error_code_t.CUGRAPH_NOT_IMPLEMENTED:
            code_str = "CUGRAPH_NOT_IMPLEMENTED"
            error_msg = f"non-success value returned from {api_name}: {code_str}\ "\
                        f"{c_error}"
            raise NotImplementedError(error_msg)
        elif code == cugraph_error_code_t.CUGRAPH_UNSUPPORTED_TYPE_COMBINATION:
            code_str = "CUGRAPH_UNSUPPORTED_TYPE_COMBINATION"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise ValueError(error_msg)
        else:
            code_str = "unknown error code"
            error_msg = f"non-success value returned from {api_name}: {code_str} "\
                        f"{c_error}"
            raise RuntimeError(error_msg)


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
    elif c_type == data_type_id_t.SIZE_T:
        return numpy.int64
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

cdef get_c_weight_type_from_numpy_edge_ids_type(numpy_type):
    if numpy_type == numpy.int32:
        return data_type_id_t.FLOAT32
    else:
        return data_type_id_t.FLOAT64

cdef get_numpy_edge_ids_type_from_c_weight_type(data_type_id_t c_weight_type):
    if c_weight_type == data_type_id_t.FLOAT32:
        return numpy.int32
    else:
        return numpy.int64


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

cdef copy_to_cupy_array_ids(
   cugraph_resource_handle_t* c_resource_handle_ptr,
   cugraph_type_erased_device_array_view_t* device_array_view_ptr):
    """
    Copy the contents from a device array view as returned by various cugraph_*
    APIs to a new cupy device array, typically intended to be used as a return
    value from pylibcugraph APIs then convert float to int
    """
    cdef c_type = cugraph_type_erased_device_array_view_type(
        device_array_view_ptr)

    array_size = cugraph_type_erased_device_array_view_size(
        device_array_view_ptr)

    cupy_array = cupy.zeros(
        array_size, dtype=get_numpy_edge_ids_type_from_c_weight_type(c_type))

    cdef uintptr_t cupy_array_ptr = \
        cupy_array.__cuda_array_interface__["data"][0]

    cdef cugraph_type_erased_device_array_view_t* cupy_array_view_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cupy_array_ptr, array_size, get_c_type_from_numpy_type(cupy_array.dtype))

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

cdef cugraph_type_erased_device_array_view_t* \
    create_cugraph_type_erased_device_array_view_from_py_obj(python_obj):
        cdef uintptr_t cai_ptr = <uintptr_t>NULL
        cdef cugraph_type_erased_device_array_view_t* view_ptr = NULL
        if python_obj is not None:
            cai_ptr = python_obj.__cuda_array_interface__["data"][0]
            view_ptr = cugraph_type_erased_device_array_view_create(
                <void*>cai_ptr,
                len(python_obj),
                get_c_type_from_numpy_type(python_obj.dtype))

        return view_ptr

cdef create_cupy_array_view_for_device_ptr(
    cugraph_type_erased_device_array_view_t* device_array_view_ptr,
    owning_py_object):

    if device_array_view_ptr == NULL:
        raise ValueError("device_array_view_ptr cannot be NULL")

    cdef c_type = cugraph_type_erased_device_array_view_type(
        device_array_view_ptr)
    array_size = cugraph_type_erased_device_array_view_size(
        device_array_view_ptr)
    dtype = get_numpy_type_from_c_type(c_type)

    cdef uintptr_t ptr_value = \
        <uintptr_t> cugraph_type_erased_device_array_view_pointer(device_array_view_ptr)

    if ptr_value == <uintptr_t> NULL:
        # For the case of a NULL ptr, just create a new empty ndarray of the
        # appropriate type. This will not be associated with the
        # owning_py_object, but will still be garbage collected correctly.
        cupy_array = cupy.ndarray(0, dtype=dtype)

    else:
        # cupy.cuda.UnownedMemory takes a reference to an owning python object
        # which is used to increment the refcount on the owning python object.
        # This prevents the owning python object from being garbage collected
        # and having the memory freed when there are instances of the
        # cupy_array still in use that need the memory.  When the cupy_array
        # instance returned here is deleted, it will decrement the refcount on
        # the owning python object, and when that refcount reaches zero the
        # owning python object will be garbage collected and the memory freed.
        cpmem = cupy.cuda.UnownedMemory(ptr_value,
                                        array_size,
                                        owning_py_object)
        cpmem_ptr = cupy.cuda.MemoryPointer(cpmem, 0)
        cupy_array = cupy.ndarray(
            array_size,
            dtype=dtype,
            memptr=cpmem_ptr)

    return cupy_array
