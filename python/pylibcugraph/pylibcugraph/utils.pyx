# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
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
    cugraph_type_erased_host_array_view_create,
)

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_message,
    cugraph_error_free
)

from pylibcugraph._cugraph_c.dlpack_interop cimport (
    cugraph_dlpack_is_device_accessible,
    cugraph_dlpack_is_host_accessible,
    cugraph_dlpack_get_array_info,
)
from pylibcugraph._cugraph_c.types cimport (
    bool_t,
    FALSE,
    TRUE,
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


cdef _assert_dlpack(obj, var_name, allow_None=False):
    if obj is None and allow_None:
        return
    values = getattr(obj, "values", None)
    if (not hasattr(obj, "__dlpack__") and
            (values is None or not hasattr(values, "__dlpack__"))):
        optional = "None or " if allow_None else ""
        raise TypeError(f"{var_name} must be {optional}an object supporting DLPack")


def _get_dlpack_capsule(obj):
    if hasattr(obj, "__dlpack__"):
        producer = obj
    else:
        producer = getattr(obj, "values", None)
        if producer is None or not hasattr(producer, "__dlpack__"):
            raise TypeError("object does not expose a standard DLPack producer")

    try:
        return producer.__dlpack__(max_version=(1, 0))
    except TypeError:
        # max_version was added to the Python DLPack protocol after its
        # initial release. Retain compatibility with legacy producers.
        return producer.__dlpack__()


cdef void* _get_dlpack_pointer(object capsule, bool_t* versioned) except NULL:
    if PyCapsule_IsValid(capsule, "dltensor_versioned"):
        versioned[0] = TRUE
        return PyCapsule_GetPointer(capsule, "dltensor_versioned")

    if PyCapsule_IsValid(capsule, "dltensor"):
        versioned[0] = FALSE
        return PyCapsule_GetPointer(capsule, "dltensor")

    raise ValueError(
        "expected an unconsumed 'dltensor' or 'dltensor_versioned' "
        "DLPack capsule"
    )


cdef _get_dlpack_array_info(
    object python_obj,
    void** data,
    size_t* size,
    cugraph_data_type_id_t* dtype,
):
    capsule = _get_dlpack_capsule(python_obj)
    cdef bool_t versioned
    cdef void* managed_tensor = _get_dlpack_pointer(capsule, &versioned)
    cdef cugraph_error_t* error = NULL
    cdef cugraph_error_code_t code = cugraph_dlpack_get_array_info(
        managed_tensor, versioned, data, size, dtype, &error
    )
    assert_success(code, error, "cugraph_dlpack_get_array_info")


cpdef bint is_device_accessible(obj):
    # Pinned or managed memory is recognized only when the DLPack producer
    # reports it as CUDA host or CUDA managed memory, respectively.
    _assert_dlpack(obj, "array")
    capsule = _get_dlpack_capsule(obj)
    cdef bool_t versioned
    cdef void* managed_tensor = _get_dlpack_pointer(capsule, &versioned)
    cdef bool_t result
    cdef cugraph_error_t* error = NULL
    cdef cugraph_error_code_t code = cugraph_dlpack_is_device_accessible(
        managed_tensor, versioned, &result, &error
    )
    assert_success(code, error, "cugraph_dlpack_is_device_accessible")
    return result == TRUE


cpdef bint is_host_accessible(obj):
    # Pinned or managed memory is recognized only when the DLPack producer
    # reports it as CUDA host or CUDA managed memory, respectively.
    _assert_dlpack(obj, "array")
    capsule = _get_dlpack_capsule(obj)
    cdef bool_t versioned
    cdef void* managed_tensor = _get_dlpack_pointer(capsule, &versioned)
    cdef bool_t result
    cdef cugraph_error_t* error = NULL
    cdef cugraph_error_code_t code = cugraph_dlpack_is_host_accessible(
        managed_tensor, versioned, &result, &error
    )
    assert_success(code, error, "cugraph_dlpack_is_host_accessible")
    return result == TRUE


cdef assert_device_accessible(obj, var_name, allow_None=False):
    _assert_dlpack(obj, var_name, allow_None)
    if obj is None:
        return
    if not is_device_accessible(obj):
        raise ValueError(f"{var_name} must be accessible from a CUDA device")


cdef assert_host_accessible(obj, var_name, allow_None=False):
    _assert_dlpack(obj, var_name, allow_None)
    if obj is None:
        return
    if not is_host_accessible(obj):
        raise ValueError(f"{var_name} must be accessible from the host")


cdef get_numpy_type_from_c_type(cugraph_data_type_id_t c_type):
    if c_type == cugraph_data_type_id_t.INT32:
        return numpy.int32
    elif c_type == cugraph_data_type_id_t.INT64:
        return numpy.int64
    elif c_type == cugraph_data_type_id_t.FLOAT32:
        return numpy.float32
    elif c_type == cugraph_data_type_id_t.FLOAT64:
        return numpy.float64
    elif c_type == cugraph_data_type_id_t.SIZE_T:
        return numpy.int64
    elif c_type == cugraph_data_type_id_t.BOOL:
        return bool
    else:
        raise RuntimeError("Internal error: got invalid data type enum value "
                           f"from C: {c_type}")


cpdef cugraph_data_type_id_t get_c_type_from_py_obj(object python_obj) except *:
    _assert_dlpack(python_obj, "array")
    cdef void* data
    cdef size_t size
    cdef cugraph_data_type_id_t dtype
    _get_dlpack_array_info(python_obj, &data, &size, &dtype)
    return dtype


cdef get_dtype_name_from_c_type(cugraph_data_type_id_t c_type):
    return numpy.dtype(get_numpy_type_from_c_type(c_type)).name


cdef size_t get_size_from_py_obj(object python_obj) except *:
    _assert_dlpack(python_obj, "array")
    cdef void* data
    cdef size_t size
    cdef cugraph_data_type_id_t dtype
    _get_dlpack_array_info(python_obj, &data, &size, &dtype)
    return size


cdef get_last_item_from_py_obj(object python_obj):
    assert_device_accessible(python_obj, "array")
    producer = python_obj
    if not hasattr(producer, "__dlpack__"):
        producer = producer.values
    return cupy.from_dlpack(producer)[-1].item()


cdef uintptr_t get_data_ptr_from_py_obj(object python_obj) except *:
    _assert_dlpack(python_obj, "array")
    cdef void* data
    cdef size_t size
    cdef cugraph_data_type_id_t dtype
    _get_dlpack_array_info(python_obj, &data, &size, &dtype)
    return <uintptr_t>data


cdef get_c_type_from_numpy_type(numpy_type):
    dt = numpy.dtype(numpy_type)
    if dt == numpy.int32:
        return cugraph_data_type_id_t.INT32
    elif dt == numpy.int64:
        return cugraph_data_type_id_t.INT64
    elif dt == numpy.float32:
        return cugraph_data_type_id_t.FLOAT32
    elif dt == numpy.float64:
        return cugraph_data_type_id_t.FLOAT64
    else:
        raise RuntimeError("Internal error: got invalid data type enum value "
                          f"from Numpy: {numpy_type}")

cdef get_c_weight_type_from_numpy_edge_ids_type(numpy_type):
    if numpy_type == numpy.int32:
        return cugraph_data_type_id_t.FLOAT32
    else:
        return cugraph_data_type_id_t.FLOAT64

cdef get_numpy_edge_ids_type_from_c_weight_type(cugraph_data_type_id_t c_weight_type):
    if c_weight_type == cugraph_data_type_id_t.FLOAT32:
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
        get_data_ptr_from_py_obj(cupy_array)

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
        get_data_ptr_from_py_obj(cupy_array)

    cdef cugraph_type_erased_device_array_view_t* cupy_array_view_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cupy_array_ptr, array_size, get_c_type_from_py_obj(cupy_array))

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
        cdef cugraph_type_erased_device_array_view_t* view_ptr = NULL
        cdef void* data = NULL
        cdef size_t size
        cdef cugraph_data_type_id_t dtype
        if python_obj is not None:
            assert_device_accessible(python_obj, "array")
            _get_dlpack_array_info(python_obj, &data, &size, &dtype)
            view_ptr = cugraph_type_erased_device_array_view_create(
                data, size, dtype
            )

        return view_ptr


cdef cugraph_type_erased_device_array_view_t* \
    create_cugraph_type_erased_device_array_view_from_py_obj_as_type(
        python_obj,
        cugraph_data_type_id_t dtype
    ):
        cdef cugraph_type_erased_device_array_view_t* view_ptr = NULL
        cdef void* data = NULL
        cdef size_t size
        cdef cugraph_data_type_id_t actual_dtype
        if python_obj is not None:
            assert_device_accessible(python_obj, "array")
            _get_dlpack_array_info(python_obj, &data, &size, &actual_dtype)
            view_ptr = cugraph_type_erased_device_array_view_create(
                data, size, dtype
            )

        return view_ptr


cdef cugraph_type_erased_host_array_view_t* \
    create_cugraph_type_erased_host_array_view_from_py_obj(python_obj):
        cdef cugraph_type_erased_host_array_view_t* view_ptr = NULL
        cdef void* data = NULL
        cdef size_t size
        cdef cugraph_data_type_id_t dtype
        if python_obj is not None:
            assert_host_accessible(python_obj, "array")
            _get_dlpack_array_info(python_obj, &data, &size, &dtype)
            view_ptr = cugraph_type_erased_host_array_view_create(
                data, size, dtype
            )

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
