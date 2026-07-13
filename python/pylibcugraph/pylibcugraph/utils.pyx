# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
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
    DLDataType,
    DLTensor,
    DLManagedTensor,
    kDLBool,
    kDLFloat,
    kDLInt,
    kDLUInt,
    cugraph_data_type_id_from_dlpack,
    kDLCPU,
    kDLCUDA,
    kDLCUDAHost,
    kDLCUDAManaged,
)

cdef extern from *:
    """
    #include <cuda_runtime_api.h>

    static bool pylibcugraph_is_device_accessible(const void* pointer)
    {
      cudaPointerAttributes attributes{};
      auto const status = cudaPointerGetAttributes(&attributes, pointer);
      if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      return attributes.type == cudaMemoryTypeDevice ||
             attributes.type == cudaMemoryTypeHost ||
             attributes.type == cudaMemoryTypeManaged;
    }

    static bool pylibcugraph_is_host_accessible(const void* pointer)
    {
      cudaPointerAttributes attributes{};
      auto const status = cudaPointerGetAttributes(&attributes, pointer);
      if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      return attributes.type == cudaMemoryTypeHost ||
             attributes.type == cudaMemoryTypeManaged;
    }
    """
    bint pylibcugraph_is_device_accessible(const void* pointer)
    bint pylibcugraph_is_host_accessible(const void* pointer)

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
        return obj.__dlpack__()
    # cuDF Series currently exposes the legacy ``to_dlpack`` method while its
    # zero-copy values view implements the standard protocol. Prefer that view
    # so the input object, rather than a temporary legacy export, owns the
    # storage for the duration of the libcugraph call.
    values = getattr(obj, "values", None)
    if values is not None and hasattr(values, "__dlpack__"):
        return values.__dlpack__()
    raise TypeError("object does not expose a standard DLPack producer")


cpdef bint is_device_accessible(obj):
    _assert_dlpack(obj, "array")
    capsule = _get_dlpack_capsule(obj)
    cdef DLManagedTensor* managed = get_dlpack_managed_tensor_from_capsule(capsule)
    device_type = managed.dl_tensor.device.device_type
    if device_type in (kDLCUDA, kDLCUDAHost, kDLCUDAManaged):
        return True
    return pylibcugraph_is_device_accessible(
        <const void*>(<uintptr_t>managed.dl_tensor.data +
                      managed.dl_tensor.byte_offset)
    )


cpdef bint is_host_accessible(obj):
    _assert_dlpack(obj, "array")
    capsule = _get_dlpack_capsule(obj)
    cdef DLManagedTensor* managed = get_dlpack_managed_tensor_from_capsule(capsule)
    device_type = managed.dl_tensor.device.device_type
    if device_type in (kDLCPU, kDLCUDAHost, kDLCUDAManaged):
        return True
    return pylibcugraph_is_host_accessible(
        <const void*>(<uintptr_t>managed.dl_tensor.data +
                      managed.dl_tensor.byte_offset)
    )


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


cdef DLManagedTensor* get_dlpack_managed_tensor_from_capsule(
    object dlpack_capsule
) except NULL:
    """Return the managed tensor owned by a legacy DLPack capsule.

    The caller must keep ``dlpack_capsule`` alive for as long as the returned
    pointer is in use.
    """
    if not PyCapsule_IsValid(dlpack_capsule, "dltensor"):
        raise ValueError("expected an unconsumed DLPack 'dltensor' capsule")

    return <DLManagedTensor*>PyCapsule_GetPointer(
        dlpack_capsule,
        "dltensor",
    )


cdef cugraph_data_type_id_t get_c_type_from_dlpack_dtype(
    const DLDataType* dl_dtype
):
    cdef cugraph_data_type_id_t c_type
    cdef cugraph_error_t* error = NULL
    cdef cugraph_error_code_t code = \
        cugraph_data_type_id_from_dlpack(dl_dtype, &c_type, &error)
    assert_success(code, error, "cugraph_data_type_id_from_dlpack")
    return c_type


cdef _assert_valid_dlpack_tensor(const DLTensor* tensor):
    if tensor.ndim != 1:
        raise ValueError("pylibcugraph array inputs must be one-dimensional")
    if tensor.shape == NULL:
        raise ValueError("DLPack tensor shape cannot be NULL")
    if tensor.shape[0] < 0:
        raise ValueError("DLPack tensor dimensions cannot be negative")
    if tensor.strides != NULL and tensor.strides[0] != 1:
        raise ValueError("pylibcugraph array inputs must be contiguous")


cpdef cugraph_data_type_id_t get_c_type_from_py_obj(object python_obj) except *:
    _assert_dlpack(python_obj, "array")
    capsule = _get_dlpack_capsule(python_obj)
    cdef DLManagedTensor* managed = get_dlpack_managed_tensor_from_capsule(capsule)
    _assert_valid_dlpack_tensor(&managed.dl_tensor)
    return get_c_type_from_dlpack_dtype(&managed.dl_tensor.dtype)


cdef size_t get_size_from_py_obj(object python_obj) except *:
    _assert_dlpack(python_obj, "array")
    capsule = _get_dlpack_capsule(python_obj)
    cdef DLManagedTensor* managed = get_dlpack_managed_tensor_from_capsule(capsule)
    _assert_valid_dlpack_tensor(&managed.dl_tensor)
    return <size_t>managed.dl_tensor.shape[0]


cdef get_last_item_from_py_obj(object python_obj):
    assert_device_accessible(python_obj, "array")
    producer = python_obj
    if not hasattr(producer, "__dlpack__"):
        producer = producer.values
    return cupy.from_dlpack(producer)[-1].item()


cdef uintptr_t get_data_ptr_from_py_obj(object python_obj) except *:
    _assert_dlpack(python_obj, "array")
    capsule = _get_dlpack_capsule(python_obj)
    cdef DLManagedTensor* managed = get_dlpack_managed_tensor_from_capsule(capsule)
    _assert_valid_dlpack_tensor(&managed.dl_tensor)
    return (<uintptr_t>managed.dl_tensor.data + managed.dl_tensor.byte_offset)


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
        cdef uintptr_t data_ptr = <uintptr_t>NULL
        cdef cugraph_type_erased_device_array_view_t* view_ptr = NULL
        cdef DLManagedTensor* managed = NULL
        if python_obj is not None:
            assert_device_accessible(python_obj, "array")
            capsule = _get_dlpack_capsule(python_obj)
            managed = get_dlpack_managed_tensor_from_capsule(capsule)
            _assert_valid_dlpack_tensor(&managed.dl_tensor)
            data_ptr = (<uintptr_t>managed.dl_tensor.data +
                        managed.dl_tensor.byte_offset)
            view_ptr = cugraph_type_erased_device_array_view_create(
                <void*>data_ptr,
                <size_t>managed.dl_tensor.shape[0],
                get_c_type_from_dlpack_dtype(&managed.dl_tensor.dtype))

        return view_ptr


cdef cugraph_type_erased_host_array_view_t* \
    create_cugraph_type_erased_host_array_view_from_py_obj(python_obj):
        cdef uintptr_t data_ptr = <uintptr_t>NULL
        cdef cugraph_type_erased_host_array_view_t* view_ptr = NULL
        cdef DLManagedTensor* managed = NULL
        if python_obj is not None:
            assert_host_accessible(python_obj, "array")
            capsule = _get_dlpack_capsule(python_obj)
            managed = get_dlpack_managed_tensor_from_capsule(capsule)
            _assert_valid_dlpack_tensor(&managed.dl_tensor)
            data_ptr = (<uintptr_t>managed.dl_tensor.data +
                        managed.dl_tensor.byte_offset)
            view_ptr = cugraph_type_erased_host_array_view_create(
                <void*>data_ptr,
                <size_t>managed.dl_tensor.shape[0],
                get_c_type_from_dlpack_dtype(&managed.dl_tensor.dtype))

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
