# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.types cimport (
    cugraph_data_type_id_t,
)

from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_host_array_view_t,
)

from libc.stdint cimport uintptr_t
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.dlpack_interop cimport (
    cugraph_dlpack_data_type_t,
    cugraph_dlpack_tensor_t,
)


cdef assert_success(cugraph_error_code_t code,
                    cugraph_error_t* err,
                    api_name)

cpdef bint is_device_accessible(obj)

cpdef bint is_host_accessible(obj)

cdef assert_device_accessible(obj, var_name, allow_None=*)

cdef assert_host_accessible(obj, var_name, allow_None=*)

cdef get_numpy_type_from_c_type(cugraph_data_type_id_t c_type)

cdef cugraph_dlpack_tensor_t* get_dlpack_tensor_from_capsule(
    object dlpack_capsule
) except NULL

cdef cugraph_data_type_id_t get_c_type_from_dlpack_dtype(
    const cugraph_dlpack_data_type_t* dl_dtype
)

cpdef cugraph_data_type_id_t get_c_type_from_py_obj(object python_obj) except *

cdef size_t get_size_from_py_obj(object python_obj) except *

cdef get_last_item_from_py_obj(object python_obj)

cdef uintptr_t get_data_ptr_from_py_obj(object python_obj) except *

cdef get_c_type_from_numpy_type(numpy_type)

cdef get_c_weight_type_from_numpy_edge_ids_type(numpy_type)

cdef get_numpy_edge_ids_type_from_c_weight_type(cugraph_data_type_id_t c_type)

cdef copy_to_cupy_array(
   cugraph_resource_handle_t* c_resource_handle_ptr,
   cugraph_type_erased_device_array_view_t* device_array_view_ptr)

cdef copy_to_cupy_array_ids(
   cugraph_resource_handle_t* c_resource_handle_ptr,
   cugraph_type_erased_device_array_view_t* device_array_view_ptr)

cdef cugraph_type_erased_device_array_view_t* \
    create_cugraph_type_erased_device_array_view_from_py_obj(python_obj)

cdef cugraph_type_erased_host_array_view_t* \
    create_cugraph_type_erased_host_array_view_from_py_obj(python_obj)

cdef create_cupy_array_view_for_device_ptr(
    cugraph_type_erased_device_array_view_t* device_array_view_ptr,
    owning_py_object)

cdef extern from "stdint.h":
    size_t SIZE_MAX
