# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

#from cugraph.structure.graph cimport *
from cudf._lib.cudf cimport get_column_data_ptr, get_column_valid_ptr
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import numpy as np


cdef gdf_column get_gdf_column_view(col):
    """
    This function returns a C++ gdf_column object from the Python cudf Series
    object by shallow copying. The returned C++ object is expected to be used
    as a temporary variable to pass the column data encapsulated in the Python
    cudf Series object to C++ functions expecting (pointers to) C++ gdf_column
    objects. It is the caller's responsibility to insure that col out-lives the
    returned view object. cudf has column_view_from_column and using this is,
    in general, better design than creating our own, but we will keep this as
    cudf is planning to remove the function. cudf plans to redesign
    cudf::column to fundamentally solve this problem, so once they finished the
    redesign, we need to update this code to use their new features. Until that
    time, we may rely on this as a temporary solution.
    """

    cdef gdf_column c_col
    cdef uintptr_t data_ptr = get_column_data_ptr(col._column)
    cdef uintptr_t valid_ptr
    if col._column._mask is None:
        valid_ptr = 0
    else:
        valid_ptr = get_column_valid_ptr(col._column)
    cdef uintptr_t category = 0
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(
        time_unit=TIME_UNIT_NONE,
        category=<void*>category
    )

    err = gdf_column_view_augmented(<gdf_column*> &c_col,
                                    <void*> data_ptr,
                                    <valid_type*> valid_ptr,
                                    <size_type> len(col),
                                    gdf_dtype_from_value(col),
                                    <size_type> col.null_count,
                                    c_extra_dtype_info)
    libcudf.cudf.check_gdf_error(err)

    return c_col


cdef gdf_column* get_gdf_column_ptr(ipc_data_ptr, col_len):
    cdef gdf_column* c_col = <gdf_column*>malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr = ipc_data_ptr
    cdef uintptr_t valid_ptr = 0
    cdef uintptr_t category = 0
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(
        time_unit=TIME_UNIT_NONE,
        category=<void*>category
    )

    err = gdf_column_view_augmented(<gdf_column*> c_col,
                                    <void*> data_ptr,
                                    <valid_type*> valid_ptr,
                                    <size_type> col_len,
                                    gdf_dtype_from_value(None, np.int32),
                                    <size_type> 0,
                                    c_extra_dtype_info)
    libcudf.cudf.check_gdf_error(err)
    return c_col

#
#  Really want something like this to clean up the code... but
#  this can't work with the current interface.
#
#cdef wrap_column_not_working(col_ptr):
#    cdef gdf_column * col = <gdf_column *> col_ptr
#
#    print("  size = ", col.size)
#    print("  data = ", <uintptr_t> col.data)
#    print("col.dtype = ", col.dtype)
#    print("dtypes_inv = ", dtypes_inv)
#    print("  inv = ", dtypes_inv[col.dtype])
#
#    return rmm.device_array_from_ptr(<uintptr_t> col.data,
#                                     nelem=col.size,
#                                     dtype=dtypes_inv[col.dtype],
#                                     finalizer=rmm._make_finalizer(<uintptr_t> col_ptr, 0))
#