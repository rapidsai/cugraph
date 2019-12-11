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

cimport cugraph.utilities.grmat as c_grmat
from cugraph.structure.graph cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def grmat_gen(argv):
    """
    Call grmat_gen
    """
    cdef size_t vertices = 0
    cdef size_t edges = 0
    cdef gdf_column* c_source_col = <gdf_column*>malloc(sizeof(gdf_column))
    c_source_col.dtype = GDF_INT32
    c_source_col.valid = NULL
    c_source_col.null_count = 0
    cdef gdf_column* c_dest_col = <gdf_column*>malloc(sizeof(gdf_column))
    c_dest_col.dtype = GDF_INT32
    c_dest_col.valid = NULL
    c_dest_col.null_count = 0
    #cdef gdf_column* c_val_col = <gdf_column*>malloc(sizeof(gdf_column))
    argv_bytes = argv.encode()
    cdef char* c_argv = argv_bytes

    c_grmat.grmat_gen (<char*>c_argv, vertices, edges, 
                       <gdf_column*>c_source_col, 
                       <gdf_column*>c_dest_col, <gdf_column*>0)


    col_size = c_source_col.size
    cdef uintptr_t src_col_data = <uintptr_t>c_source_col.data
    cdef uintptr_t dest_col_data = <uintptr_t>c_dest_col.data
    
    src_data = rmm.device_array_from_ptr(src_col_data,
                                     nelem=col_size,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(src_col_data, 0))
    dest_data = rmm.device_array_from_ptr(dest_col_data,
                                     nelem=col_size,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(dest_col_data, 0))
    return vertices, edges, cudf.Series(src_data), cudf.Series(dest_data)
