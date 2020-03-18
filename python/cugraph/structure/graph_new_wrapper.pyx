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

cimport cugraph.structure.graph_new as c_graph
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import rmm
import numpy as np


"""
cdef cppclass GraphBase[WT]:
    cdef GraphBase[WT] c_base

    def __cinit__(self, WT const *edge_data, size_t number_of_vertices, size_t number_of_edges):
        self.c_base = GraphBase(edge_data, number_of_vertices, number_of_edges)

cdef cppclass GraphCOO[VT,WT]:
    cdef GraphCOO c_base

    def __cinit__(self):
        self.c_base = GraphCOO()
        
    def __cinit__(self, VT const *src_indices, VT const *dst_indices, WT const *edge_data, size_t number_of_vertices, size_t number_of_edges):
        self.c_base = GraphCOO(src_indices, dst_indices, edge_data, number_of_vertices, number_of_edges)

cdef cppclass GraphCSRBase[VT,WT]:
    cdef GraphCSRBase c_base

    def __cinit__(self, VT const *src_indices, VT const *dst_indices, WT const *edge_data, size_t number_of_vertices, size_t number_of_edges):
        self.c_base = GraphCSRBase(src_indices, dst_indices, edge_data, number_of_vertices, number_of_edges)

cdef cppclass GraphCSR[VT,WT]:
    cdef GraphCSR c_base

    def __cinit__(self):
        self.c_base = GraphCSR()
        
    def __cinit__(self, VT const *src_indices, VT const *dst_indices, WT const *edge_data, size_t number_of_vertices, size_t number_of_edges):
        self.c_base = GraphCSR(src_indices, dst_indices, edge_data, number_of_vertices, number_of_edges)

cdef cppclass GraphCSC[VT,WT]:
    cdef GraphCSC c_base

    def __cinit__(self):
        self.c_base = GraphCSC()
        
    def __cinit__(self, VT const *src_indices, VT const *dst_indices, WT const *edge_data, size_t number_of_vertices, size_t number_of_edges):
        self.c_base = GraphCSC(src_indices, dst_indices, edge_data, number_of_vertices, number_of_edges)
"""
