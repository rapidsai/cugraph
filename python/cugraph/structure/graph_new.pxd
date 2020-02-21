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

from libcpp cimport bool

cdef extern from "graph.hpp" namespace "cugraph::experimental":

    ctypedef enum prop_type:
        GDF_PROP_UNDEF = 0
        GDF_PROP_FALSE
        GDF_PROP_TRUE

    struct GraphProperties:
        bool directed
        bool weighted
        bool multigraph
        bool bipartite
        bool tree
        prop_type has_negative_edges

    cdef cppclass GraphBase[VT,ET,WT]:
        WT *edge_data
        GraphProperties prop
        VT number_of_vertices
        ET number_of_edges
        GraphBase(WT*,VT,ET)

    cdef cppclass GraphCOO[VT,ET,WT](GraphBase[VT,ET,WT]):
        const VT *src_indices
        const VT *dst_indices
        GraphCOO()
        GraphCOO(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCompressedSparseBase[VT,ET,WT](GraphBase[VT,ET,WT]):
        const VT *offsets
        const VT *indices

        void get_vertex_identifiers(VT *) const
        void get_source_indices(VT *) const
        
        GraphCompressedSparseBase(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCSR[VT,ET,WT](GraphCompressedSparseBase[VT,ET,WT]):
        GraphCSR()
        GraphCSR(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCSC[VT,ET,WT](GraphCompressedSparseBase[VT,ET,WT]):
        GraphCSC()
        GraphCSC(const VT *, const ET *, const WT *, size_t, size_t)
