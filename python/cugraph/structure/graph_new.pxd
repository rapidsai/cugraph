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
from libcpp.memory cimport unique_ptr

from rmm._lib.device_buffer cimport device_buffer

cdef extern from "graph.hpp" namespace "cugraph::experimental":

    ctypedef enum PropType:
        PROP_UNDEF "cugraph::experimental::PROP_UNDEF"
        PROP_FALSE "cugraph::experimental::PROP_FALSE"
        PROP_TRUE "cugraph::experimental::PROP_TRUE"

    ctypedef enum DegreeDirection:
        DIRECTION_IN_PLUS_OUT "cugraph::experimental::DegreeDirection::IN_PLUS_OUT"
        DIRECTION_IN "cugraph::experimental::DegreeDirection::IN"
        DIRECTION_OUT "cugraph::experimental::DegreeDirection::OUT"

    struct GraphProperties:
        bool directed
        bool weighted
        bool multigraph
        bool bipartite
        bool tree
        PropType has_negative_edges

    cdef cppclass GraphViewBase[VT,ET,WT]:
        WT *edge_data
        GraphProperties prop
        VT number_of_vertices
        ET number_of_edges

        void get_vertex_identifiers(VT *) const

        GraphViewBase(WT*,VT,ET)

    cdef cppclass GraphCOOView[VT,ET,WT](GraphViewBase[VT,ET,WT]):
        VT *src_indices
        VT *dst_indices

        void degree(ET *,DegreeDirection) const

        GraphCOOView()
        GraphCOOView(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCompressedSparseBaseView[VT,ET,WT](GraphViewBase[VT,ET,WT]):
        ET *offsets
        VT *indices

        void get_source_indices(VT *) const
        void degree(ET *,DegreeDirection) const
        
        GraphCompressedSparseBaseView(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCSRView[VT,ET,WT](GraphCompressedSparseBaseView[VT,ET,WT]):
        GraphCSRView()
        GraphCSRView(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCSCView[VT,ET,WT](GraphCompressedSparseBaseView[VT,ET,WT]):
        GraphCSCView()
        GraphCSCView(const VT *, const ET *, const WT *, size_t, size_t)

    cdef cppclass GraphCOOContents[VT,ET,WT]:
        VT number_of_vertices
        ET number_of_edges
        unique_ptr[device_buffer] src_indices
        unique_ptr[device_buffer] dst_indices
        unique_ptr[device_buffer] edge_data

    cdef cppclass GraphCOO[VT,ET,WT]:
        GraphCOO(
                VT nv,
                ET ne,
                bool has_data) except+
        GraphCOOContents[VT,ET,WT] release()
        GraphCOOView[VT,ET,WT] view()

    cdef cppclass GraphSparseContents[VT,ET,WT]:
        VT number_of_vertices
        ET number_of_edges
        unique_ptr[device_buffer] offsets
        unique_ptr[device_buffer] indices
        unique_ptr[device_buffer] edge_data

    cdef cppclass GraphCSC[VT,ET,WT]:
        GraphCSC(
                VT nv,
                ET ne,
                bool has_data) except+
        GraphSparseContents[VT,ET,WT] release()
        GraphCSCView[VT,ET,WT] view()

    cdef cppclass GraphCSR[VT,ET,WT]:
        GraphCSR(
                VT nv,
                ET ne,
                bool has_data) except+
        GraphSparseContents[VT,ET,WT] release()
        GraphCSRView[VT,ET,WT] view()



cdef extern from "algorithms.hpp" namespace "cugraph":

    cdef ET get_two_hop_neighbors[VT,ET,WT](
        const GraphCSRView[VT, ET, WT] &graph,
        VT **first,
        VT **second) except +

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[GraphCOO[int,int,float]] move(unique_ptr[GraphCOO[int,int,float]])
    cdef unique_ptr[GraphCOO[int,int,double]] move(unique_ptr[GraphCOO[int,int,double]])
    cdef GraphCOOContents[int,int,float] move(GraphCOOContents[int,int,float])
    cdef GraphCOOContents[int,int,double] move(GraphCOOContents[int,int,double])
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
    cdef unique_ptr[GraphCSR[int,int,float]] move(unique_ptr[GraphCSR[int,int,float]])
    cdef unique_ptr[GraphCSR[int,int,double]] move(unique_ptr[GraphCSR[int,int,double]])
