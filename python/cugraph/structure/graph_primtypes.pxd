# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport device_buffer

cdef extern from "raft/handle.hpp" namespace "raft":
    cdef cppclass handle_t:
        handle_t() except +

cdef extern from "graph.hpp" namespace "cugraph":

    ctypedef enum PropType:
        PROP_UNDEF "cugraph::PROP_UNDEF"
        PROP_FALSE "cugraph::PROP_FALSE"
        PROP_TRUE "cugraph::PROP_TRUE"

    ctypedef enum DegreeDirection:
        DIRECTION_IN_PLUS_OUT "cugraph::DegreeDirection::IN_PLUS_OUT"
        DIRECTION_IN "cugraph::DegreeDirection::IN"
        DIRECTION_OUT "cugraph::DegreeDirection::OUT"

    struct GraphProperties:
        bool directed
        bool weighted
        bool multigraph
        bool bipartite
        bool tree
        PropType has_negative_edges

    cdef cppclass GraphViewBase[VT,ET,WT]:
        WT *edge_data
        handle_t *handle;
        GraphProperties prop
        VT number_of_vertices
        ET number_of_edges
        VT* local_vertices
        ET* local_edges
        VT* local_offsets
        void set_handle(handle_t*)
        void set_local_data(VT* local_vertices_, ET* local_edges_, VT* local_offsets_)
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

    cdef unique_ptr[GraphCOO[VT, ET, WT]] get_two_hop_neighbors[VT,ET,WT](
        const GraphCSRView[VT, ET, WT] &graph) except +

cdef extern from "functions.hpp" namespace "cugraph":

    cdef unique_ptr[device_buffer] renumber_vertices[VT_IN,VT_OUT,ET](
        ET number_of_edges,
        const VT_IN *src,
        const VT_IN *dst,
        VT_OUT *src_renumbered,
        VT_OUT *dst_renumbered,
        ET *map_size) except +


# renumber_edgelist() interface:
#
#
# 1. `cdef extern partition_t`:
#
cdef extern from "experimental/graph_view.hpp" namespace "cugraph::experimental":

    cdef cppclass partition_t[vertex_t]:
        pass


# 2. return type for shuffle:
#
cdef extern from "utilities/cython.hpp" namespace "cugraph::cython":

    cdef cppclass major_minor_weights_t[vertex_t, weight_t]:
        major_minor_weights_t(const handle_t &handle)
        pair[unique_ptr[device_buffer], size_t] get_major_wrap()
        pair[unique_ptr[device_buffer], size_t] get_minor_wrap()
        pair[unique_ptr[device_buffer], size_t] get_weights_wrap()


ctypedef fused shuffled_vertices_t:
    major_minor_weights_t[int, float]
    major_minor_weights_t[int, double]
    major_minor_weights_t[long, float]
    major_minor_weights_t[long, double]
    
# 3. return type for renumber:
#
cdef extern from "utilities/cython.hpp" namespace "cugraph::cython":

    cdef cppclass renum_quad_t[vertex_t, edge_t]:
        renum_quad_t(const handle_t &handle)
        pair[unique_ptr[device_buffer], size_t] get_dv_wrap()
        vertex_t& get_num_vertices()
        edge_t& get_num_edges()
        int get_part_row_size()
        int get_part_col_size()
        int get_part_comm_rank()
        unique_ptr[vector[vertex_t]] get_partition_offsets()
        pair[vertex_t, vertex_t] get_part_local_vertex_range()
        vertex_t get_part_local_vertex_first()
        vertex_t get_part_local_vertex_last()
        pair[vertex_t, vertex_t] get_part_vertex_partition_range(size_t vertex_partition_idx)
        vertex_t get_part_vertex_partition_first(size_t vertex_partition_idx)
        vertex_t get_part_vertex_partition_last(size_t vertex_partition_idx)
        vertex_t get_part_vertex_partition_size(size_t vertex_partition_idx)
        size_t get_part_number_of_matrix_partitions()
        vertex_t get_part_matrix_partition_major_first(size_t partition_idx)
        vertex_t get_part_matrix_partition_major_last(size_t partition_idx)
        vertex_t get_part_matrix_partition_major_value_start_offset(size_t partition_idx)
        pair[vertex_t, vertex_t] get_part_matrix_partition_minor_range()
        vertex_t get_part_matrix_partition_minor_first()
        vertex_t get_part_matrix_partition_minor_last()        

# 4. `sort_and_shuffle_values()` wrapper:
#
cdef extern from "utilities/cython.hpp" namespace "cugraph::cython":

    cdef unique_ptr[major_minor_weights_t[vertex_t, weight_t]] call_shuffle[vertex_t, edge_t, weight_t](
        const handle_t &handle,
        vertex_t *edgelist_major_vertices,
        vertex_t *edgelist_minor_vertices,
        weight_t* edgelist_weights,
        edge_t num_edges,
        bool is_hyper_partitioned) except +


# 5. `renumber_edgelist()` wrapper
#
cdef extern from "utilities/cython.hpp" namespace "cugraph::cython":

    cdef unique_ptr[renum_quad_t[vertex_t, edge_t]] call_renumber[vertex_t, edge_t](
        const handle_t &handle,
        vertex_t *edgelist_major_vertices,
        vertex_t *edgelist_minor_vertices,
        edge_t num_edges,
        bool is_hyper_partitioned,
        bool do_check,
        bool multi_gpu) except +


cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[GraphCOO[int,int,float]] move(unique_ptr[GraphCOO[int,int,float]])
    cdef unique_ptr[GraphCOO[int,int,double]] move(unique_ptr[GraphCOO[int,int,double]])
    cdef GraphCOOContents[int,int,float] move(GraphCOOContents[int,int,float])
    cdef GraphCOOContents[int,int,double] move(GraphCOOContents[int,int,double])
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
    cdef unique_ptr[GraphCSR[int,int,float]] move(unique_ptr[GraphCSR[int,int,float]])
    cdef unique_ptr[GraphCSR[int,int,double]] move(unique_ptr[GraphCSR[int,int,double]])
    cdef GraphSparseContents[int,int,float] move(GraphSparseContents[int,int,float])
    cdef GraphSparseContents[int,int,double] move(GraphSparseContents[int,int,double])

ctypedef unique_ptr[GraphCOO[int,int,float]] GraphCOOPtrFloat
ctypedef unique_ptr[GraphCOO[int,int,double]] GraphCOOPtrDouble

ctypedef fused GraphCOOPtrType:
    GraphCOOPtrFloat
    GraphCOOPtrDouble

ctypedef unique_ptr[GraphCSR[int,int,float]] GraphCSRPtrFloat
ctypedef unique_ptr[GraphCSR[int,int,double]] GraphCSRPtrDouble

ctypedef fused GraphCSRPtrType:
    GraphCSRPtrFloat
    GraphCSRPtrDouble

ctypedef GraphCOOView[int,int,float] GraphCOOViewFloat
ctypedef GraphCOOView[int,int,double] GraphCOOViewDouble
ctypedef GraphCSRView[int,int,float] GraphCSRViewFloat
ctypedef GraphCSRView[int,int,double] GraphCSRViewDouble

ctypedef fused GraphCOOViewType:
    GraphCOOViewFloat
    GraphCOOViewDouble

ctypedef fused GraphCSRViewType:
    GraphCSRViewFloat
    GraphCSRViewDouble

ctypedef fused GraphViewType:
    GraphCOOViewFloat
    GraphCOOViewDouble
    GraphCSRViewFloat
    GraphCSRViewDouble

cdef coo_to_df(GraphCOOPtrType graph)
cdef csr_to_series(GraphCSRPtrType graph)
cdef GraphViewType get_graph_view(input_graph, bool weightless=*, GraphViewType* dummy=*)


# C++ utilities specifically for Cython
cdef extern from "utilities/cython.hpp" namespace "cugraph::cython":

    ctypedef enum numberTypeEnum:
        int32Type "cugraph::cython::numberTypeEnum::int32Type"
        int64Type "cugraph::cython::numberTypeEnum::int64Type"
        floatType "cugraph::cython::numberTypeEnum::floatType"
        doubleType "cugraph::cython::numberTypeEnum::doubleType"

    cdef cppclass graph_container_t:
       pass

    cdef void populate_graph_container(
        graph_container_t &graph_container,
        handle_t &handle,
        void *src_vertices,
        void *dst_vertices,
        void *weights,
        void *vertex_partition_offsets,
        numberTypeEnum vertexType,
        numberTypeEnum edgeType,
        numberTypeEnum weightType,
        size_t num_partition_edges,
        size_t num_global_vertices,
        size_t num_global_edges,
        bool sorted_by_degree,
        bool transposed,
        bool multi_gpu) except +

    ctypedef enum graphTypeEnum:
        LegacyCSR "cugraph::cython::graphTypeEnum::LegacyCSR"
        LegacyCSC "cugraph::cython::graphTypeEnum::LegacyCSC"
        LegacyCOO "cugraph::cython::graphTypeEnum::LegacyCOO"

    cdef void populate_graph_container_legacy(
        graph_container_t &graph_container,
        graphTypeEnum legacyType,
        const handle_t &handle,
        void *offsets,
        void *indices,
        void *weights,
        numberTypeEnum offsetType,
        numberTypeEnum indexType,
        numberTypeEnum weightType,
        size_t num_global_vertices,
        size_t num_global_edges,
        int *local_vertices,
        int *local_edges,
        int *local_offsets) except +
