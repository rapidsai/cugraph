# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from pylibraft.common.handle cimport handle_t


# C++ graph utilities
cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":

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
        void *segment_offsets,
        size_t num_segments,
        numberTypeEnum vertexType,
        numberTypeEnum edgeType,
        numberTypeEnum weightType,
        size_t num_local_edges,
        size_t num_global_vertices,
        size_t num_global_edges,
        bool is_weighted,
        bool is_symmetric,
        bool transposed,
        bool multi_gpu) except +

    ctypedef enum graphTypeEnum:
        LegacyCSR "cugraph::cython::graphTypeEnum::LegacyCSR"
        LegacyCSC "cugraph::cython::graphTypeEnum::LegacyCSC"
        LegacyCOO "cugraph::cython::graphTypeEnum::LegacyCOO"

    cdef cppclass cy_multi_edgelists_t:
        size_t number_of_vertices
        size_t number_of_edges
        size_t number_of_subgraph
        unique_ptr[device_buffer] src_indices
        unique_ptr[device_buffer] dst_indices
        unique_ptr[device_buffer] edge_data
        unique_ptr[device_buffer] subgraph_offsets

    cdef cppclass random_walk_ret_t:
        size_t coalesced_sz_v_
        size_t coalesced_sz_w_
        size_t num_paths_
        size_t max_depth_
        unique_ptr[device_buffer] d_coalesced_v_
        unique_ptr[device_buffer] d_coalesced_w_
        unique_ptr[device_buffer] d_sizes_

    cdef cppclass random_walk_path_t:
        unique_ptr[device_buffer] d_v_offsets
        unique_ptr[device_buffer] d_w_sizes
        unique_ptr[device_buffer] d_w_offsets

    cdef cppclass graph_generator_t:
        unique_ptr[device_buffer] d_source
        unique_ptr[device_buffer] d_destination

cdef extern from "<utility>" namespace "std" nogil:
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
    cdef cy_multi_edgelists_t move(cy_multi_edgelists_t)
    cdef unique_ptr[cy_multi_edgelists_t] move(unique_ptr[cy_multi_edgelists_t])

# renumber_edgelist() interface utilities:
#
#
# 1. `cdef extern partition_t`:
#
cdef extern from "cugraph/graph_view.hpp" namespace "cugraph":

    cdef cppclass partition_t[vertex_t]:
        pass


# 2. return type for shuffle:
#
cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":

    cdef cppclass major_minor_weights_t[vertex_t, edge_t, weight_t]:
        major_minor_weights_t(const handle_t &handle)
        pair[unique_ptr[device_buffer], size_t] get_major_wrap()
        pair[unique_ptr[device_buffer], size_t] get_minor_wrap()
        pair[unique_ptr[device_buffer], size_t] get_weights_wrap()
        unique_ptr[vector[edge_t]] get_edge_counts_wrap()


ctypedef fused shuffled_vertices_t:
    major_minor_weights_t[int, int, float]
    major_minor_weights_t[int, int, double]
    major_minor_weights_t[int, long, float]
    major_minor_weights_t[int, long, double]
    major_minor_weights_t[long, long, float]
    major_minor_weights_t[long, long, double]

# 3. return type for renumber:
#
cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":

    cdef cppclass renum_tuple_t[vertex_t, edge_t]:
        renum_tuple_t(const handle_t &handle)
        pair[unique_ptr[device_buffer], size_t] get_dv_wrap()
        vertex_t& get_num_vertices()
        edge_t& get_num_edges()
        vector[vertex_t]& get_segment_offsets()
        unique_ptr[vector[vertex_t]] get_segment_offsets_wrap()
        int get_part_row_size()
        int get_part_col_size()
        int get_part_comm_rank()
        unique_ptr[vector[vertex_t]] get_partition_offsets_wrap()
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
cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":

    cdef unique_ptr[major_minor_weights_t[vertex_t, edge_t, weight_t]] call_shuffle[vertex_t, edge_t, weight_t](
        const handle_t &handle,
        vertex_t *edgelist_major_vertices,
        vertex_t *edgelist_minor_vertices,
        weight_t* edgelist_weights,
        edge_t num_edges,
        bool is_weighted) except +

# 5. `renumber_edgelist()` wrapper
#
cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":

    cdef unique_ptr[renum_tuple_t[vertex_t, edge_t]] call_renumber[vertex_t, edge_t](
        const handle_t &handle,
        vertex_t *edgelist_major_vertices,
        vertex_t *edgelist_minor_vertices,
        const vector[edge_t]& edge_counts,
        bool store_transposed,
        bool do_check,
        bool multi_gpu) except +
