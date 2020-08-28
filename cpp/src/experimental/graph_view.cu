/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph_view.hpp>
#include <utilities/error.hpp>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/count.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace experimental {

namespace {

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_view_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t>
struct out_of_range_t {
  vertex_t min{};
  vertex_t max{};

  __device__ bool operator()(vertex_t v) { return (v < min) || (v >= max); }
};

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_view_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t, typename edge_t>
struct degree_from_offsets_t {
  edge_t const* offsets{nullptr};

  __device__ edge_t operator()(vertex_t v) { return offsets[v + 1] - offsets[v]; }
};

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_view_t(raft::handle_t const& handle,
               std::vector<edge_t const*> const& adj_matrix_partition_offsets,
               std::vector<vertex_t const*> const& adj_matrix_partition_indices,
               std::vector<weight_t const*> const& adj_matrix_partition_weights,
               std::vector<vertex_t> const& vertex_partition_segment_offsets,
               partition_t<vertex_t> const& partition,
               vertex_t number_of_vertices,
               edge_t number_of_edges,
               bool is_symmetric,
               bool is_multigraph,
               bool is_weighted,
               bool sorted_by_global_degree_within_vertex_partition,
               bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, number_of_vertices, number_of_edges, is_symmetric, is_multigraph, is_weighted),
    adj_matrix_partition_offsets_(adj_matrix_partition_offsets),
    adj_matrix_partition_indices_(adj_matrix_partition_indices),
    adj_matrix_partition_weights_(adj_matrix_partition_weights),
    partition_(partition),
    vertex_partition_segment_offsets_(vertex_partition_segment_offsets)
{
  // cheap error checks

  auto comm_p_size     = this->get_handle_ptr()->get_comms().get_size();
  auto comm_p_row_size = this->get_handle_ptr()->get_subcomm(comm_p_row_key).get_size();
  auto comm_p_col_size = this->get_handle_ptr()->get_subcomm(comm_p_col_key).get_size();

  CUGRAPH_EXPECTS(adj_matrix_partition_offsets.size() == adj_matrix_partition_indices.size(),
                  "Invalid API parameter: adj_matrix_partition_offsets.size() and "
                  "adj_matrix_partition_indices.size() should coincide.");
  CUGRAPH_EXPECTS(
    (is_weighted && (adj_matrix_partition_weights.size() == adj_matrix_partition_offsets.size())) ||
      (!is_weighted && (adj_matrix_partition_weights.size() == 0)),
    "Invalid API parameter: adj_matrix_partition_weights.size() should coincide with "
    "adj_matrix_partition_offsets.size() (if is_weighted is true) or 0 (if is_weighted is false).");

  CUGRAPH_EXPECTS(
    (partition.hypergraph_partitioned &&
     (adj_matrix_partition_offsets.size() == static_cast<size_t>(comm_p_row_size))) ||
      (!(partition.hypergraph_partitioned) && (adj_matrix_partition_offsets.size() == 1)),
    "Invalid API parameter: errneous adj_matrix_partition_offsets.size().");

  CUGRAPH_EXPECTS((sorted_by_global_degree_within_vertex_partition &&
                   (vertex_partition_segment_offsets.size() ==
                    comm_p_col_size * (detail::num_segments_per_vertex_partition + 1))) ||
                    (!sorted_by_global_degree_within_vertex_partition &&
                     (vertex_partition_segment_offsets.size() == 0)),
                  "Invalid API parameter: vertex_partition_segment_offsets.size() does not match "
                  "with sorted_by_global_degree_within_vertex_partition.");

  CUGRAPH_EXPECTS(partition.vertex_partition_offsets.size() == static_cast<size_t>(comm_p_size),
                  "Invalid API parameter: erroneous partition.vertex_partition_offsets.size().");

  // optinoal expensive checks

  if (do_expensive_check) {
    auto default_stream = this->get_handle_ptr()->get_stream();

    auto comm_p_row_rank = this->get_handle_ptr()->get_subcomm(comm_p_row_key).get_rank();
    auto comm_p_col_rank = this->get_handle_ptr()->get_subcomm(comm_p_col_key).get_rank();

    for (size_t i = 0; i < adj_matrix_partition_offsets.size(); ++i) {
      auto major_first =
        partition.hypergraph_partitioned
          ? partition.vertex_partition_offsets[comm_p_row_size * i + comm_p_row_rank]
          : partition.vertex_partition_offsets[comm_p_row_rank * comm_p_col_size];
      auto major_last =
        partition.hypergraph_partitioned
          ? partition.vertex_partition_offsets[comm_p_row_size * i + comm_p_row_rank + 1]
          : partition.vertex_partition_offsets[(comm_p_row_rank + 1) * comm_p_col_size];
      auto minor_first = partition.vertex_partition_offsets[comm_p_col_rank * comm_p_row_size];
      auto minor_last = partition.vertex_partition_offsets[(comm_p_col_rank + 1) * comm_p_row_size];
      CUGRAPH_EXPECTS(
        thrust::is_sorted(rmm::exec_policy(default_stream)->on(default_stream),
                          adj_matrix_partition_offsets[i],
                          adj_matrix_partition_offsets[i] + (major_last - major_first + 1)),
        "Invalid API parameter: adj_matrix_partition_offsets[] is not sorted.");
      edge_t number_of_local_edges{};
      raft::update_host(&number_of_local_edges,
                        adj_matrix_partition_offsets[i] + (major_last - major_first),
                        1,
                        default_stream);

      // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
      CUGRAPH_EXPECTS(
        thrust::count_if(rmm::exec_policy(default_stream)->on(default_stream),
                         adj_matrix_partition_indices[i],
                         adj_matrix_partition_indices[i] + number_of_local_edges,
                         out_of_range_t<vertex_t>{minor_first, minor_last}) == 0,
        "Invalid API parameter: adj_matrix_partition_indices[] have out-of-range vertex IDs.");

      edge_t number_of_local_edges_sum{};
      this->get_handle_ptr()->get_comms().allreduce(&number_of_local_edges,
                                                    &number_of_local_edges_sum,
                                                    1,
                                                    raft::comms::op_t::SUM,
                                                    default_stream);
      CUGRAPH_EXPECTS(number_of_local_edges_sum == number_of_edges,
                      "Invalid API parameter: the sum of local edges doe counts not match with "
                      "number_of_local_edges.");
    }

    if (sorted_by_global_degree_within_vertex_partition) {
      auto degrees = detail::compute_major_degree(handle, adj_matrix_partition_offsets, partition);
      CUGRAPH_EXPECTS(thrust::is_sorted(rmm::exec_policy(default_stream)->on(default_stream),
                                        degrees.begin(),
                                        degrees.end(),
                                        thrust::greater<edge_t>{}),
                      "Invalid API parameter: sorted_by_global_degree_within_vertex_partition is "
                      "set to true, but degrees are not non-ascending.");

      for (int i = 0; i < comm_p_col_size; ++i) {
        CUGRAPH_EXPECTS(std::is_sorted(vertex_partition_segment_offsets.begin() +
                                         (detail::num_segments_per_vertex_partition + 1) * i,
                                       vertex_partition_segment_offsets.begin() +
                                         (detail::num_segments_per_vertex_partition + 1) * (i + 1)),
                        "Invalid API parameter: erroneous vertex_partition_segment_offsets.");
        CUGRAPH_EXPECTS(
          vertex_partition_segment_offsets[(detail::num_segments_per_vertex_partition + 1) * i] ==
            0,
          "Invalid API parameter: erroneous vertex_partition_segment_offsets.");
        auto vertex_partition_idx = partition.hypergraph_partitioned
                                      ? comm_p_row_size * i + comm_p_row_rank
                                      : comm_p_col_size * comm_p_row_rank + i;
        CUGRAPH_EXPECTS(
          vertex_partition_segment_offsets[(detail::num_segments_per_vertex_partition + 1) * i +
                                           detail::num_segments_per_vertex_partition] ==
            partition.vertex_partition_offsets[vertex_partition_idx],
          "Invalid API parameter: erroneous vertex_partition_segment_offsets.");
      }
    }

    CUGRAPH_EXPECTS(
      std::is_sorted(partition.vertex_partition_offsets.begin(),
                     partition.vertex_partition_offsets.end()),
      "Invalid API parameter: partition.vertex_partition_offsets values should be non-descending.");
    CUGRAPH_EXPECTS(partition.vertex_partition_offsets[0] == edge_t{0},
                    "Invalid API parameter: partition.vertex_partition_offsets[0] should be 0.");
    CUGRAPH_EXPECTS(partition.vertex_partition_offsets.back() == number_of_vertices,
                    "Invalid API parameter: partition.vertex_partition_offsets.back() should be "
                    "number_of_vertices.");

    if (is_symmetric) {}
    if (!is_multigraph) {}
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_view_t<vertex_t,
             edge_t,
             weight_t,
             store_transposed,
             multi_gpu,
             std::enable_if_t<!multi_gpu>>::graph_view_t(raft::handle_t const& handle,
                                                         edge_t const* offsets,
                                                         vertex_t const* indices,
                                                         weight_t const* weights,
                                                         std::vector<vertex_t> const&
                                                           segment_offsets,
                                                         vertex_t number_of_vertices,
                                                         edge_t number_of_edges,
                                                         bool is_symmetric,
                                                         bool is_multigraph,
                                                         bool is_weighted,
                                                         bool sorted_by_degree,
                                                         bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, number_of_vertices, number_of_edges, is_symmetric, is_multigraph, is_weighted),
    offsets_(offsets),
    indices_(indices),
    weights_(weights),
    segment_offsets_(segment_offsets)
{
  // cheap error checks

  CUGRAPH_EXPECTS((is_weighted && (weights != nullptr)) || (!is_weighted && (weights == nullptr)),
                  "Invalid API parameter: weights shouldn't be nullptr if is_weighted is true and "
                  "should be nullptr if is_weighted is false.");

  CUGRAPH_EXPECTS(
    (sorted_by_degree &&
     (segment_offsets.size() == (detail::num_segments_per_vertex_partition + 1))) ||
      (!sorted_by_degree && (segment_offsets.size() == 0)),
    "Invalid API parameter: segment_offsets.size() does not match with sorted_by_degree.");

  // optinoal expensive checks

  if (do_expensive_check) {
    auto default_stream = this->get_handle_ptr()->get_stream();

    CUGRAPH_EXPECTS(thrust::is_sorted(rmm::exec_policy(default_stream)->on(default_stream),
                                      offsets,
                                      offsets + (this->get_number_of_vertices() + 1)),
                    "Invalid API parameter: offsets is not sorted.");

    // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
    CUGRAPH_EXPECTS(
      thrust::count_if(rmm::exec_policy(default_stream)->on(default_stream),
                       indices,
                       indices + this->get_number_of_edges(),
                       out_of_range_t<vertex_t>{0, this->get_number_of_vertices()}) == 0,
      "Invalid API parameter: adj_matrix_partition_indices[] have out-of-range vertex IDs.");

    if (sorted_by_degree) {
      auto degree_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(vertex_t{0}),
        degree_from_offsets_t<vertex_t, edge_t>{offsets});
      CUGRAPH_EXPECTS(thrust::is_sorted(rmm::exec_policy(default_stream)->on(default_stream),
                                        degree_first,
                                        degree_first + this->get_number_of_vertices(),
                                        thrust::greater<edge_t>{}),
                      "Invalid API parameter: sorted_by_degree is set to true, but degrees are not "
                      "non-ascending.");

      CUGRAPH_EXPECTS(std::is_sorted(segment_offsets.begin(), segment_offsets.end()),
                      "Invalid API parameter: erroneous segment_offsets.");
      CUGRAPH_EXPECTS(segment_offsets[0] == 0, "Invalid API parameter: segment_offsets.");
      CUGRAPH_EXPECTS(segment_offsets.back() == this->get_number_of_vertices(),
                      "Invalid API parameter: segment_offsets.");
    }

    if (is_symmetric) {}
    if (!is_multigraph) {}
  }
}

// explicit instantiation

template class graph_view_t<uint32_t, uint32_t, float, true, true>;
template class graph_view_t<uint32_t, uint32_t, float, false, true>;
template class graph_view_t<uint32_t, uint32_t, double, true, true>;
template class graph_view_t<uint32_t, uint32_t, double, false, true>;
template class graph_view_t<uint32_t, uint64_t, float, true, true>;
template class graph_view_t<uint32_t, uint64_t, float, false, true>;
template class graph_view_t<uint32_t, uint64_t, double, true, true>;
template class graph_view_t<uint32_t, uint64_t, double, false, true>;
template class graph_view_t<uint64_t, uint64_t, float, true, true>;
template class graph_view_t<uint64_t, uint64_t, float, false, true>;
template class graph_view_t<uint64_t, uint64_t, double, true, true>;
template class graph_view_t<uint64_t, uint64_t, double, false, true>;

template class graph_view_t<uint32_t, uint32_t, float, true, false>;
template class graph_view_t<uint32_t, uint32_t, float, false, false>;
template class graph_view_t<uint32_t, uint32_t, double, true, false>;
template class graph_view_t<uint32_t, uint32_t, double, false, false>;
template class graph_view_t<uint32_t, uint64_t, float, true, false>;
template class graph_view_t<uint32_t, uint64_t, float, false, false>;
template class graph_view_t<uint32_t, uint64_t, double, true, false>;
template class graph_view_t<uint32_t, uint64_t, double, false, false>;
template class graph_view_t<uint64_t, uint64_t, float, true, false>;
template class graph_view_t<uint64_t, uint64_t, float, false, false>;
template class graph_view_t<uint64_t, uint64_t, double, true, false>;
template class graph_view_t<uint64_t, uint64_t, double, false, false>;

}  // namespace experimental
}  // namespace cugraph