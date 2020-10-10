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
#include <partition_manager.hpp>
#include <utilities/comm_utils.cuh>
#include <utilities/error.hpp>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
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
               graph_properties_t properties,
               bool sorted_by_global_degree_within_vertex_partition,
               bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, number_of_vertices, number_of_edges, properties),
    adj_matrix_partition_offsets_(adj_matrix_partition_offsets),
    adj_matrix_partition_indices_(adj_matrix_partition_indices),
    adj_matrix_partition_weights_(adj_matrix_partition_weights),
    partition_(partition),
    vertex_partition_segment_offsets_(vertex_partition_segment_offsets)
{
  // cheap error checks

  auto const comm_size     = this->get_handle_ptr()->get_comms().get_size();
  auto const row_comm_size = this->get_handle_ptr()
                               ->get_subcomm(cugraph::partition_2d::key_naming_t().row_name())
                               .get_size();
  auto const col_comm_size = this->get_handle_ptr()
                               ->get_subcomm(cugraph::partition_2d::key_naming_t().col_name())
                               .get_size();

  CUGRAPH_EXPECTS(adj_matrix_partition_offsets.size() == adj_matrix_partition_indices.size(),
                  "Invalid API parameter: adj_matrix_partition_offsets.size() and "
                  "adj_matrix_partition_indices.size() should coincide.");
  CUGRAPH_EXPECTS((adj_matrix_partition_weights.size() == adj_matrix_partition_offsets.size()) ||
                    (adj_matrix_partition_weights.size() == 0),
                  "Invalid API parameter: adj_matrix_partition_weights.size() should coincide with "
                  "adj_matrix_partition_offsets.size() (if weighted) or 0 (if unweighted).");

  CUGRAPH_EXPECTS(
    (partition.is_hypergraph_partitioned() &&
     (adj_matrix_partition_offsets.size() == static_cast<size_t>(row_comm_size))) ||
      (!(partition.is_hypergraph_partitioned()) && (adj_matrix_partition_offsets.size() == 1)),
    "Invalid API parameter: errneous adj_matrix_partition_offsets.size().");

  CUGRAPH_EXPECTS((sorted_by_global_degree_within_vertex_partition &&
                   (vertex_partition_segment_offsets.size() ==
                    (partition.is_hypergraph_partitioned() ? col_comm_size : row_comm_size) *
                      (detail::num_segments_per_vertex_partition + 1))) ||
                    (!sorted_by_global_degree_within_vertex_partition &&
                     (vertex_partition_segment_offsets.size() == 0)),
                  "Invalid API parameter: vertex_partition_segment_offsets.size() does not match "
                  "with sorted_by_global_degree_within_vertex_partition.");

  // optional expensive checks

  if (do_expensive_check) {
    auto default_stream = this->get_handle_ptr()->get_stream();

    auto const row_comm_rank = this->get_handle_ptr()
                                 ->get_subcomm(cugraph::partition_2d::key_naming_t().row_name())
                                 .get_rank();
    auto const col_comm_rank = this->get_handle_ptr()
                                 ->get_subcomm(cugraph::partition_2d::key_naming_t().col_name())
                                 .get_rank();

    edge_t number_of_local_edges_sum{};
    for (size_t i = 0; i < adj_matrix_partition_offsets.size(); ++i) {
      vertex_t major_first{};
      vertex_t major_last{};
      vertex_t minor_first{};
      vertex_t minor_last{};
      std::tie(major_first, major_last) = partition.get_matrix_partition_major_range(i);
      std::tie(minor_first, minor_last) = partition.get_matrix_partition_minor_range();
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
      CUDA_TRY(cudaStreamSynchronize(default_stream));
      number_of_local_edges_sum += number_of_local_edges;

      // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
      CUGRAPH_EXPECTS(
        thrust::count_if(rmm::exec_policy(default_stream)->on(default_stream),
                         adj_matrix_partition_indices[i],
                         adj_matrix_partition_indices[i] + number_of_local_edges,
                         out_of_range_t<vertex_t>{minor_first, minor_last}) == 0,
        "Invalid API parameter: adj_matrix_partition_indices[] have out-of-range vertex IDs.");
    }
    number_of_local_edges_sum = host_scalar_allreduce(
      this->get_handle_ptr()->get_comms(), number_of_local_edges_sum, default_stream);
    CUGRAPH_EXPECTS(number_of_local_edges_sum == this->get_number_of_edges(),
                    "Invalid API parameter: the sum of local edges doe counts not match with "
                    "number_of_local_edges.");

    if (sorted_by_global_degree_within_vertex_partition) {
      auto degrees = detail::compute_major_degree(handle, adj_matrix_partition_offsets, partition);
      CUGRAPH_EXPECTS(thrust::is_sorted(rmm::exec_policy(default_stream)->on(default_stream),
                                        degrees.begin(),
                                        degrees.end(),
                                        thrust::greater<edge_t>{}),
                      "Invalid API parameter: sorted_by_global_degree_within_vertex_partition is "
                      "set to true, but degrees are not non-ascending.");

      for (int i = 0; i < (partition.is_hypergraph_partitioned() ? col_comm_size : row_comm_size);
           ++i) {
        CUGRAPH_EXPECTS(std::is_sorted(vertex_partition_segment_offsets.begin() +
                                         (detail::num_segments_per_vertex_partition + 1) * i,
                                       vertex_partition_segment_offsets.begin() +
                                         (detail::num_segments_per_vertex_partition + 1) * (i + 1)),
                        "Invalid API parameter: erroneous vertex_partition_segment_offsets.");
        CUGRAPH_EXPECTS(
          vertex_partition_segment_offsets[(detail::num_segments_per_vertex_partition + 1) * i] ==
            0,
          "Invalid API parameter: erroneous vertex_partition_segment_offsets.");
        auto vertex_partition_idx = partition.is_hypergraph_partitioned()
                                      ? row_comm_size * i + row_comm_rank
                                      : col_comm_rank * row_comm_size + i;
        CUGRAPH_EXPECTS(
          vertex_partition_segment_offsets[(detail::num_segments_per_vertex_partition + 1) * i +
                                           detail::num_segments_per_vertex_partition] ==
            partition.get_vertex_partition_size(vertex_partition_idx),
          "Invalid API parameter: erroneous vertex_partition_segment_offsets.");
      }
    }

    CUGRAPH_EXPECTS(
      partition.get_vertex_partition_last(comm_size - 1) == number_of_vertices,
      "Invalid API parameter: vertex partition should cover [0, number_of_vertices).");

    // FIXME: check for symmetricity may better be implemetned with transpose().
    if (this->is_symmetric()) {}
    // FIXME: check for duplicate edges may better be implemented after deciding whether to sort
    // neighbor list or not.
    if (!this->is_multigraph()) {}
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
                                                         graph_properties_t properties,
                                                         bool sorted_by_degree,
                                                         bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, number_of_vertices, number_of_edges, properties),
    offsets_(offsets),
    indices_(indices),
    weights_(weights),
    segment_offsets_(segment_offsets)
{
  // cheap error checks

  CUGRAPH_EXPECTS(
    (sorted_by_degree &&
     (segment_offsets.size() == (detail::num_segments_per_vertex_partition + 1))) ||
      (!sorted_by_degree && (segment_offsets.size() == 0)),
    "Invalid API parameter: segment_offsets.size() does not match with sorted_by_degree.");

  // optional expensive checks

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
      auto degree_first =
        thrust::make_transform_iterator(thrust::make_counting_iterator(vertex_t{0}),
                                        detail::degree_from_offsets_t<vertex_t, edge_t>{offsets});
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

    // FIXME: check for symmetricity may better be implemetned with transpose().
    if (this->is_symmetric()) {}
    // FIXME: check for duplicate edges may better be implemented after deciding whether to sort
    // neighbor list or not.
    if (!this->is_multigraph()) {}
  }
}

// explicit instantiation

template class graph_view_t<int32_t, int32_t, float, true, true>;
template class graph_view_t<int32_t, int32_t, float, false, true>;
template class graph_view_t<int32_t, int32_t, double, true, true>;
template class graph_view_t<int32_t, int32_t, double, false, true>;
template class graph_view_t<int32_t, int64_t, float, true, true>;
template class graph_view_t<int32_t, int64_t, float, false, true>;
template class graph_view_t<int32_t, int64_t, double, true, true>;
template class graph_view_t<int32_t, int64_t, double, false, true>;
template class graph_view_t<int64_t, int64_t, float, true, true>;
template class graph_view_t<int64_t, int64_t, float, false, true>;
template class graph_view_t<int64_t, int64_t, double, true, true>;
template class graph_view_t<int64_t, int64_t, double, false, true>;
template class graph_view_t<int64_t, int32_t, float, true, true>;
template class graph_view_t<int64_t, int32_t, float, false, true>;
template class graph_view_t<int64_t, int32_t, double, true, true>;
template class graph_view_t<int64_t, int32_t, double, false, true>;

template class graph_view_t<int32_t, int32_t, float, true, false>;
template class graph_view_t<int32_t, int32_t, float, false, false>;
template class graph_view_t<int32_t, int32_t, double, true, false>;
template class graph_view_t<int32_t, int32_t, double, false, false>;
template class graph_view_t<int32_t, int64_t, float, true, false>;
template class graph_view_t<int32_t, int64_t, float, false, false>;
template class graph_view_t<int32_t, int64_t, double, true, false>;
template class graph_view_t<int32_t, int64_t, double, false, false>;
template class graph_view_t<int64_t, int64_t, float, true, false>;
template class graph_view_t<int64_t, int64_t, float, false, false>;
template class graph_view_t<int64_t, int64_t, double, true, false>;
template class graph_view_t<int64_t, int64_t, double, false, false>;
template class graph_view_t<int64_t, int32_t, float, true, false>;
template class graph_view_t<int64_t, int32_t, float, false, false>;
template class graph_view_t<int64_t, int32_t, double, true, false>;
template class graph_view_t<int64_t, int32_t, double, false, false>;

}  // namespace experimental
}  // namespace cugraph
