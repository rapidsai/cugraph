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
#include <experimental/graph.hpp>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace experimental {

namespace {

// FIXME: threshold values require tuning
size_t constexpr low_degree_threshold{raft::warp_size()};
size_t constexpr mid_degree_threshold{1024};
size_t constexpr num_segments_per_vertex_partition{3};

template <typename vertex_t, typename edge_t, typename weight_t>
edge_t sum_number_of_edges(std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const &edgelists)
{
  edge_t number_of_edges{0};
  for (size_t i = 0; i < edgelists.size(); ++i) { number_of_edges += edgelists[i].number_of_edges; }
  return number_of_edges;
}

template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
std::
  tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  edge_list_to_compressed_sparse(raft::handle_t const &handle,
                                 edgelist_t<vertex_t, edge_t, weight_t> const &edgelist,
                                 vertex_t row_first,
                                 vertex_t row_last,
                                 vertex_t col_first,
                                 vertex_t col_last)
{
  rmm::device_uvector<edge_t> offsets(
    store_transposed ? (row_last - row_first) + 1 : (col_last - col_first) + 1,
    handle.get_stream());
  rmm::device_uvector<vertex_t> indices(edgelist.number_of_edges, handle.get_stream());
  rmm::device_uvector<weight_t> weights(
    edgelist.p_edge_weights != nullptr ? edgelist.number_of_edges : 0, handle.get_stream());
  thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               offsets.begin(),
               offsets.end(),
               edge_t{0});
  thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               indices.begin(),
               indices.end(),
               vertex_t{0});

  // FIXME: need to performance test this code with R-mat graphs having highly-skewed degree
  // distribution. If there is a small number of vertices with very large degrees, atomicAdd can
  // sequentialize execution. CUDA9+ & Kepler+ provide complier/architectural optimizations to
  // mitigate this impact
  // (https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/),
  // and we need to check this thrust::for_each based approach delivers the expected performance.

  // FIXME: also need to verify this approach is at least not significantly slower than the sorting
  // based approach (this approach does not use extra memory, so better stick to this approach
  // unless performance is significantly worse).

  auto p_offsets = offsets.data();
  auto p_indices = indices.data();
  auto p_weights =
    edgelist.p_edge_weights != nullptr ? weights.data() : static_cast<weight_t *>(nullptr);

  auto major_first = store_transposed ? row_first : col_first;
  thrust::for_each(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   store_transposed ? edgelist.p_src_vertices : edgelist.p_dst_vertices,
                   store_transposed ? edgelist.p_src_vertices + edgelist.number_of_edges
                                    : edgelist.p_dst_vertices + edgelist.number_of_edges,
                   [p_offsets, major_first] __device__(auto v) {
                     atomicAdd(p_offsets + (v - major_first), edge_t{1});
                   });

  thrust::exclusive_scan(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                         offsets.begin(),
                         offsets.end(),
                         offsets.begin());

  if (edgelist.p_edge_weights != nullptr) {
    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      edgelist.p_src_vertices, edgelist.p_dst_vertices, edgelist.p_edge_weights));
    thrust::for_each(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     edge_first,
                     edge_first + edgelist.number_of_edges,
                     [p_offsets, p_indices, p_weights, major_first] __device__(auto e) {
                       auto s      = thrust::get<0>(e);
                       auto d      = thrust::get<1>(e);
                       auto w      = thrust::get<2>(e);
                       auto major  = store_transposed ? s : d;
                       auto minor  = store_transposed ? d : s;
                       auto start  = p_offsets[major - major_first];
                       auto degree = p_offsets[(major - major_first) + 1] - start;
                       auto idx    = atomicAdd(p_indices + (start + degree - 1),
                                            vertex_t{1});  // use the last element as a counter
                       // FIXME: we can actually store minor - minor_first instead of minor to save
                       // memory if minor can be larger than 32 bit but minor - minor_first fits
                       // within 32 bit
                       p_indices[start + idx] =
                         minor;  // overwrite the counter only if idx == degree - 1 (no race)
                       p_weights[start + idx] = w;
                     });
  } else {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist.p_src_vertices, edgelist.p_dst_vertices));
    thrust::for_each(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     edge_first,
                     edge_first + edgelist.number_of_edges,
                     [p_offsets, p_indices, p_weights, major_first] __device__(auto e) {
                       auto s      = thrust::get<0>(e);
                       auto d      = thrust::get<1>(e);
                       auto major  = store_transposed ? s : d;
                       auto minor  = store_transposed ? d : s;
                       auto start  = p_offsets[major - major_first];
                       auto degree = p_offsets[(major - major_first) + 1] - start;
                       auto idx    = atomicAdd(p_indices + (start + degree - 1),
                                            vertex_t{1});  // use the last element as a counter
                       // FIXME: we can actually store minor - minor_first instead of minor to save
                       // memory if minor can be larger than 32 bit but minor - minor_first fits
                       // within 32 bit
                       p_indices[start + idx] =
                         minor;  // overwrite the counter only if idx == degree - 1 (no race)
                     });
  }

  return std::make_tuple(std::move(offsets), std::move(indices), std::move(weights));
}

template <typename vertex_t, typename DegreeIterator, typename ThresholdIterator>
std::vector<vertex_t> segment_degree_sorted_vertex_partition(raft::handle_t const &handle,
                                                             DegreeIterator degree_first,
                                                             DegreeIterator degree_last,
                                                             ThresholdIterator threshold_first,
                                                             ThresholdIterator threshold_last)
{
  auto num_elements = thrust::distance(degree_first, degree_last);
  auto num_segments = thrust::distance(threshold_first, threshold_last) + 1;

  std::vector<vertex_t> h_segment_offsets(num_segments + 1);
  h_segment_offsets[0]     = 0;
  h_segment_offsets.back() = num_elements;

  rmm::device_uvector<vertex_t> d_segment_offsets(num_segments - 1, handle.get_stream());

  thrust::upper_bound(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      degree_first,
                      degree_last,
                      threshold_first,
                      threshold_last,
                      d_segment_offsets.begin());

  raft::update_host(h_segment_offsets.begin() + 1,
                    d_segment_offsets.begin(),
                    d_segment_offsets.size(),
                    handle.get_stream());

  return h_segment_offsets;
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_t(raft::handle_t const &handle,
          std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const &edgelists,
          partition_t<vertex_t> const &partition,
          vertex_t number_of_vertices,
          bool is_symmetric,
          bool is_multigraph,
          bool is_weighted,
          bool sorted_by_global_degree_within_vertex_partition,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(handle,
                                                     number_of_vertices,
                                                     sum_number_of_edges(edgelists),
                                                     is_symmetric,
                                                     is_multigraph,
                                                     is_weighted),
    partition_(partition)
{
  auto &comm_p_row     = this->get_handle_ptr()->get_subcomm(comm_p_row_key);
  auto comm_p_row_rank = comm_p_row.get_rank();
  auto comm_p_row_size = comm_p_row.get_size();
  auto &comm_p_col     = this->get_handle_ptr()->get_subcomm(comm_p_col_key);
  auto comm_p_col_rank = comm_p_col.get_rank();
  auto comm_p_col_size = comm_p_col.get_size();
  auto default_stream  = this->get_handle_ptr()->get_stream();

  // FIXME: error checks

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  adj_matrix_partition_offsets_.reserve(edgelists.size());
  adj_matrix_partition_indices_.reserve(edgelists.size());
  adj_matrix_partition_weights_.reserve(this->is_weighted() ? edgelists.size() : 0);
  for (size_t i = 0; i < edgelists.size(); ++i) {
    CUGRAPH_EXPECTS((is_weighted == false) || (edgelists[i].p_edge_weights != nullptr),
                    "Invalid API parameter, edgelists[i].p_edge_weights shoud not be nullptr if "
                    "is_weighted == true");

    auto row_first = partition_.hypergraph_partitioned
                       ? partition_.vertex_partition_offsets[comm_p_row_size * i + comm_p_row_rank]
                       : partition_.vertex_partition_offsets[comm_p_col_size * comm_p_row_rank];
    auto row_last =
      partition_.hypergraph_partitioned
        ? partition_.vertex_partition_offsets[comm_p_row_size * i + comm_p_row_rank + 1]
        : partition_.vertex_partition_offsets[comm_p_col_size * (comm_p_row_rank + 1)];
    auto col_first = partition_.vertex_partition_offsets[comm_p_row_size * comm_p_col_rank];
    auto col_last  = partition_.vertex_partition_offsets[comm_p_row_size * (comm_p_col_rank + 1)];

    rmm::device_uvector<edge_t> offsets(0, this->get_handle_ptr()->get_stream());
    rmm::device_uvector<vertex_t> indices(0, this->get_handle_ptr()->get_stream());
    rmm::device_uvector<weight_t> weights(0, this->get_handle_ptr()->get_stream());
    std::tie(offsets, indices, weights) = edge_list_to_compressed_sparse<store_transposed>(
      *(this->get_handle_ptr()), edgelists[i], row_first, row_last, col_first, col_last);
    adj_matrix_partition_offsets_.push_back(std::move(offsets));
    adj_matrix_partition_indices_.push_back(std::move(indices));
    if (this->is_weighted()) { adj_matrix_partition_weights_.push_back(std::move(weights)); }
  }

  // update degree-based segment offsets (to be used for graph analytics kernel optimization)

  auto degrees = detail::compute_major_degree(
    *(this->get_handle_ptr()), adj_matrix_partition_offsets_, partition_);

  static_assert(num_segments_per_vertex_partition == 3);
  static_assert((low_degree_threshold <= mid_degree_threshold) &&
                (mid_degree_threshold <= std::numeric_limits<edge_t>::max()));
  rmm::device_uvector<edge_t> d_thresholds(num_segments_per_vertex_partition - 1, default_stream);
  std::vector<edge_t> h_thresholds = {static_cast<edge_t>(low_degree_threshold),
                                      static_cast<edge_t>(mid_degree_threshold)};
  raft::update_device(
    d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), default_stream);

  rmm::device_uvector<vertex_t> segment_offsets(num_segments_per_vertex_partition + 1,
                                                default_stream);
  segment_offsets.set_element_async(0, 0, default_stream);
  segment_offsets.set_element_async(
    num_segments_per_vertex_partition, degrees.size(), default_stream);

  thrust::upper_bound(rmm::exec_policy(default_stream)->on(default_stream),
                      degrees.begin(),
                      degrees.end(),
                      d_thresholds.begin(),
                      d_thresholds.end(),
                      segment_offsets.begin() + 1);

  rmm::device_uvector<vertex_t> aggregate_segment_offsets(comm_p_row_size * segment_offsets.size(),
                                                          default_stream);
  comm_p_row.allgather(segment_offsets.data(),
                       aggregate_segment_offsets.data(),
                       segment_offsets.size(),
                       default_stream);

  vertex_partition_segment_offsets_.resize(comm_p_row_size * (segment_offsets.size()));
  raft::update_host(vertex_partition_segment_offsets_.data(),
                    aggregate_segment_offsets.data(),
                    aggregate_segment_offsets.size(),
                    default_stream);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  graph_t(raft::handle_t const &handle,
          edgelist_t<vertex_t, edge_t, weight_t> const &edgelist,
          vertex_t number_of_vertices,
          bool is_symmetric,
          bool is_multigraph,
          bool is_weighted,
          bool sorted_by_global_degree,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(handle,
                                                     number_of_vertices,
                                                     edgelist.number_of_edges,
                                                     is_symmetric,
                                                     is_multigraph,
                                                     is_weighted),
    offsets_(rmm::device_uvector<edge_t>(0, handle.get_stream())),
    indices_(rmm::device_uvector<vertex_t>(0, handle.get_stream())),
    weights_(rmm::device_uvector<weight_t>(0, handle.get_stream()))
{
  auto default_stream = this->get_handle_ptr()->get_stream();

  // FIXME: error checks

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  CUGRAPH_EXPECTS(
    (is_weighted == false) || (edgelist.p_edge_weights != nullptr),
    "Invalid API parameter, edgelist.p_edge_weights shoud not be nullptr if is_weighted == true");

  std::tie(offsets_, indices_, weights_) =
    edge_list_to_compressed_sparse<store_transposed>(*(this->get_handle_ptr()),
                                                     edgelist,
                                                     vertex_t{0},
                                                     this->get_number_of_vertices(),
                                                     vertex_t{0},
                                                     this->get_number_of_vertices());

  // update degree-based segment offsets (to be used for graph analytics kernel optimization)

  rmm::device_uvector<edge_t> degrees(this->get_number_of_vertices(), default_stream);
  thrust::adjacent_difference(rmm::exec_policy(default_stream)->on(default_stream),
                              offsets_.begin() + 1,
                              offsets_.end(),
                              degrees.begin());

  static_assert(num_segments_per_vertex_partition == 3);
  static_assert((low_degree_threshold <= mid_degree_threshold) &&
                (mid_degree_threshold <= std::numeric_limits<edge_t>::max()));
  rmm::device_uvector<edge_t> d_thresholds(num_segments_per_vertex_partition - 1, default_stream);
  std::vector<edge_t> h_thresholds = {static_cast<edge_t>(low_degree_threshold),
                                      static_cast<edge_t>(mid_degree_threshold)};
  raft::update_device(
    d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), default_stream);

  rmm::device_uvector<vertex_t> segment_offsets(num_segments_per_vertex_partition + 1,
                                                default_stream);
  segment_offsets.set_element_async(0, 0, default_stream);
  segment_offsets.set_element_async(
    num_segments_per_vertex_partition, degrees.size(), default_stream);

  thrust::upper_bound(rmm::exec_policy(default_stream)->on(default_stream),
                      degrees.begin(),
                      degrees.end(),
                      d_thresholds.begin(),
                      d_thresholds.end(),
                      segment_offsets.begin() + 1);

  raft::update_host(
    segment_offsets_.data(), segment_offsets.data(), segment_offsets.size(), default_stream);
}

// explicit instantiation

template class graph_t<int32_t, int32_t, float, true, true>;
template class graph_t<int32_t, int32_t, float, false, true>;

template class graph_t<int32_t, int32_t, float, true, false>;
template class graph_t<int32_t, int32_t, float, false, false>;

}  // namespace experimental
}  // namespace cugraph