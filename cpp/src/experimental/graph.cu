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

#include <exp_graph.hpp>

#include <utilities/error.hpp>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace experimental {

namespace {

template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
auto edge_list_to_compressed_sparse(raft::handel_t &handle,
                                    edgelist_t const &edgelist,
                                    vertex_t row_first,
                                    vertex_t row_last,
                                    vertex_t col_first,
                                    vertex_t col_last)
{
  rmm::device_uvector<edge_t> offsets(
    store_transposed ? (row_last - row_first) + 1 : (col_last - col_first) + 1,
    edge_t{0},
    handle.get_stream());
  rmm::device_uvector<vertex_t> indices(
    edge_list.number_of_edges, vertex_t{0}, handle.get_stream());
  rmm::device_uvector<weight_t> weights(
    edge_list.p_edge_weights != nullptr ? edge_list.number_of_edges : 0, handle.get_stream());
  thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               offsets.begin(),
               offsets.end(),
               edge_t{0});

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
    edge_list.p_edge_weights != nullptr ? weights.data() : reinterpret_cast<weight_t *>(nullptr);

  auto major_first = store_transposed ? row_first : col_first;
  thrust::for_each(
    rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
    store_transposed ? edgelist.p_src_vertices : edgelist.p_dst_vertices,
    store_transposed ? edgelist.p_src_vertices + edgelist.number_of_edges
                     : edgelist.p_dst_vertices + edgelist.number_of_edges,
    [p_offsets, major_first] __device__(auto v) { atomicAdd(p_offsets + (v - major_first), 1); });

  thrust::exclusive_scan(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                         offsets.begin(),
                         offsets.end(),
                         offsets.begin());

  if (edgelist.edge_weights != nullptr) {
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
                       auto idx    = atmoicAdd(p_indices + (degree - 1),
                                            1);  // use the last element as a counter
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
                       auto idx    = atmoicAdd(p_indices + (degree - 1),
                                            1);  // use the last element as a counter
                       // FIXME: we can actually store minor - minor_first instead of minor to save
                       // memory if minor can be larger than 32 bit but minor - minor_first fits
                       // within 32 bit
                       p_indices[start + idx] =
                         minor;  // overwrite the counter only if idx == degree - 1 (no race)
                     });
  }

  return std::make_tuple(offsets, indices, weights);
}

// compute the numbers of nonzeros in rows of the (transposed) graph adjacency matrix
rmm::device_uvector<edge_t> compute_row_degree(
  raft::handle_t &handle,
  std::vector<rmm::device_uvector<edge_t>> const &adj_matrix_partition_offsets{};
  bool hypergraph_partitioned)
{
  auto &comm_p_row     = handle.get_subcomm(comm_p_row_key);
  auto comm_p_row_rank = comm_p_row.get_rank();
  auto comm_p_row_size = comm_p_row.get_size();
  auto &comm_p_col     = handle.get_subcomm(comm_p_col_key);
  auto comm_p_col_rank = comm_p_col.get_rank();
  auto comm_p_col_size = comm_p_col.get_size();

  rmm::device_uvector<edge_t> local_degrees{};
  rmm::device_uvector<edge_t> degrees{};

  vertex_t max_num_local_degrees{0};
  for (int i = 0; i < comm_p_col_size; ++i) {
    auto vertex_partition_id = hypergraph_partitioned ? comm_p_row_size * i + row_rank
                                                      : comm_p_col_size * comm_p_row_rank + i;
    auto row_first        = vertex_partition_offsets[vertex_partition_id];
    auto row_last         = vertex_partition_offsets[vertex_partition_id + 1];
    max_num_local_degrees = std::max(max_num_local_degrees, row_last - row_first);
    if (i == comm_p_col_rank) { degrees.resize(row_last - row_first, handle.get_stream()); }
  }
  local_degrees.reisze(local_degree_size, handle.get_stream());
  for (int i = 0; i < comm_p_col_size; ++i) {
    auto vertex_partition_id = hypergraph_partitioned ? comm_p_row_size * i + row_rank
                                                      : comm_p_col_size * comm_p_row_rank + i;
    auto row_first = vertex_partition_offsets[vertex_partition_id];
    auto row_last  = vertex_partition_offsets[vertex_partition_id + 1];
    auto p_offsets =
      hypergraph_partitioned
        ? adj_matrix_partition_offsets_[i].data()
        : adj_matrix_partition_offsets_[0].data() +
            (row_first - vertex_partition_offsets[comm_p_col_size * comm_p_row_rank]);
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(row_last - row_first),
                      local_degrees.data(),
                      [p_offsets] __device__(auto i) { return p_offsets[i + 1] - p_offsets[i]; });
    comm_p_row.reduce(local_degrees.data(),
                      i == comm_p_col_rank ? degrees.data() : nullptr,
                      degrees.size(),
                      raft::comms::get_type<edge_t>(),
                      hanlde.get_stream());
  }

  return degrees;
}

template <typename vertex_t, typename DegreeIterator, typename ThresholdIterator>
std::vector<vertex_t> segment_degree_sorted_vertex_partition(raft::handle_t &handle,
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
                      d_segments_offsets.begin());

  raft::update_host(h_segment_offsets.begin() + 1,
                    d_segment_offsets.begin(),
                    d_segment_offsets.size(),
                    handle.get_stream());

  return h_segment_offsets;
}

}  // namespace

}  // namespace experimental

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_t(raft::handel_t &handle,
          std::vector<edgelist_t<vertex_t, edge_t>> const &edge_lists,
          vertex_partition_t<vertex_t> const &vertex_partition,
          std::vector<adj_matrix_partition_t<vertex_t>> const &adj_matrix_partitions,
          vertex_t number_of_vertices,
          edge_t number_of_edges,
          bool is_symmetric,
          bool is_multigraph,
          bool is_weighted,
          bool do_expensive_check)
  : number_of_vertices_(number_of_vertices),
    number_of_edges_(number_of_edges),
    properties_({is_symmetric, is_multigraph, is_weighted}),
    handle_ptr_(&handle),
    vertex_partition_(vertex_partition),
    adj_matrix_partitions_(adj_matrix_partitions)
{
  CUGRAPH_EXPECTS(edge_lists.size() == adj_matrix_partitions.size(),
                  "Invalid API parameter, edge_lists.size() != adj_matrix_partitions.size()");

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  adj_matrix_partition_offsets_.resize(edge_lists.size());
  adj_matrix_partition_indices_.resize(edge_lists.size());
  adj_matrix_partition_weights_.resize(edge_lists.size());
  for (size_t i = 0; i < edge_lists.size(); ++i) {
    CUGRAPH_EXPECTS((is_weighted == false) || (edge_lists[i].p_edge_weights != nullptr),
                    "Invalid API parameter, edge_lists[i].p_edge_weights shoud not be nullptr if "
                    "is_weighted == true");

    rmm::device_uvector<edge_t> offets{};
    rmm::device_uvector<vertex_t> indices{};
    rmm::device_uvector<weight_t> weights{};

    std::tie(offsets, indices, weights) = detail::edge_list_to_compressed_sparse(
      handle,
      edge_lists[i],
      vertex_partition_.partition_offsets[adj_matrix_partitions[i].row_vertex_partition_first],
      vertex_partition_.partition_offsets[adj_matrix_partitions[i].row_vertex_partition_last],
      vertex_partition_.partition_offsets[adj_matrix_partitions[i].col_vertex_partition_first],
      vertex_partition_.partition_offsets[adj_matrix_partitions[i].col_vertex_partition_last]);
    adj_matrix_partitions_offsets_[i] = std::move(offsets);
    adj_matrix_partitions_indices_[i] = std::move(indices);
    if (is_weighted) { adj_matrix_partitions_weights_[i] = std::move(weights); }
  }

  // update degree-based segment offsets (to be used for graph analytics kernel optimization)

  auto &comm_p_row     = handle_ptr_->get_subcomm(comm_p_row_key);
  auto comm_p_row_rank = comm_p_row.get_rank();
  auto comm_p_row_size = comm_p_row.get_size();

  auto degrees = compute_row_degree<store_row_major, multi_gpu>();

  static_assert(detail::num_segments_per_subpartition == 3);
  static_assert((detail::low_degree_threshold <= detail::high_degree_threshold) &&
                (detail::high_degree_threshold <= std::numeric_limits<vertex_t>::max()));
  auto segment_degree_threshold_first =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), [] __device__(auto i) {
      if (i == 0) {
        return static_cast<vertex_t>(detail::low_degree_threshold);
      } else if (i == 1) {
        return static_cast<vertex_t>(detail::high_degree_threshold);
      } else {
        assert(0);  // should not be reached
        return vertex_t{0};
      }
    });

  auto segment_offsets = segment_degree_sorted_vertex_partition(
    *handle_ptr_,
    degrees.begin(),
    degrees.end(),
    segment_degree_threshold_first,
    segment_degree_threshold_first + num_segments_per_subpartition - 1);

  comm_p_row.allgather(segment_offsets.begin(),
                       aggregate_segment_offsets_.begin(),
                       (num_segments_per_partition + 1),
                       raft::comms::get_type<vertex_t>(),
                       hanlde_ptr_->get_stream());
}

// explicit instantiation

template class graph_t<int32_t, int32_t, float, true, true>;
template class graph_t<int32_t, int32_t, float, false, true>;

template class graph_t<int32_t, int32_t, float, true, false>;
template class graph_t<int32_t, int32_t, float, false, false>;

}  // namespace cugraph
}  // namespace cugraph