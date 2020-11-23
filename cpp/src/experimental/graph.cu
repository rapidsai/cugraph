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
#include <partition_manager.hpp>
#include <utilities/comm_utils.cuh>
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

#include <algorithm>
#include <tuple>

namespace cugraph {
namespace experimental {

namespace {

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_view_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t>
struct out_of_range_t {
  vertex_t major_first{};
  vertex_t major_last{};
  vertex_t minor_first{};
  vertex_t minor_last{};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> t)
  {
    auto major = thrust::get<0>(t);
    auto minor = thrust::get<1>(t);
    return (major < major_first) || (major >= major_last) || (minor < minor_first) ||
           (minor >= minor_last);
  }
};

template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
std::
  tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  edge_list_to_compressed_sparse(raft::handle_t const &handle,
                                 edgelist_t<vertex_t, edge_t, weight_t> const &edgelist,
                                 vertex_t major_first,
                                 vertex_t major_last,
                                 vertex_t minor_first,
                                 vertex_t minor_last)
{
  rmm::device_uvector<edge_t> offsets((major_last - major_first) + 1, handle.get_stream());
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

  thrust::for_each(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   store_transposed ? edgelist.p_dst_vertices : edgelist.p_src_vertices,
                   store_transposed ? edgelist.p_dst_vertices + edgelist.number_of_edges
                                    : edgelist.p_src_vertices + edgelist.number_of_edges,
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
                       auto major  = store_transposed ? d : s;
                       auto minor  = store_transposed ? s : d;
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
                       auto major  = store_transposed ? d : s;
                       auto minor  = store_transposed ? s : d;
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

  // FIXME: need to add an option to sort neighbor lists

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

  CUDA_TRY(cudaStreamSynchronize(
    handle.get_stream()));  // this is necessary as d_segment_offsets will become out-of-scope once
                            // this function returns and this function returns a host variable which
                            // can be used right after return.

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
          edge_t number_of_edges,
          graph_properties_t properties,
          bool sorted_by_global_degree_within_vertex_partition,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, number_of_vertices, number_of_edges, properties),
    partition_(partition)
{
  // cheap error checks

  auto &comm           = this->get_handle_ptr()->get_comms();
  auto const comm_size = comm.get_size();
  auto &row_comm =
    this->get_handle_ptr()->get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();
  auto &col_comm =
    this->get_handle_ptr()->get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();
  auto default_stream      = this->get_handle_ptr()->get_stream();

  CUGRAPH_EXPECTS(edgelists.size() > 0,
                  "Invalid API parameter: edgelists.size() should be non-zero.");

  bool is_weighted = edgelists[0].p_edge_weights != nullptr;

  CUGRAPH_EXPECTS(
    std::any_of(edgelists.begin() + 1,
                edgelists.end(),
                [is_weighted](auto edgelist) {
                  return (edgelist.p_src_vertices == nullptr) ||
                         (edgelist.p_dst_vertices == nullptr) ||
                         (is_weighted && (edgelist.p_edge_weights == nullptr)) ||
                         (!is_weighted && (edgelist.p_edge_weights != nullptr));
                }) == false,
    "Invalid API parameter: edgelists[].p_src_vertices and edgelists[].p_dst_vertices should not "
    "be nullptr and edgelists[].p_edge_weights should be nullptr (if edgelists[0].p_edge_weights "
    "is nullptr) or should not be nullptr (otherwise).");

  CUGRAPH_EXPECTS((partition.is_hypergraph_partitioned() &&
                   (edgelists.size() == static_cast<size_t>(col_comm_size))) ||
                    (!(partition.is_hypergraph_partitioned()) && (edgelists.size() == 1)),
                  "Invalid API parameter: errneous edgelists.size().");

  // optional expensive checks (part 1/3)

  if (do_expensive_check) {
    edge_t number_of_local_edges_sum{};
    for (size_t i = 0; i < edgelists.size(); ++i) {
      vertex_t major_first{};
      vertex_t major_last{};
      vertex_t minor_first{};
      vertex_t minor_last{};
      std::tie(major_first, major_last) = partition.get_matrix_partition_major_range(i);
      std::tie(minor_first, minor_last) = partition.get_matrix_partition_minor_range();

      number_of_local_edges_sum += edgelists[i].number_of_edges;

      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        store_transposed ? edgelists[i].p_dst_vertices : edgelists[i].p_src_vertices,
        store_transposed ? edgelists[i].p_src_vertices : edgelists[i].p_dst_vertices));
      // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
      CUGRAPH_EXPECTS(thrust::count_if(rmm::exec_policy(default_stream)->on(default_stream),
                                       edge_first,
                                       edge_first + edgelists[i].number_of_edges,
                                       out_of_range_t<vertex_t>{
                                         major_first, major_last, minor_first, minor_last}) == 0,
                      "Invalid API parameter: edgelists[] have out-of-range values.");
    }
    number_of_local_edges_sum =
      host_scalar_allreduce(comm, number_of_local_edges_sum, default_stream);
    CUGRAPH_EXPECTS(number_of_local_edges_sum == this->get_number_of_edges(),
                    "Invalid API parameter: the sum of local edges doe counts not match with "
                    "number_of_local_edges.");

    CUGRAPH_EXPECTS(
      partition.get_vertex_partition_last(comm_size - 1) == number_of_vertices,
      "Invalid API parameter: vertex partition should cover [0, number_of_vertices).");
  }

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  adj_matrix_partition_offsets_.reserve(edgelists.size());
  adj_matrix_partition_indices_.reserve(edgelists.size());
  adj_matrix_partition_weights_.reserve(is_weighted ? edgelists.size() : 0);
  for (size_t i = 0; i < edgelists.size(); ++i) {
    vertex_t major_first{};
    vertex_t major_last{};
    vertex_t minor_first{};
    vertex_t minor_last{};
    std::tie(major_first, major_last) = partition.get_matrix_partition_major_range(i);
    std::tie(minor_first, minor_last) = partition.get_matrix_partition_minor_range();

    rmm::device_uvector<edge_t> offsets(0, default_stream);
    rmm::device_uvector<vertex_t> indices(0, default_stream);
    rmm::device_uvector<weight_t> weights(0, default_stream);
    std::tie(offsets, indices, weights) = edge_list_to_compressed_sparse<store_transposed>(
      *(this->get_handle_ptr()), edgelists[i], major_first, major_last, minor_first, minor_last);
    adj_matrix_partition_offsets_.push_back(std::move(offsets));
    adj_matrix_partition_indices_.push_back(std::move(indices));
    if (is_weighted) { adj_matrix_partition_weights_.push_back(std::move(weights)); }
  }

  // update degree-based segment offsets (to be used for graph analytics kernel optimization)

  if (sorted_by_global_degree_within_vertex_partition) {
    auto degrees = detail::compute_major_degree(
      *(this->get_handle_ptr()), adj_matrix_partition_offsets_, partition_);

    // optional expensive checks (part 2/3)

    if (do_expensive_check) {
      CUGRAPH_EXPECTS(thrust::is_sorted(rmm::exec_policy(default_stream)->on(default_stream),
                                        degrees.begin(),
                                        degrees.end(),
                                        thrust::greater<edge_t>{}),
                      "Invalid API parameter: sorted_by_global_degree_within_vertex_partition is "
                      "set to true, but degrees are not non-ascending.");
    }

    static_assert(detail::num_segments_per_vertex_partition == 3);
    static_assert((detail::low_degree_threshold <= detail::mid_degree_threshold) &&
                  (detail::mid_degree_threshold <= std::numeric_limits<edge_t>::max()));
    rmm::device_uvector<edge_t> d_thresholds(detail::num_segments_per_vertex_partition - 1,
                                             default_stream);
    std::vector<edge_t> h_thresholds = {static_cast<edge_t>(detail::low_degree_threshold),
                                        static_cast<edge_t>(detail::mid_degree_threshold)};
    raft::update_device(
      d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), default_stream);

    rmm::device_uvector<vertex_t> segment_offsets(detail::num_segments_per_vertex_partition + 1,
                                                  default_stream);
    segment_offsets.set_element_async(0, 0, default_stream);
    segment_offsets.set_element_async(
      detail::num_segments_per_vertex_partition, degrees.size(), default_stream);

    thrust::upper_bound(rmm::exec_policy(default_stream)->on(default_stream),
                        degrees.begin(),
                        degrees.end(),
                        d_thresholds.begin(),
                        d_thresholds.end(),
                        segment_offsets.begin() + 1);

    rmm::device_uvector<vertex_t> aggregate_segment_offsets(0, default_stream);
    if (partition.is_hypergraph_partitioned()) {
      rmm::device_uvector<vertex_t> aggregate_segment_offsets(
        col_comm_size * segment_offsets.size(), default_stream);
      col_comm.allgather(segment_offsets.data(),
                         aggregate_segment_offsets.data(),
                         segment_offsets.size(),
                         default_stream);
    } else {
      rmm::device_uvector<vertex_t> aggregate_segment_offsets(
        row_comm_size * segment_offsets.size(), default_stream);
      row_comm.allgather(segment_offsets.data(),
                         aggregate_segment_offsets.data(),
                         segment_offsets.size(),
                         default_stream);
    }

    vertex_partition_segment_offsets_.resize(aggregate_segment_offsets.size());
    raft::update_host(vertex_partition_segment_offsets_.data(),
                      aggregate_segment_offsets.data(),
                      aggregate_segment_offsets.size(),
                      default_stream);

    raft::comms::status_t status{};
    if (partition.is_hypergraph_partitioned()) {
      status = col_comm.sync_stream(
        default_stream);  // this is necessary as degrees, d_thresholds, and segment_offsets will
                          // become out-of-scope once control flow exits this block and
                          // vertex_partition_segment_offsets_ can be used right after return.
    } else {
      status = row_comm.sync_stream(
        default_stream);  // this is necessary as degrees, d_thresholds, and segment_offsets will
                          // become out-of-scope once control flow exits this block and
                          // vertex_partition_segment_offsets_ can be used right after return.
    }
    CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  }

  // optional expensive checks (part 3/3)

  if (do_expensive_check) {
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
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  graph_t(raft::handle_t const &handle,
          edgelist_t<vertex_t, edge_t, weight_t> const &edgelist,
          vertex_t number_of_vertices,
          graph_properties_t properties,
          bool sorted_by_degree,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, number_of_vertices, edgelist.number_of_edges, properties),
    offsets_(rmm::device_uvector<edge_t>(0, handle.get_stream())),
    indices_(rmm::device_uvector<vertex_t>(0, handle.get_stream())),
    weights_(rmm::device_uvector<weight_t>(0, handle.get_stream()))
{
  // cheap error checks

  auto default_stream = this->get_handle_ptr()->get_stream();

  CUGRAPH_EXPECTS(
    (edgelist.p_src_vertices != nullptr) && (edgelist.p_dst_vertices != nullptr),
    "Invalid API parameter: edgelist.p_src_vertices and edgelist.p_dst_vertices should "
    "not be nullptr.");

  // optional expensive checks (part 1/2)

  if (do_expensive_check) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(store_transposed ? edgelist.p_dst_vertices : edgelist.p_src_vertices,
                         store_transposed ? edgelist.p_src_vertices : edgelist.p_dst_vertices));
    // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
    CUGRAPH_EXPECTS(thrust::count_if(
                      rmm::exec_policy(default_stream)->on(default_stream),
                      edge_first,
                      edge_first + edgelist.number_of_edges,
                      out_of_range_t<vertex_t>{
                        0, this->get_number_of_vertices(), 0, this->get_number_of_vertices()}) == 0,
                    "Invalid API parameter: edgelist have out-of-range values.");

    // FIXME: check for symmetricity may better be implemetned with transpose().
    if (this->is_symmetric()) {}
    // FIXME: check for duplicate edges may better be implemented after deciding whether to sort
    // neighbor list or not.
    if (!this->is_multigraph()) {}
  }

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  std::tie(offsets_, indices_, weights_) =
    edge_list_to_compressed_sparse<store_transposed>(*(this->get_handle_ptr()),
                                                     edgelist,
                                                     vertex_t{0},
                                                     this->get_number_of_vertices(),
                                                     vertex_t{0},
                                                     this->get_number_of_vertices());

  // update degree-based segment offsets (to be used for graph analytics kernel optimization)

  if (sorted_by_degree) {
    auto degree_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      detail::degree_from_offsets_t<vertex_t, edge_t>{offsets_.data()});

    // optional expensive checks (part 2/2)

    if (do_expensive_check) {
      CUGRAPH_EXPECTS(thrust::is_sorted(rmm::exec_policy(default_stream)->on(default_stream),
                                        degree_first,
                                        degree_first + this->get_number_of_vertices(),
                                        thrust::greater<edge_t>{}),
                      "Invalid API parameter: sorted_by_degree is set to true, but degrees are not "
                      "non-ascending.");
    }

    static_assert(detail::num_segments_per_vertex_partition == 3);
    static_assert((detail::low_degree_threshold <= detail::mid_degree_threshold) &&
                  (detail::mid_degree_threshold <= std::numeric_limits<edge_t>::max()));
    rmm::device_uvector<edge_t> d_thresholds(detail::num_segments_per_vertex_partition - 1,
                                             default_stream);
    std::vector<edge_t> h_thresholds = {static_cast<edge_t>(detail::low_degree_threshold),
                                        static_cast<edge_t>(detail::mid_degree_threshold)};
    raft::update_device(
      d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), default_stream);

    rmm::device_uvector<vertex_t> segment_offsets(detail::num_segments_per_vertex_partition + 1,
                                                  default_stream);
    segment_offsets.set_element_async(0, 0, default_stream);
    segment_offsets.set_element_async(
      detail::num_segments_per_vertex_partition, this->get_number_of_vertices(), default_stream);

    thrust::upper_bound(rmm::exec_policy(default_stream)->on(default_stream),
                        degree_first,
                        degree_first + this->get_number_of_vertices(),
                        d_thresholds.begin(),
                        d_thresholds.end(),
                        segment_offsets.begin() + 1);

    segment_offsets_.resize(segment_offsets.size());
    raft::update_host(
      segment_offsets_.data(), segment_offsets.data(), segment_offsets.size(), default_stream);

    CUDA_TRY(cudaStreamSynchronize(
      default_stream));  // this is necessary as d_thresholds and segment_offsets will become
                         // out-of-scpe once control flow exits this block and segment_offsets_ can
                         // be used right after return.
  }

  // optional expensive checks (part 3/3)

  if (do_expensive_check) {
    // FIXME: check for symmetricity may better be implemetned with transpose().
    if (this->is_symmetric()) {}
    // FIXME: check for duplicate edges may better be implemented after deciding whether to sort
    // neighbor list or not.
    if (!this->is_multigraph()) {}
  }
}

// explicit instantiation

template class graph_t<int32_t, int32_t, float, true, true>;
template class graph_t<int32_t, int32_t, float, false, true>;
template class graph_t<int32_t, int32_t, double, true, true>;
template class graph_t<int32_t, int32_t, double, false, true>;
template class graph_t<int32_t, int64_t, float, true, true>;
template class graph_t<int32_t, int64_t, float, false, true>;
template class graph_t<int32_t, int64_t, double, true, true>;
template class graph_t<int32_t, int64_t, double, false, true>;
template class graph_t<int64_t, int64_t, float, true, true>;
template class graph_t<int64_t, int64_t, float, false, true>;
template class graph_t<int64_t, int64_t, double, true, true>;
template class graph_t<int64_t, int64_t, double, false, true>;
//
template class graph_t<int32_t, int32_t, float, true, false>;
template class graph_t<int32_t, int32_t, float, false, false>;
template class graph_t<int32_t, int32_t, double, true, false>;
template class graph_t<int32_t, int32_t, double, false, false>;
template class graph_t<int32_t, int64_t, float, true, false>;
template class graph_t<int32_t, int64_t, float, false, false>;
template class graph_t<int32_t, int64_t, double, true, false>;
template class graph_t<int32_t, int64_t, double, false, false>;
template class graph_t<int64_t, int64_t, float, true, false>;
template class graph_t<int64_t, int64_t, float, false, false>;
template class graph_t<int64_t, int64_t, double, true, false>;
template class graph_t<int64_t, int64_t, double, false, false>;

}  // namespace experimental
}  // namespace cugraph

#include <experimental/eidir_graph.hpp>
