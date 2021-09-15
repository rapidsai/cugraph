/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#pragma once

#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

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

// compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
std::tuple<rmm::device_uvector<edge_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
compress_edgelist(edgelist_t<vertex_t, edge_t, weight_t> const& edgelist,
                  vertex_t major_first,
                  std::optional<vertex_t> major_hypersparse_first,
                  vertex_t major_last,
                  vertex_t /* minor_first */,
                  vertex_t /* minor_last */,
                  rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<edge_t> offsets((major_last - major_first) + 1, stream_view);
  rmm::device_uvector<vertex_t> indices(edgelist.number_of_edges, stream_view);
  auto weights = edgelist.p_edge_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                             edgelist.number_of_edges, stream_view)
                                         : std::nullopt;
  thrust::fill(rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), edge_t{0});
  thrust::fill(rmm::exec_policy(stream_view), indices.begin(), indices.end(), vertex_t{0});

  auto p_offsets = offsets.data();
  thrust::for_each(rmm::exec_policy(stream_view),
                   store_transposed ? edgelist.p_dst_vertices : edgelist.p_src_vertices,
                   store_transposed ? edgelist.p_dst_vertices + edgelist.number_of_edges
                                    : edgelist.p_src_vertices + edgelist.number_of_edges,
                   [p_offsets, major_first] __device__(auto v) {
                     atomicAdd(p_offsets + (v - major_first), edge_t{1});
                   });
  thrust::exclusive_scan(
    rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), offsets.begin());

  auto p_indices = indices.data();
  if (edgelist.p_edge_weights) {
    auto p_weights = (*weights).data();

    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      edgelist.p_src_vertices, edgelist.p_dst_vertices, *(edgelist.p_edge_weights)));
    thrust::for_each(rmm::exec_policy(stream_view),
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
    thrust::for_each(rmm::exec_policy(stream_view),
                     edge_first,
                     edge_first + edgelist.number_of_edges,
                     [p_offsets, p_indices, major_first] __device__(auto e) {
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

  auto dcs_nzd_vertices = major_hypersparse_first
                            ? std::make_optional<rmm::device_uvector<vertex_t>>(
                                major_last - *major_hypersparse_first, stream_view)
                            : std::nullopt;
  if (dcs_nzd_vertices) {
    auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

    thrust::transform(
      rmm::exec_policy(stream_view),
      thrust::make_counting_iterator(*major_hypersparse_first),
      thrust::make_counting_iterator(major_last),
      (*dcs_nzd_vertices).begin(),
      [major_first, offsets = offsets.data()] __device__(auto major) {
        auto major_offset = major - major_first;
        return offsets[major_offset + 1] - offsets[major_offset] > 0 ? major : invalid_vertex;
      });

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      (*dcs_nzd_vertices).begin(), offsets.begin() + (*major_hypersparse_first - major_first)));
    (*dcs_nzd_vertices)
      .resize(thrust::distance(pair_first,
                               thrust::remove_if(rmm::exec_policy(stream_view),
                                                 pair_first,
                                                 pair_first + (*dcs_nzd_vertices).size(),
                                                 [] __device__(auto pair) {
                                                   return thrust::get<0>(pair) == invalid_vertex;
                                                 })),
              stream_view);
    (*dcs_nzd_vertices).shrink_to_fit(stream_view);
    if (static_cast<vertex_t>((*dcs_nzd_vertices).size()) < major_last - *major_hypersparse_first) {
      thrust::copy(
        rmm::exec_policy(stream_view),
        offsets.begin() + (major_last - major_first),
        offsets.end(),
        offsets.begin() + (*major_hypersparse_first - major_first) + (*dcs_nzd_vertices).size());
      offsets.resize((*major_hypersparse_first - major_first) + (*dcs_nzd_vertices).size() + 1,
                     stream_view);
      offsets.shrink_to_fit(stream_view);
    }
  }

  // FIXME: need to add an option to sort neighbor lists

  return std::make_tuple(
    std::move(offsets), std::move(indices), std::move(weights), std::move(dcs_nzd_vertices));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_t(raft::handle_t const& handle,
          std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const& edgelists,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, meta.number_of_edges, meta.properties),
    partition_(meta.partition)
{
  // cheap error checks

  auto& comm           = this->get_handle_ptr()->get_comms();
  auto const comm_size = comm.get_size();
  auto& row_comm =
    this->get_handle_ptr()->get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm =
    this->get_handle_ptr()->get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();
  auto default_stream_view = this->get_handle_ptr()->get_stream_view();

  CUGRAPH_EXPECTS(edgelists.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input argument: errneous edgelists.size().");
  CUGRAPH_EXPECTS(
    !(meta.segment_offsets).has_value() ||
      ((*(meta.segment_offsets)).size() ==
       (detail::num_sparse_segments_per_vertex_partition + 1)) ||
      ((*(meta.segment_offsets)).size() == (detail::num_sparse_segments_per_vertex_partition + 2)),
    "Invalid input argument: (*(meta.segment_offsets)).size() returns an invalid value.");

  auto is_weighted = edgelists[0].p_edge_weights.has_value();
  auto use_dcs =
    meta.segment_offsets
      ? ((*(meta.segment_offsets)).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
      : false;

  CUGRAPH_EXPECTS(
    std::any_of(edgelists.begin(),
                edgelists.end(),
                [is_weighted](auto edgelist) {
                  return ((edgelist.number_of_edges > 0) && (edgelist.p_src_vertices == nullptr)) ||
                         ((edgelist.number_of_edges > 0) && (edgelist.p_dst_vertices == nullptr)) ||
                         (is_weighted && (edgelist.number_of_edges > 0) &&
                          ((edgelist.p_edge_weights.has_value() == false) ||
                           (*(edgelist.p_edge_weights) == nullptr)));
                }) == false,
    "Invalid input argument: edgelists[].p_src_vertices and edgelists[].p_dst_vertices should not "
    "be nullptr if edgelists[].number_of_edges > 0 and edgelists[].p_edge_weights should be "
    "neither std::nullopt nor nullptr if weighted and edgelists[].number_of_edges >  0.");

  // optional expensive checks (part 1/2)

  if (do_expensive_check) {
    edge_t number_of_local_edges_sum{};
    for (size_t i = 0; i < edgelists.size(); ++i) {
      auto [major_first, major_last] = partition_.get_matrix_partition_major_range(i);
      auto [minor_first, minor_last] = partition_.get_matrix_partition_minor_range();

      number_of_local_edges_sum += edgelists[i].number_of_edges;

      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        store_transposed ? edgelists[i].p_dst_vertices : edgelists[i].p_src_vertices,
        store_transposed ? edgelists[i].p_src_vertices : edgelists[i].p_dst_vertices));
      // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
      CUGRAPH_EXPECTS(thrust::count_if(rmm::exec_policy(default_stream_view),
                                       edge_first,
                                       edge_first + edgelists[i].number_of_edges,
                                       out_of_range_t<vertex_t>{
                                         major_first, major_last, minor_first, minor_last}) == 0,
                      "Invalid input argument: edgelists[] have out-of-range values.");
    }
    number_of_local_edges_sum =
      host_scalar_allreduce(comm, number_of_local_edges_sum, default_stream_view.value());
    CUGRAPH_EXPECTS(number_of_local_edges_sum == this->get_number_of_edges(),
                    "Invalid input argument: the sum of local edge counts does not match with "
                    "meta.number_of_edges.");

    CUGRAPH_EXPECTS(
      partition_.get_vertex_partition_last(comm_size - 1) == meta.number_of_vertices,
      "Invalid input argument: vertex partition should cover [0, meta.number_of_vertices).");
  }

  // aggregate segment_offsets

  if (meta.segment_offsets) {
    // FIXME: we need to add host_allgather
    rmm::device_uvector<vertex_t> d_segment_offsets((*(meta.segment_offsets)).size(),
                                                    default_stream_view);
    raft::update_device(d_segment_offsets.data(),
                        (*(meta.segment_offsets)).data(),
                        (*(meta.segment_offsets)).size(),
                        default_stream_view.value());
    rmm::device_uvector<vertex_t> d_aggregate_segment_offsets(
      col_comm_size * d_segment_offsets.size(), default_stream_view);
    col_comm.allgather(d_segment_offsets.data(),
                       d_aggregate_segment_offsets.data(),
                       d_segment_offsets.size(),
                       default_stream_view.value());

    adj_matrix_partition_segment_offsets_ =
      std::vector<vertex_t>(d_aggregate_segment_offsets.size(), vertex_t{0});
    raft::update_host((*adj_matrix_partition_segment_offsets_).data(),
                      d_aggregate_segment_offsets.data(),
                      d_aggregate_segment_offsets.size(),
                      default_stream_view.value());

    default_stream_view
      .synchronize();  // this is necessary as adj_matrix_partition_segment_offsets_ can be used
                       // right after return.
  }

  // compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid

  adj_matrix_partition_offsets_.reserve(edgelists.size());
  adj_matrix_partition_indices_.reserve(edgelists.size());
  if (is_weighted) {
    adj_matrix_partition_weights_ = std::vector<rmm::device_uvector<weight_t>>{};
    (*adj_matrix_partition_weights_).reserve(edgelists.size());
  }
  if (use_dcs) {
    adj_matrix_partition_dcs_nzd_vertices_      = std::vector<rmm::device_uvector<vertex_t>>{};
    adj_matrix_partition_dcs_nzd_vertex_counts_ = std::vector<vertex_t>{};
    (*adj_matrix_partition_dcs_nzd_vertices_).reserve(edgelists.size());
    (*adj_matrix_partition_dcs_nzd_vertex_counts_).reserve(edgelists.size());
  }
  for (size_t i = 0; i < edgelists.size(); ++i) {
    auto [major_first, major_last] = partition_.get_matrix_partition_major_range(i);
    auto [minor_first, minor_last] = partition_.get_matrix_partition_minor_range();
    auto major_hypersparse_first =
      use_dcs ? std::optional<vertex_t>{major_first +
                                        (*adj_matrix_partition_segment_offsets_)
                                          [(*(meta.segment_offsets)).size() * i +
                                           detail::num_sparse_segments_per_vertex_partition]}
              : std::nullopt;
    auto [offsets, indices, weights, dcs_nzd_vertices] =
      compress_edgelist<store_transposed>(edgelists[i],
                                          major_first,
                                          major_hypersparse_first,
                                          major_last,
                                          minor_first,
                                          minor_last,
                                          default_stream_view);

    adj_matrix_partition_offsets_.push_back(std::move(offsets));
    adj_matrix_partition_indices_.push_back(std::move(indices));
    if (is_weighted) { (*adj_matrix_partition_weights_).push_back(std::move(*weights)); }
    if (use_dcs) {
      auto dcs_nzd_vertex_count = static_cast<vertex_t>((*dcs_nzd_vertices).size());
      (*adj_matrix_partition_dcs_nzd_vertices_).push_back(std::move(*dcs_nzd_vertices));
      (*adj_matrix_partition_dcs_nzd_vertex_counts_).push_back(dcs_nzd_vertex_count);
    }
  }

  // optional expensive checks (part 2/2)

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
  graph_t(raft::handle_t const& handle,
          edgelist_t<vertex_t, edge_t, weight_t> const& edgelist,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, edgelist.number_of_edges, meta.properties),
    offsets_(rmm::device_uvector<edge_t>(0, handle.get_stream_view())),
    indices_(rmm::device_uvector<vertex_t>(0, handle.get_stream_view())),
    segment_offsets_(meta.segment_offsets)
{
  // cheap error checks

  auto default_stream_view = this->get_handle_ptr()->get_stream_view();

  auto is_weighted = edgelist.p_edge_weights.has_value();

  CUGRAPH_EXPECTS(
    ((edgelist.number_of_edges == 0) || (edgelist.p_src_vertices != nullptr)) &&
      ((edgelist.number_of_edges == 0) || (edgelist.p_dst_vertices != nullptr)) &&
      (!is_weighted || (is_weighted && ((edgelist.number_of_edges == 0) ||
                                        (*(edgelist.p_edge_weights) != nullptr)))),
    "Invalid input argument: edgelist.p_src_vertices and edgelist.p_dst_vertices should not be "
    "nullptr if edgelist.number_of_edges > 0 and edgelist.p_edge_weights should be neither "
    "std::nullopt nor nullptr if weighted and edgelist.number_of_edges > 0.");

  CUGRAPH_EXPECTS(
    !segment_offsets_.has_value() ||
      ((*segment_offsets_).size() == (detail::num_sparse_segments_per_vertex_partition + 1)),
    "Invalid input argument: (*(meta.segment_offsets)).size() returns an invalid value.");

  // optional expensive checks (part 1/2)

  if (do_expensive_check) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(store_transposed ? edgelist.p_dst_vertices : edgelist.p_src_vertices,
                         store_transposed ? edgelist.p_src_vertices : edgelist.p_dst_vertices));
    // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
    CUGRAPH_EXPECTS(thrust::count_if(
                      rmm::exec_policy(default_stream_view),
                      edge_first,
                      edge_first + edgelist.number_of_edges,
                      out_of_range_t<vertex_t>{
                        0, this->get_number_of_vertices(), 0, this->get_number_of_vertices()}) == 0,
                    "Invalid input argument: edgelist have out-of-range values.");

    // FIXME: check for symmetricity may better be implemetned with transpose().
    if (this->is_symmetric()) {}
    // FIXME: check for duplicate edges may better be implemented after deciding whether to sort
    // neighbor list or not.
    if (!this->is_multigraph()) {}
  }

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  std::tie(offsets_, indices_, weights_, std::ignore) =
    compress_edgelist<store_transposed>(edgelist,
                                        vertex_t{0},
                                        std::optional<vertex_t>{std::nullopt},
                                        this->get_number_of_vertices(),
                                        vertex_t{0},
                                        this->get_number_of_vertices(),
                                        default_stream_view);

  // optional expensive checks (part 3/3)

  if (do_expensive_check) {
    // FIXME: check for symmetricity may better be implemetned with transpose().
    if (this->is_symmetric()) {}
    // FIXME: check for duplicate edges may better be implemented after deciding whether to sort
    // neighbor list or not.
    if (!this->is_multigraph()) {}
  }
}

}  // namespace cugraph
