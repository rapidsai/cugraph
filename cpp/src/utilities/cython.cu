/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>

#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/cython.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <numeric>
#include <vector>

namespace cugraph {
namespace cython {

// Wrapper for graph generate_rmat_edgelist()
// to expose the API to cython
// enum class generator_distribution_t { POWER_LAW = 0, UNIFORM };
template <typename vertex_t>
std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist(raft::handle_t const& handle,
                                                               size_t scale,
                                                               size_t num_edges,
                                                               double a,
                                                               double b,
                                                               double c,
                                                               uint64_t seed,
                                                               bool clip_and_flip,
                                                               bool scramble_vertex_ids)
{
  auto src_dst_tuple = cugraph::generate_rmat_edgelist<vertex_t>(
    handle, scale, num_edges, a, b, c, seed, clip_and_flip);

  if (scramble_vertex_ids) {
    cugraph::scramble_vertex_ids<vertex_t>(
      handle, std::get<0>(src_dst_tuple), std::get<1>(src_dst_tuple), vertex_t{0}, seed);
  }

  graph_generator_t gg_vals{
    std::make_unique<rmm::device_buffer>(std::get<0>(src_dst_tuple).release()),
    std::make_unique<rmm::device_buffer>(std::get<1>(src_dst_tuple).release())};

  return std::make_unique<graph_generator_t>(std::move(gg_vals));
}

template <typename vertex_t>
std::vector<std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists(raft::handle_t const& handle,
                             size_t n_edgelists,
                             size_t min_scale,
                             size_t max_scale,
                             size_t edge_factor,
                             cugraph::generator_distribution_t size_distribution,
                             cugraph::generator_distribution_t edge_distribution,
                             uint64_t seed,
                             bool clip_and_flip,
                             bool scramble_vertex_ids)
{
  auto src_dst_vec_tuple = cugraph::generate_rmat_edgelists<vertex_t>(handle,
                                                                      n_edgelists,
                                                                      min_scale,
                                                                      max_scale,
                                                                      edge_factor,
                                                                      size_distribution,
                                                                      edge_distribution,
                                                                      seed,
                                                                      clip_and_flip);

  if (scramble_vertex_ids) {
    std::for_each(
      src_dst_vec_tuple.begin(), src_dst_vec_tuple.end(), [&handle, seed](auto& src_dst_tuple) {
        cugraph::scramble_vertex_ids<vertex_t>(
          handle, std::get<0>(src_dst_tuple), std::get<1>(src_dst_tuple), vertex_t{0}, seed);
      });
  }

  std::vector<std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
    gg_vec;

  std::transform(
    src_dst_vec_tuple.begin(),
    src_dst_vec_tuple.end(),
    std::back_inserter(gg_vec),
    [](auto& tpl_dev_uvec) {
      return std::make_pair(
        std::move(std::make_unique<rmm::device_buffer>(std::get<0>(tpl_dev_uvec).release())),
        std::move(std::make_unique<rmm::device_buffer>(std::get<1>(tpl_dev_uvec).release())));
    });

  return gg_vec;
}

// wrapper for shuffling:
//
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<major_minor_weights_t<vertex_t, edge_t, weight_t>> call_shuffle(
  raft::handle_t const& handle,
  vertex_t*
    edgelist_major_vertices,  // [IN / OUT]: groupby_gpu_id_and_shuffle_values() sorts in-place
  vertex_t* edgelist_minor_vertices,  // [IN / OUT]
  weight_t* edgelist_weights,         // [IN / OUT]
  edge_t num_edgelist_edges,
  bool is_weighted)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  std::unique_ptr<major_minor_weights_t<vertex_t, edge_t, weight_t>> ptr_ret =
    std::make_unique<major_minor_weights_t<vertex_t, edge_t, weight_t>>(handle);

  if (is_weighted) {
    auto zip_edge = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights));

    std::forward_as_tuple(
      std::tie(ptr_ret->get_major(), ptr_ret->get_minor(), ptr_ret->get_weights()),
      std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        comm,  // handle.get_comms(),
        zip_edge,
        zip_edge + num_edgelist_edges,
        [key_func =
           cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
             comm.get_size(), row_comm.get_size(), col_comm.get_size()}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  } else {
    auto zip_edge = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices));

    std::forward_as_tuple(std::tie(ptr_ret->get_major(), ptr_ret->get_minor()),
                          std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        comm,  // handle.get_comms(),
        zip_edge,
        zip_edge + num_edgelist_edges,
        [key_func =
           cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
             comm.get_size(), row_comm.get_size(), col_comm.get_size()}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  }

  auto local_partition_id_op =
    [comm_size,
     key_func = cugraph::detail::compute_partition_id_from_ext_edge_endpoints_t<vertex_t>{
       comm_size, row_comm_size, col_comm_size}] __device__(auto pair) {
      return key_func(thrust::get<0>(pair), thrust::get<1>(pair)) /
             comm_size;  // global partition id to local partition id
    };
  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(ptr_ret->get_major().data(), ptr_ret->get_minor().data()));

  auto edge_counts = (is_weighted)
                       ? cugraph::groupby_and_count(pair_first,
                                                    pair_first + ptr_ret->get_major().size(),
                                                    ptr_ret->get_weights().data(),
                                                    local_partition_id_op,
                                                    col_comm_size,
                                                    false,
                                                    handle.get_stream())
                       : cugraph::groupby_and_count(pair_first,
                                                    pair_first + ptr_ret->get_major().size(),
                                                    local_partition_id_op,
                                                    col_comm_size,
                                                    false,
                                                    handle.get_stream());

  std::vector<size_t> h_edge_counts(edge_counts.size());
  raft::update_host(
    h_edge_counts.data(), edge_counts.data(), edge_counts.size(), handle.get_stream());
  handle.sync_stream();

  ptr_ret->get_edge_counts().resize(h_edge_counts.size());
  for (size_t i = 0; i < h_edge_counts.size(); ++i) {
    ptr_ret->get_edge_counts()[i] = static_cast<edge_t>(h_edge_counts[i]);
  }

  return ptr_ret;  // RVO-ed
}

// Wrapper for calling renumber_edeglist() inplace:
// TODO: check if return type needs further handling...
//
template <typename vertex_t, typename edge_t>
std::unique_ptr<renum_tuple_t<vertex_t, edge_t>> call_renumber(
  raft::handle_t const& handle,
  vertex_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  vertex_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<edge_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu)  // bc. cython cannot take non-type template params
{
  // caveat: return values have different types on the 2 branches below:
  //
  std::unique_ptr<renum_tuple_t<vertex_t, edge_t>> p_ret =
    std::make_unique<renum_tuple_t<vertex_t, edge_t>>(handle);

  if (multi_gpu) {
    std::vector<edge_t> displacements(edge_counts.size(), edge_t{0});
    std::partial_sum(edge_counts.begin(), edge_counts.end() - 1, displacements.begin() + 1);
    std::vector<vertex_t*> src_ptrs(edge_counts.size());
    std::vector<vertex_t*> dst_ptrs(src_ptrs.size());
    for (size_t i = 0; i < edge_counts.size(); ++i) {
      src_ptrs[i] = shuffled_edgelist_src_vertices + displacements[i];
      dst_ptrs[i] = shuffled_edgelist_dst_vertices + displacements[i];
    }

    cugraph::renumber_meta_t<vertex_t, edge_t, true> meta{};
    std::tie(p_ret->get_dv(), meta) =
      cugraph::renumber_edgelist<vertex_t, edge_t, true>(handle,
                                                         std::nullopt,
                                                         src_ptrs,
                                                         dst_ptrs,
                                                         edge_counts,
                                                         std::nullopt,
                                                         store_transposed,
                                                         do_expensive_check);
    p_ret->get_num_vertices()    = meta.number_of_vertices;
    p_ret->get_num_edges()       = meta.number_of_edges;
    p_ret->get_partition()       = meta.partition;
    p_ret->get_segment_offsets() = meta.edge_partition_segment_offsets;
  } else {
    cugraph::renumber_meta_t<vertex_t, edge_t, false> meta{};
    std::tie(p_ret->get_dv(), meta) =
      cugraph::renumber_edgelist<vertex_t, edge_t, false>(handle,
                                                          std::nullopt,
                                                          shuffled_edgelist_src_vertices,
                                                          shuffled_edgelist_dst_vertices,
                                                          edge_counts[0],
                                                          store_transposed,
                                                          do_expensive_check);

    p_ret->get_num_vertices()    = static_cast<vertex_t>(p_ret->get_dv().size());
    p_ret->get_num_edges()       = edge_counts[0];
    p_ret->get_partition()       = cugraph::partition_t<vertex_t>{};  // dummy
    p_ret->get_segment_offsets() = meta.segment_offsets;
  }

  return p_ret;  // RVO-ed (copy ellision)
}

// Helper for setting up subcommunicators
void init_subcomms(raft::handle_t& handle, size_t row_comm_size)
{
  partition_2d::subcomm_factory_t<partition_2d::key_naming_t> subcomm_factory(handle,
                                                                              row_comm_size);
}

template std::unique_ptr<major_minor_weights_t<int32_t, int32_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int32_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int32_t, int64_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int32_t, int32_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int32_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int32_t, int64_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int64_t, int64_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int64_t* edgelist_major_vertices,
  int64_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int64_t, int64_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int64_t* edgelist_major_vertices,
  int64_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

// TODO: add the remaining relevant EIDIr's:
//
template std::unique_ptr<renum_tuple_t<int32_t, int32_t>> call_renumber(
  raft::handle_t const& handle,
  int32_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  int32_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<int32_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<renum_tuple_t<int32_t, int64_t>> call_renumber(
  raft::handle_t const& handle,
  int32_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  int32_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<int64_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<renum_tuple_t<int64_t, int64_t>> call_renumber(
  raft::handle_t const& handle,
  int64_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  int64_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<int64_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist<int32_t>(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool clip_and_flip,
  bool scramble_vertex_ids);

template std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist<int64_t>(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool clip_and_flip,
  bool scramble_vertex_ids);

template std::vector<
  std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists<int32_t>(raft::handle_t const& handle,
                                      size_t n_edgelists,
                                      size_t min_scale,
                                      size_t max_scale,
                                      size_t edge_factor,
                                      cugraph::generator_distribution_t size_distribution,
                                      cugraph::generator_distribution_t edge_distribution,
                                      uint64_t seed,
                                      bool clip_and_flip,
                                      bool scramble_vertex_ids);

template std::vector<
  std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists<int64_t>(raft::handle_t const& handle,
                                      size_t n_edgelists,
                                      size_t min_scale,
                                      size_t max_scale,
                                      size_t edge_factor,
                                      cugraph::generator_distribution_t size_distribution,
                                      cugraph::generator_distribution_t edge_distribution,
                                      uint64_t seed,
                                      bool clip_and_flip,
                                      bool scramble_vertex_ids);

}  // namespace cython
}  // namespace cugraph
