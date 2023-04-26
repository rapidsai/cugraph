/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cugraph/graph_generators.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/cython.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

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
      handle, std::get<0>(src_dst_tuple), std::get<1>(src_dst_tuple), vertex_t{0});
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
          handle, std::get<0>(src_dst_tuple), std::get<1>(src_dst_tuple), vertex_t{0});
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

// Helper for setting up subcommunicators
void init_subcomms(raft::handle_t& handle, size_t row_comm_size)
{
  partition_manager::init_subcomm(handle, row_comm_size);
}

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
