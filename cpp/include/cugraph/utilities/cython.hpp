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
#pragma once

#include <cugraph/graph_generators.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace cython {

enum class numberTypeEnum : int { int32Type, int64Type, floatType, doubleType };

// replacement for std::tuple<,,>, since std::tuple is not
// supported in cython
//
template <typename vertex_t, typename edge_t, typename weight_t>
struct major_minor_weights_t {
  explicit major_minor_weights_t(raft::handle_t const& handle)
    : shuffled_major_vertices_(0, handle.get_stream()),
      shuffled_minor_vertices_(0, handle.get_stream()),
      shuffled_weights_(0, handle.get_stream())
  {
  }

  rmm::device_uvector<vertex_t>& get_major(void) { return shuffled_major_vertices_; }

  rmm::device_uvector<vertex_t>& get_minor(void) { return shuffled_minor_vertices_; }

  rmm::device_uvector<weight_t>& get_weights(void) { return shuffled_weights_; }

  std::vector<edge_t>& get_edge_counts(void) { return edge_counts_; }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_major_wrap(
    void)  // const: triggers errors in Cython autogen-ed C++
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(shuffled_major_vertices_.release()),
                          sizeof(vertex_t));
  }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_minor_wrap(void)  // const
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(shuffled_minor_vertices_.release()),
                          sizeof(vertex_t));
  }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_weights_wrap(void)  // const
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(shuffled_weights_.release()),
                          sizeof(weight_t));
  }

  std::unique_ptr<std::vector<edge_t>> get_edge_counts_wrap(void)  // const
  {
    return std::make_unique<std::vector<edge_t>>(edge_counts_);
  }

 private:
  rmm::device_uvector<vertex_t> shuffled_major_vertices_;
  rmm::device_uvector<vertex_t> shuffled_minor_vertices_;
  rmm::device_uvector<weight_t> shuffled_weights_;
  std::vector<edge_t> edge_counts_{};
};

struct graph_generator_t {
  std::unique_ptr<rmm::device_buffer> d_source;
  std::unique_ptr<rmm::device_buffer> d_destination;
};

// wrapper for renumber_edgelist() return
// (unrenumbering maps, etc.)
//
template <typename vertex_t, typename edge_t>
struct renum_tuple_t {
  explicit renum_tuple_t(raft::handle_t const& handle) : dv_(0, handle.get_stream()), part_() {}

  rmm::device_uvector<vertex_t>& get_dv(void) { return dv_; }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_dv_wrap(
    void)  // const: see above explanation
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(dv_.release()), sizeof(vertex_t));
  }

  cugraph::partition_t<vertex_t>& get_partition(void) { return part_; }
  vertex_t& get_num_vertices(void) { return nv_; }
  edge_t& get_num_edges(void) { return ne_; }

  std::vector<vertex_t>& get_segment_offsets(void) { return segment_offsets_; }

  std::unique_ptr<std::vector<vertex_t>> get_segment_offsets_wrap()
  {  // const
    return std::make_unique<std::vector<vertex_t>>(segment_offsets_);
  }

  // `partition_t` pass-through getters
  //
  int get_part_row_size() const { return part_.row_comm_size(); }

  int get_part_col_size() const { return part_.col_comm_size(); }

  int get_part_comm_rank() const { return part_.comm_rank(); }

  // FIXME: part_.vertex_partition_offsets() returns a std::vector
  //
  std::unique_ptr<std::vector<vertex_t>> get_partition_offsets_wrap(void)  // const
  {
    return std::make_unique<std::vector<vertex_t>>(part_.vertex_partition_range_offsets());
  }

  std::pair<vertex_t, vertex_t> get_part_local_vertex_range() const
  {
    auto tpl_v = part_.local_vertex_partition_range();
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_local_vertex_first() const
  {
    return part_.local_vertex_partition_range_first();
  }

  vertex_t get_part_local_vertex_last() const { return part_.local_vertex_partition_range_last(); }

  std::pair<vertex_t, vertex_t> get_part_vertex_partition_range(size_t vertex_partition_idx) const
  {
    auto tpl_v = part_.vertex_partition_range(vertex_partition_idx);
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_vertex_partition_first(size_t vertex_partition_idx) const
  {
    return part_.vertex_partition_range_first(vertex_partition_idx);
  }

  vertex_t get_part_vertex_partition_last(size_t vertex_partition_idx) const
  {
    return part_.vertex_partition_range_last(vertex_partition_idx);
  }

  vertex_t get_part_vertex_partition_size(size_t vertex_partition_idx) const
  {
    return part_.vertex_partition_range_size(vertex_partition_idx);
  }

  size_t get_part_number_of_matrix_partitions() const
  {
    return part_.number_of_local_edgex_partitions();
  }

  std::pair<vertex_t, vertex_t> get_part_matrix_partition_major_range(size_t partition_idx) const
  {
    auto tpl_v = part_.local_edgex_partition_major_range(partition_idx);
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_matrix_partition_major_first(size_t partition_idx) const
  {
    return part_.local_edge_partition_major_first(partition_idx);
  }

  vertex_t get_part_matrix_partition_major_last(size_t partition_idx) const
  {
    return part_.local_edge_partition_major_range_last(partition_idx);
  }

  vertex_t get_part_matrix_partition_major_value_start_offset(size_t partition_idx) const
  {
    return part_.local_edge_partition_major_value_start_offset(partition_idx);
  }

  std::pair<vertex_t, vertex_t> get_part_matrix_partition_minor_range() const
  {
    auto tpl_v = part_.local_edge_partition_minor_range();
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_matrix_partition_minor_first() const
  {
    return part_.local_edge_partition_minor_range_first();
  }

  vertex_t get_part_matrix_partition_minor_last() const
  {
    return part_.local_edge_partition_minor_range_last();
  }

 private:
  rmm::device_uvector<vertex_t> dv_;
  cugraph::partition_t<vertex_t> part_;
  vertex_t nv_{0};
  edge_t ne_{0};
  std::vector<vertex_t> segment_offsets_;
};

// Wrapper for calling graph generator
template <typename vertex_t>
std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist(raft::handle_t const& handle,
                                                               size_t scale,
                                                               size_t num_edges,
                                                               double a,
                                                               double b,
                                                               double c,
                                                               uint64_t seed,
                                                               bool clip_and_flip,
                                                               bool scramble_vertex_ids);
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
                             bool scramble_vertex_ids);

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
  bool is_weighted);

// Wrapper for calling renumber_edgelist() inplace:
//
template <typename vertex_t, typename edge_t>
std::unique_ptr<renum_tuple_t<vertex_t, edge_t>> call_renumber(
  raft::handle_t const& handle,
  vertex_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  vertex_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<edge_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu);

// Helper for setting up subcommunicators, typically called as part of the
// user-initiated comms initialization in Python.
//
// raft::handle_t& handle
//   Raft handle for which the new subcommunicators will be created. The
//   subcommunicators will then be accessible from the handle passed to the
//   parallel processes.
//
// size_t row_comm_size
//   Number of items in a partition row (ie. pcols), needed for creating the
//   appropriate number of subcommunicator instances.
void init_subcomms(raft::handle_t& handle, size_t row_comm_size);

}  // namespace cython
}  // namespace cugraph
