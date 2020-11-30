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
#pragma once

#include <experimental/graph_view.hpp>
#include <utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace experimental {

template <typename vertex_t, typename edge_t, typename weight_t>
struct edgelist_t {
  vertex_t const *p_src_vertices{nullptr};
  vertex_t const *p_dst_vertices{nullptr};
  weight_t const *p_edge_weights{nullptr};
  edge_t number_of_edges{0};
};

// graph_t is an owning graph class (note that graph_view_t is a non-owning graph class)
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu,
          typename Enable = void>
class graph_t;

// multi-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
class graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t, weight_t> {
 public:
  using vertex_type                              = vertex_t;
  using edge_type                                = edge_t;
  using weight_type                              = weight_t;
  static constexpr bool is_adj_matrix_transposed = store_transposed;
  static constexpr bool is_multi_gpu             = multi_gpu;

  graph_t(raft::handle_t const &handle,
          std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const &edge_lists,
          partition_t<vertex_t> const &partition,
          vertex_t number_of_vertices,
          edge_t number_of_edges,
          graph_properties_t properties,
          bool sorted_by_global_degree_within_vertex_partition,
          bool do_expensive_check = false);

  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> view()
  {
    std::vector<edge_t const *> offsets(adj_matrix_partition_offsets_.size(), nullptr);
    std::vector<vertex_t const *> indices(adj_matrix_partition_indices_.size(), nullptr);
    std::vector<weight_t const *> weights(adj_matrix_partition_weights_.size(), nullptr);
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] = adj_matrix_partition_offsets_[i].data();
      indices[i] = adj_matrix_partition_indices_[i].data();
      if (weights.size() > 0) { weights[i] = adj_matrix_partition_weights_[i].data(); }
    }

    return graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      *(this->get_handle_ptr()),
      offsets,
      indices,
      weights,
      vertex_partition_segment_offsets_,
      partition_,
      this->get_number_of_vertices(),
      this->get_number_of_edges(),
      this->get_graph_properties(),
      vertex_partition_segment_offsets_.size() > 0,
      false);
  }

 private:
  std::vector<rmm::device_uvector<edge_t>> adj_matrix_partition_offsets_{};
  std::vector<rmm::device_uvector<vertex_t>> adj_matrix_partition_indices_{};
  std::vector<rmm::device_uvector<weight_t>> adj_matrix_partition_weights_{};

  partition_t<vertex_t> partition_{};

  std::vector<vertex_t>
    vertex_partition_segment_offsets_{};  // segment offsets within the vertex partition based on
                                          // vertex degree, relevant only if
                                          // sorted_by_global_degree_within_vertex_partition is true
};

// single-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
class graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t, weight_t> {
 public:
  using vertex_type                              = vertex_t;
  using edge_type                                = edge_t;
  using weight_type                              = weight_t;
  static constexpr bool is_adj_matrix_transposed = store_transposed;
  static constexpr bool is_multi_gpu             = multi_gpu;

  graph_t(raft::handle_t const &handle,
          edgelist_t<vertex_t, edge_t, weight_t> const &edge_list,
          vertex_t number_of_vertices,
          graph_properties_t properties,
          bool sorted_by_degree,
          bool do_expensive_check = false);

  vertex_t get_number_of_local_vertices() const { return this->get_number_of_vertices(); }

  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> view()
  {
    return graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      *(this->get_handle_ptr()),
      offsets_.data(),
      indices_.data(),
      weights_.data(),
      segment_offsets_,
      this->get_number_of_vertices(),
      this->get_number_of_edges(),
      this->get_graph_properties(),
      segment_offsets_.size() > 0,
      false);
  }

 private:
  rmm::device_uvector<edge_t> offsets_;
  rmm::device_uvector<vertex_t> indices_;
  rmm::device_uvector<weight_t> weights_;
  std::vector<vertex_t> segment_offsets_{};  // segment offsets based on vertex degree, relevant
                                             // only if sorted_by_global_degree is true
};

template <typename T, typename Enable = void>
struct invalid_idx;

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>>
  : std::integral_constant<T, -1> {
};

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>>
  : std::integral_constant<T, std::numeric_limits<T>::max()> {
};

template <typename vertex_t>
struct invalid_vertex_id : invalid_idx<vertex_t> {
};

template <typename edge_t>
struct invalid_edge_id : invalid_idx<edge_t> {
};

}  // namespace experimental
}  // namespace cugraph

#include "eidecl_graph.hpp"
