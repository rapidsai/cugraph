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

#include <experimental/graph.hpp>
#include <utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <functional>
#include <memory>
#include <type_traits>

namespace cugraph {
namespace experimental {

// Common for both single-GPU and multi-GPU versions
template <typename GraphiViewType>
class graph_base_device_view_t {
 public:
  using vertex_type                              = typename GraphiViewType::vertex_type;
  using edge_type                                = typename GraphiViewType::edge_type;
  using weight_type                              = typename GraphiViewType::weight_type;
  static constexpr bool is_adj_matrix_transposed = GraphiViewType::is_adj_matrix_transposed;
  static constexpr bool is_multi_gpu             = GraphiViewType::is_multi_gpu;

  graph_base_device_view_t()                                = delete;
  ~graph_base_device_view_t()                               = default;
  graph_base_device_view_t(graph_base_device_view_t const&) = default;
  graph_base_device_view_t(graph_base_device_view_t&&)      = default;
  graph_base_device_view_t& operator=(graph_base_device_view_t const&) = default;
  graph_base_device_view_t& operator=(graph_base_device_view_t&&) = default;

  __host__ __device__ vertex_type get_number_of_vertices() const noexcept { return number_of_vertices_; }

  __host__ __device__ edge_type get_number_of_edges() const noexcept { return number_of_edges_; }

  __host__ __device__ bool is_symmetric() const noexcept { return properties_.is_symmetric; }
  __host__ __device__ bool is_multigraph() const noexcept { return properties_.is_multigraph; }
  __host__ __device__ bool is_weighted() const noexcept { return properties_.is_weighted; }

  template <typename vertex_t = vertex_type>
  __device__ std::enable_if_t<std::is_signed<vertex_t>::value, bool> is_valid_vertex(
    vertex_type v) const noexcept
  {
    return ((v >= 0) && (v < number_of_vertices_));
  }

  template <typename vertex_t = vertex_type>
  __device__ std::enable_if_t<std::is_unsigned<vertex_t>::value, bool> is_valid_vertex(
    vertex_type v) const noexcept
  {
    return (v < number_of_vertices_);
  }

 protected:
  vertex_type number_of_vertices_{0};
  edge_type number_of_edges_{0};

  detail::graph_properties_t properties_;

  graph_base_device_view_t(GraphiViewType const& graph_view)
  {
    number_of_vertices_ = graph_view.get_number_of_vertices();
    number_of_edges_    = graph_view.get_number_of_edges();

    properties_.is_symmetric  = graph_view.is_symmetric();
    properties_.is_multigraph = graph_view.is_multigraph();
    properties_.is_weighted   = graph_view.is_weighted();
  }
};

template <typename GraphiViewType, typename Enable = void>
class graph_device_view_t;

// multi-GPU version
template <typename GraphiViewType>
class graph_device_view_t<GraphiViewType, std::enable_if_t<GraphiViewType::is_multi_gpu>>
  : public graph_base_device_view_t<GraphiViewType> {
 public:
  using vertex_type                              = typename GraphiViewType::vertex_type;
  using edge_type                                = typename GraphiViewType::edge_type;
  using weight_type                              = typename GraphiViewType::weight_type;
  static constexpr bool is_adj_matrix_transposed = GraphiViewType::is_adj_matrix_transposed;
  static constexpr bool is_multi_gpu             = true;

  graph_device_view_t()                           = delete;
  ~graph_device_view_t()                          = default;
  graph_device_view_t(graph_device_view_t const&) = default;
  graph_device_view_t(graph_device_view_t&&)      = default;
  graph_device_view_t& operator=(graph_device_view_t const&) = default;
  graph_device_view_t& operator=(graph_device_view_t&&) = default;

  graph_device_view_t(GraphiViewType const& graph_view, void* d_ptr)
    : graph_base_device_view_t<GraphiViewType>(graph_view)
  {
    num_adj_matrix_partitions_ = graph_view.get_number_of_adj_matrix_partitions();
    auto ptr                   = static_cast<uint8_t*>(d_ptr);

    adj_matrix_partition_offsets_ = reinterpret_cast<edge_type const**>(ptr);
    ptr += sizeof(edge_type const*) * num_adj_matrix_partitions_;
    adj_matrix_partition_indices_ = reinterpret_cast<vertex_type const**>(ptr);
    ptr += sizeof(vertex_type const*) * num_adj_matrix_partitions_;
    adj_matrix_partition_weights_ = reinterpret_cast<weight_type const**>(ptr);
    ptr += sizeof(weight_type const*) * num_adj_matrix_partitions_;

    adj_matrix_partition_major_range_firsts_ = reinterpret_cast<vertex_type*>(ptr);
    ptr += sizeof(vertex_type) * num_adj_matrix_partitions_;
    adj_matrix_partition_major_range_lasts_ = reinterpret_cast<vertex_type*>(ptr);
    ptr += sizeof(vertex_type) * num_adj_matrix_partitions_;
    adj_matrix_partition_minor_range_firsts_ = reinterpret_cast<vertex_type*>(ptr);
    ptr += sizeof(vertex_type) * num_adj_matrix_partitions_;
    adj_matrix_partition_minor_range_lasts_ = reinterpret_cast<vertex_type*>(ptr);

    for (size_t i = 0; i < num_adj_matrix_partitions_; ++i) {
      adj_matrix_partition_offsets_[i] = graph_view.offsets(i);
      adj_matrix_partition_indices_[i] = graph_view.indices(i);
      adj_matrix_partition_weights_[i] = graph_view.weights(i);

      std::tie(adj_matrix_partition_major_range_firsts_[i], adj_matrix_partition_major_range_lasts_[i]) =
        graph_view.get_xxx_range(i);
      std::tie(adj_matrix_partition_minor_range_firsts_[i], adj_matrix_partition_minor_range_lasts_[i]) =
        graph_view.get_xxx_range(i);
    }

    std::tie(vertex_partition_range_first_, vertex_partition_range_last_) = graph_view.get_vertex_partition_range();
  }

  // only for the create() function
  void destroy() { delete this; }

  static std::unique_ptr<graph_device_view_t, std::function<void(graph_device_view_t*)>> create(
    GraphiViewType const& graph_view)
  {
    auto buffer_size = (sizeof(edge_type const*) + sizeof(vertex_type const*) +
                        sizeof(weight_type const*) + sizeof(vertex_type) * 4) *
                       graph_view.get_number_of_adj_matrix_partitions();
    rmm::device_buffer* buffer_ptr = new rmm::device_buffer(buffer_size);
    auto deleter                   = [buffer_ptr](graph_device_view_t* graph_device_view) {
      graph_device_view->destroy();
      delete buffer_ptr;
    };
    std::unique_ptr<graph_device_view_t, decltype(deleter)> graph_device_view_ptr(
      new graph_device_view_t(graph_view, buffer_ptr->data()), deleter);
    return graph_device_view_ptr;
  }

  auto local_vertex_begin() const { r }

  auto local_vertex_end() const
  {
    return thrust::make_counting_iterator(vertex_partition_range_last_);
  }

 private:
  size_t num_adj_matrix_partitions_{0};

  edge_type const** adj_matrix_partition_offsets_{nullptr};
  vertex_type const** adj_matrix_partition_indices_{nullptr};
  weight_type const** adj_matrix_partition_weights_{nullptr};

  vertex_type* adj_matrix_partition_major_range_firsts_{nullptr};
  vertex_type* adj_matrix_partition_major_range_lasts_{nullptr};
  vertex_type* adj_matrix_partition_minor_range_firsts_{nullptr};
  vertex_type* adj_matrix_partition_minor_range_lasts_{nullptr};

  vertex_type vertex_partition_range_first_{0};
  vertex_type vertex_partition_range_last_{0};
};

// single GPU version
template <typename GraphiViewType>
class graph_device_view_t<GraphiViewType, std::enable_if_t<!GraphiViewType::is_multi_gpu>>
  : public graph_base_device_view_t<GraphiViewType> {
 public:
  using vertex_type                              = typename GraphiViewType::vertex_type;
  using edge_type                                = typename GraphiViewType::edge_type;
  using weight_type                              = typename GraphiViewType::weight_type;
  static constexpr bool is_adj_matrix_transposed = GraphiViewType::is_adj_matrix_transposed;
  static constexpr bool is_multi_gpu             = false;

  graph_device_view_t()                           = delete;
  ~graph_device_view_t()                          = default;
  graph_device_view_t(graph_device_view_t const&) = default;
  graph_device_view_t(graph_device_view_t&&)      = default;
  graph_device_view_t& operator=(graph_device_view_t const&) = default;
  graph_device_view_t& operator=(graph_device_view_t&&) = default;

  graph_device_view_t(GraphiViewType const& graph_view) : graph_base_device_view_t<GraphiViewType>(graph_view)
  {
    offsets_ = graph_view.offsets();
    indices_ = graph_view.indices();
    weights_ = graph_view.weights();
  }

  static std::unique_ptr<graph_device_view_t, std::function<void(graph_device_view_t*)>> create(
    GraphiViewType const& graph_view)
  {
    return std::make_unique<graph_device_view_t>(graph_view);
  }

  __host__ __device__ vertex_type get_number_of_local_vertices() const noexcept
  {
    return this->number_of_vertices_;
  }

  __host__ __device__ vertex_type get_number_of_adj_matrix_local_rows() const noexcept
  {
    return this->number_of_vertices_;
  }

  __host__ __device__ vertex_type get_number_of_adj_matrix_local_cols() const noexcept
  {
    return this->number_of_vertices_;
  }

  __host__ __device__ constexpr bool is_local_vertex_nocheck(vertex_type v) const noexcept
  {
    return true;
  }

  __host__ __device__ constexpr bool is_adj_matrix_local_row_nocheck(vertex_type row) const noexcept
  {
    return true;
  }

  __host__ __device__ constexpr bool is_adj_matrix_local_col_nocheck(vertex_type col) const noexcept
  {
    return true;
  }

  __host__ __device__ vertex_type
  get_vertex_from_local_vertex_offset_nocheck(vertex_type offset) const noexcept
  {
    return offset;
  }

  __host__ __device__ vertex_type get_local_vertex_offset_from_vertex_nocheck(vertex_type v) const
    noexcept
  {
    return v;
  }

  __host__ __device__ vertex_type
  get_adj_matrix_local_row_offset_from_row_nocheck(vertex_type row) const noexcept
  {
    return row;
  }

  __host__ __device__ vertex_type
  get_adj_matrix_local_col_offset_from_col_nocheck(vertex_type col) const noexcept
  {
    return col;
  }

  auto local_vertex_begin() const { return thrust::make_counting_iterator(vertex_type{0}); }

  auto local_vertex_end() const
  {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }

  // FIXME: this API does not work if a single process holds more than one rectangular partitions
  // of the adjacency matrix.
  auto adj_matrix_local_row_begin() const { return thrust::make_counting_iterator(vertex_type{0}); }

  auto adj_matrix_local_row_end() const
  {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }

  auto adj_matrix_local_col_begin() const { return thrust::make_counting_iterator(vertex_type{0}); }

  auto adj_matrix_local_col_end() const
  {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }

  template <bool transposed = is_adj_matrix_transposed>
  __device__ std::enable_if_t<transposed, edge_type> get_local_in_degree_nocheck(
    vertex_type row) const noexcept
  {
    auto row_offset = get_adj_matrix_local_row_offset_from_row_nocheck(row);
    return *(this->offsets_ + row_offset + 1) - *(this->offsets_ + row_offset);
  }

  template <bool transposed = is_adj_matrix_transposed>
  __device__ std::enable_if_t<!transposed, edge_type> get_local_out_degree_nocheck(
    vertex_type row) const noexcept
  {
    auto row_offset = get_adj_matrix_local_row_offset_from_row_nocheck(row);
    return *(this->offsets_ + row_offset + 1) - *(this->offsets_ + row_offset);
  }

  __device__ thrust::tuple<vertex_type const*, weight_type const*, vertex_type> get_local_edges(
    vertex_type row) const noexcept
  {
    auto row_offset   = get_adj_matrix_local_row_offset_from_row_nocheck(row);
    auto edge_offset  = *(offsets_ + row_offset);
    auto local_degree = *(offsets_ + (row_offset + 1)) - edge_offset;
    auto indices      = indices_ + edge_offset;
    auto weights      = this->is_weighted() ?  this->weights_ + edge_offset : nullptr;
    return thrust::make_tuple(indices, weights, local_degree);
  }

private:
  edge_type const* offsets_{nullptr};
  vertex_type const* indices_{nullptr};
  weight_type const* weights_{nullptr};
};

#if 0
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<edge_t>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  in_degree()
{
  if (!store_transposed) {
    rmm::device_uvector<edge_t> degrees(get_number_of_local_vertices(), handle_ptr_.get_stream());
    copy_v_transform_reduce_in_nbr(
      *handle_ptr_,
      graph_device_view,
      thrust::make_constant_iterator(0) /* dummy */,
      thrust::make_constant_iterator(0) /* dummy */,
      [] __device__(auto src_val, auto dst_val) { return 1; },
      edge_t{0},
      degrees.data());
    return degrees;
  } else {
    return compute_row_degree(
      *handle_ptr_, adj_matrix_partition_offsets_, partition_.hypergraph_partitioned);
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<edge_t>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  out_degree()
{
  if (store_transposed) {
    rmm::device_uvector<edge_t> degrees(get_number_of_local_vertices(), handle_ptr_.get_stream());
    copy_v_transform_reduce_out_nbr(
      *handle_ptr_,
      graph_device_view,
      thrust::make_constant_iterator(0) /* dummy */,
      thrust::make_constant_iterator(0) /* dummy */,
      [] __device__(auto src_val, auto dst_val) { return 1; },
      edge_t{0},
      degrees.data());
    return degrees;
  } else {
    return compute_row_degree(
      *handle_ptr_, adj_matrix_partition_offsets_, partition_.hypergraph_partitioned);
  }
}
#endif

}  // namespace experimental
}  // namespace cugraph
