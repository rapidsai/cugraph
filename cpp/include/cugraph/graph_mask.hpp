/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace cugraph {

/**
 * Compile-time fast lookup of log2(num_bits(mask_t)) to eliminate
 * the log2 computation for powers of 2.
 * @tparam mask_t
 */
template <typename mask_t>
__host__ __device__ constexpr int log_bits()
{
  switch (std::numeric_limits<mask_t>::digits) {
    case 8: return 3;
    case 16: return 4;
    case 32: return 5;
    case 64: return 6;
    default: return log2(std::numeric_limits<mask_t>::digits);
  }
}

/**
 * Uses bit-shifting to perform a fast mod operation. This
 * is used to compute the index of a specific bit
 * @tparam mask_t
 * @tparam T
 */
template <typename mask_t, typename T>
__host__ __device__ int bit_mod(T numerator)
{
  return numerator & (std::numeric_limits<mask_t>::digits - 1);
}

namespace detail {

/**
 * Sets the bit at location h in a one-hot encoded 32-bit int array
 */
template <typename mask_type>
__device__ __host__ inline void _set_bit(mask_type* arr, mask_type h)
{
  mask_type bit = bit_mod<mask_type>(h);
  mask_type idx = h >> log_bits<mask_type>();
  atomicOr(arr + idx, 1 << bit);
}

/**
 * Unsets the bit at location h in a one-hot encoded 32-bit int array
 */
template <typename mask_type>
__device__ __host__ inline void _unset_bit(mask_type* arr, mask_type h)
{
  mask_type bit = bit_mod<mask_type>(h);
  mask_type idx = h >> log_bits<mask_type>();
  atomicAnd(arr + idx, ~(1 << bit));
}

/**
 * Returns whether or not bit at location h is nonzero in a one-hot
 * encoded 32-bit in array.
 */
template <typename mask_type>
__device__ __host__ inline bool _is_set(mask_type* arr, mask_type h)
{
  mask_type bit = bit_mod<mask_type>(h);
  mask_type idx = h >> log_bits<mask_type>();
  return arr[idx] >> bit & 1U;
}
};  // namespace detail

/**
 * Mask view to be used in device functions for reading and updating existing mask.
 * This assumes the appropriate masks (vertex/edge) have already been initialized,
 * since that will need to be done from the owning object.
 * @tparam vertex_t
 * @tparam edge_t
 * @tparam mask_t
 */
template <typename vertex_t, typename edge_t, typename mask_t = std::uint32_t>
struct graph_mask_view_t {
 public:
  graph_mask_view_t() = delete;

  graph_mask_view_t(vertex_t n_vertices,
                    edge_t n_edges,
                    std::optional<raft::device_span<mask_t>>& vertices,
                    std::optional<raft::device_span<mask_t>>& edges,
                    bool complement = false)
    : n_vertices_(n_vertices),
      n_edges_(n_edges),
      complement_(complement),
      vertices_(vertices),
      edges_(edges)
  {
  }

  graph_mask_view_t(graph_mask_view_t const& other) = default;
  using vertex_type                                 = vertex_t;
  using edge_type                                   = edge_t;
  using mask_type                                   = mask_t;
  using size_type                                   = std::size_t;

  ~graph_mask_view_t()                            = default;
  graph_mask_view_t(graph_mask_view_t&&) noexcept = default;

  graph_mask_view_t& operator=(graph_mask_view_t&&) noexcept = default;
  graph_mask_view_t& operator=(graph_mask_view_t const& other) = default;

  /**
   * Are masks complemeneted?
   *
   * - !complemented means masks are inclusive (masking in)
   * - complemented means masks are exclusive (masking out)
   */
  __host__ __device__ bool is_complemented() const { return complement_; }

  /**
   * Has the edge mask been initialized?
   */
  __host__ __device__ bool has_edge_mask() const { return edges_.has_value(); }

  /**
   * Has the vertex mask been initialized?
   */
  __host__ __device__ bool has_vertex_mask() const { return vertices_.has_value(); }

  /**
   * Get the vertex mask
   */
  __host__ __device__ std::optional<raft::device_span<mask_t>> get_vertex_mask() const
  {
    return vertices_;
  }

  /**
   * Get the edge mask
   */
  __host__ __device__ std::optional<raft::device_span<mask_t>> get_edge_mask() const
  {
    return edges_;
  }

  __host__ __device__ edge_t get_edge_mask_size() const { return n_edges_ >> log_bits<mask_t>(); }

  __host__ __device__ vertex_t get_vertex_mask_size() const
  {
    return n_vertices_ >> log_bits<mask_t>();
  }

 protected:
  vertex_t n_vertices_;
  edge_t n_edges_;
  bool complement_{false};
  std::optional<raft::device_span<mask_t>> vertices_{std::nullopt};
  std::optional<raft::device_span<mask_t>> edges_{std::nullopt};
};  // struct graph_mask_view_t

/**
 * An owning container object to manage separate bitmasks for
 * filtering vertices and edges. A compliment setting
 * determines whether the value of 1 for corresponding
 * items means they should be masked in (included) or
 * masked out (excluded).
 *
 * Creating this object does not allocate any memory on device.
 * In order to start using and querying the masks, they will
 * need to first be initialized.
 *
 * @tparam vertex_t
 * @tparam edge_t
 * @tparam mask_t
 */
template <typename vertex_t, typename edge_t, typename mask_t = std::uint32_t>
struct graph_mask_t {
 public:
  using vertex_type = vertex_t;
  using edge_type   = edge_t;
  using mask_type   = mask_t;
  using size_type   = std::size_t;

  ~graph_mask_t()                       = default;
  graph_mask_t(graph_mask_t&&) noexcept = default;

  graph_mask_t() = delete;

  explicit graph_mask_t(raft::handle_t const& handle,
                        vertex_t n_vertices,
                        edge_t n_edges,
                        bool complement = false)
    : handle_(handle),
      n_vertices_(n_vertices),
      n_edges_(n_edges),
      edges_(0, handle.get_stream()),
      vertices_(0, handle.get_stream()),
      complement_(complement)
  {
  }

  explicit graph_mask_t(graph_mask_t const& other)
    : handle_(other.handle_),
      n_vertices_(other.n_vertices_),
      n_edges_(other.n_edges_),
      edges_(other.edges_, other.handle_.get_stream()),
      vertices_(other.vertices_, other.handle_.get_stream()),
      complement_(other.complement_)
  {
  }

  graph_mask_t& operator=(graph_mask_t&&) noexcept = default;
  graph_mask_t& operator=(graph_mask_t const& other) = default;

  /**
   * Determines whether the 1 bit in a vertex or edge position
   * represents an inclusive mask or exclusive mask. Default is
   * an inclusive mask (e.g. 1 bit means the corresponding vertex
   * or edge should be included in computations).
   * @return
   */
  bool is_complemented() const { return complement_; }

  /**
   * Whether or not the current mask object has been initialized
   * with an edge mask.
   * @return
   */
  bool has_edge_mask() const { return get_edge_mask().has_value(); }

  /**
   * Whether or not the current mask object has been initialized
   * with a vertex mask.
   * @return
   */
  bool has_vertex_mask() const { return get_vertex_mask().has_value(); }

  /**
   * Returns the edge mask if it has been initialized on the instance
   * @return
   */
  std::optional<mask_t const*> get_edge_mask() const
  {
    return edges_.size() > 0 ? std::make_optional<mask_t const*>(edges_.data()) : std::nullopt;
  }

  /**
   * Retuns the vertex mask if it has been initialized on the instance
   * @return
   */
  std::optional<mask_t const*> get_vertex_mask() const
  {
    return vertices_.size() > 0 ? std::make_optional<mask_t const*>(vertices_.data())
                                : std::nullopt;
  }

  vertex_t get_n_vertices() { return n_vertices_; }

  edge_t get_n_edges() { return n_edges_; }

  edge_t get_edge_mask_size() const { return n_edges_ >> log_bits<mask_t>(); }

  vertex_t get_vertex_mask_size() const { return n_vertices_ >> log_bits<mask_t>(); }

  void initialize_edge_mask(bool init = 0)
  {
    if (!has_edge_mask()) {
      allocate_edge_mask();
      RAFT_CUDA_TRY(cudaMemsetAsync(edges_.data(),
                                    edges_.size() * sizeof(mask_t),
                                    std::numeric_limits<mask_t>::max() * init,
                                    handle_.get_stream()));
    }
  }

  void initialize_vertex_mask(bool init = 0)
  {
    if (!has_vertex_mask()) {
      allocate_vertex_mask();
      RAFT_CUDA_TRY(cudaMemsetAsync(vertices_.data(),
                                    vertices_.size() * sizeof(mask_t),
                                    std::numeric_limits<mask_t>::max() * init,
                                    handle_.get_stream()));
    }
  }

  /**
   * Initializes an edge mask by allocating the device memory
   */
  void allocate_edge_mask()
  {
    if (edges_.size() == 0) {
      edges_.resize(get_edge_mask_size(), handle_.get_stream());
      clear_edge_mask();
    }
  }

  /**
   * Initializes a vertex mask by allocating the device memory
   */
  void allocate_vertex_mask()
  {
    if (vertices_.size() == 0) {
      vertices_.resize(get_vertex_mask_size(), handle_.get_stream());
      clear_vertex_mask();
    }
  }

  /**
   * Clears out all the masked bits of the edge mask
   */
  void clear_edge_mask()
  {
    if (edges_.size() > 0) {
      RAFT_CUDA_TRY(
        cudaMemsetAsync(edges_.data(), edges_.size() * sizeof(mask_t), 0, handle_.get_stream()));
    }
  }

  /**
   * Clears out all the masked bits of the vertex mask
   */
  void clear_vertex_mask()
  {
    if (vertices_.size() > 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        vertices_.data(), vertices_.size() * sizeof(mask_t), 0, handle_.get_stream()));
    }
  }

  /**
   * Returns a view of the current mask object which can safely be used
   * in device functions.
   *
   * Note that the view will not be able to initialize the underlying
   * masks so they will need to be initialized before this method is
   * invoked.
   */
  auto view()
  {
    auto vspan = has_vertex_mask() ? std::make_optional<raft::device_span<mask_t>>(vertices_.data(),
                                                                                   vertices_.size())
                                   : std::nullopt;
    auto espan = has_edge_mask()
                   ? std::make_optional<raft::device_span<mask_t>>(edges_.data(), edges_.size())
                   : std::nullopt;
    return graph_mask_view_t<vertex_t, edge_t, mask_t>(
      n_vertices_, n_edges_, vspan, espan, complement_);
  }

 protected:
  raft::handle_t const& handle_;
  vertex_t n_vertices_;
  edge_t n_edges_;
  bool complement_ = false;
  rmm::device_uvector<mask_t> vertices_;
  rmm::device_uvector<mask_t> edges_;

};  // struct graph_mask_t
};  // namespace cugraph