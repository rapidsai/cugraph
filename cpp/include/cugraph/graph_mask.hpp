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

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <cstdint>
#include <optional>
#include <vector>
#include <limits>


namespace cugraph {
    namespace detail {

/**
 * Zeros the bit at location h in a one-hot encoded 32-bit int array
 */
template<typename mask_type = std::uint32_t>
__device__ __host__ inline void _zero_bit(mask_type *arr, mask_type h) {
    int bit = h & (std::numeric_limits<mask_type>::digits-1);
    int idx = h / std::numeric_limits<mask_type>::digits;

    mask_type assumed;
    mask_type old = arr[idx];
    do {
        assumed = old;
        old = atomicCAS(arr + idx, assumed, assumed & ~(1 << bit));
    } while (assumed != old);
}

/**
 * Returns whether or not bit at location h is nonzero in a one-hot
 * encoded 32-bit in array.
 */
 template<typename mask_type = std::uint32_t>
__device__ __host__ inline bool _get_val(mask_type *arr, mask_type h) {
    int bit = h & (std::numeric_limits<mask_type>::digits-1);
    int idx = h / std::numeric_limits<mask_type>::digits;
    return (arr[idx] & (1 << bit)) > 0;
}
}; // END namespace detail


template<typename vertex_t, typename edge_t, typename mask_t = std::uint32_t>
struct graph_mask_view_t {

public:
    graph_mask_view_t() = delete;

    graph_mask_view_t(bool has_vertex_mask,
                      bool has_edge_mask,
                      vertex_t n_vertices,
                      edge_t n_edges_,
                      mask_t *vertices,
                      mask_t *edges,
                      bool complement = false) :
                      has_vertex_mask_(has_vertex_mask),
                      has_edge_mask_(has_edge_mask),
                      n_vertices_(n_vertices),
                      n_edges_(n_edges),
                      complement_(complement),
                      vertices_(vertices),
                      edges_(edges) {}

    /**
      * Return whether or not a specific vertex is masked
      * @param vertex
      * @return
      */
    __device__ bool is_vertex_masked(vertex_t vertex) {
        if(vertices_.size() > 0) {
            // TODO: This is going to implicitly limit vertex_t to uint32_t. Need to support 64-bit types as well
            return detail::_get_val<mask_t>(vertices_, static_cast<mask_t>(vertex));
        } else {
            return !complement_;
        }
    }

    __device__ bool is_edge_masked(edge_t edge_offset) {
        if(edges_.size() > 0) {
            // TODO: This is going to implicitly limit edge_t to uint32_t. Need to support 64-bit types as well
            return detail::get_val<mask_t>(edges_, static_cast<mask_t>(edge_offset));
        } else {
            return !complement_;
        }
    }

    __host__ __device__ bool is_complemented() const { return complement_; }

    __host__ __device__ bool has_edge_mask() const { return has_vertex_mask_; }
    __host__ __device__ bool has_vertex_mask() const { return has_edge_mask_; }

    __host__ __device__ mask_t *get_vertex_mask() { return vertices_; }
    __host__ __device__ mask_t *get_edge_mask() { return edges_; }

private:
    bool has_vertex_mask_{false};
    bool has_edge_mask_{false};
    vertex_t n_vertices_;
    edge_t n_edges_;
    bool complement_{false};
    mask_t *vertices_;
    mask_t *edges_;
};


/**
 * An owning container object to manage separate bitmasks for
 * filtering vertices and edges. A compliment setting
 * determines whether the value of 1 for corresponding
 * items means they should be masked in (included) or
 * masked out (excluded).
 *
 * @tparam vertex_t
 * @tparam edge_t
 * @tparam mask_t
 */
template <typename vertex_t, typename edge_t, typename mask_t = std::uint32_t>
struct graph_mask_t {
public:
    graph_mask_t() = delete;

    graph_mask_t(raft::handle_t const &handle,
         vertex_t n_vertices,
         edge_t n_edges,
         bool complement = false) :
            n_vertices_(n_vertices),
            n_edges_(n_edges),
            edges_(0, handle.get_stream()),
            vertices_(0, handle.get_stream()),
            complement_(complement)
    {}

    bool is_complemented() const { return complement_; }

    bool has_edge_mask() const { return get_edge_mask().has_value(); }
    bool has_vertex_mask() const { return get_vertex_mask().has_value(); }

    std::optional<mask_t const*> get_edge_mask() const {
        return edges_.size() > 0 ? std::make_optional<mask_t const*> edges_.data() : std::nullopt;
    }

    std::optional<mask_t const*> get_vertex_mask() const {
        return vertices_.size() > 0 ? std::make_optional<mask_t const*> vertices_.data() : std::nullopt;
    }


    void initialize_edge_mask() {
        if(edges_.size() == 0) {
            edges_.resize(n_edges_, handle.get_stream());
        }
    }

    void initialize_vertex_mask() {
        if(vertices_.size() == 0) {
            vertices_.resize(n_vertices_, handle.get_stream());
        }
    }

private:
    raft::handle_t const &handle;
    vertex_t n_vertices_;
    edge_t n_edges_;
    bool complement_ = false;
    rmm::device_uvector<mask_t> vertices_;
    rmm::device_uvector<mask_t> edges_;

}; // end struct mask


template<typename vertex_t, typename edge_t>
__device__ is_masked(graph_mask_view_t<vertex_t, edge_t> &mask,
                     vertex_t *adjacency_indptr,
                     edge_t *adjacency_indices,
                     vertex_t src_vertex,
                     edge_t dst_idx_offset) {

    // TODO: Should this be looked up once in each kernel thread and passed in?
    IdxT start = adjacency_indptr[src_vertex];

    // TODO: Add function to mask
    return mask.is_vertex_masked(adjacency_indices[start+dst_idx_offset]) &&
           mask.is_edge_masked(start+dst_idx_offset);
}




}; // END namespace cugraph
