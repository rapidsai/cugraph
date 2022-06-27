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
__device__ __host__ inline void _set_bit(mask_type *arr, mask_type h, bool v) {
    int bit = h & (std::numeric_limits<mask_type>::digits-1);
    int idx = h / std::numeric_limits<mask_type>::digits;

    mask_type assumed;
    mask_type old = arr[idx];
    do {
        assumed = old;
        old = atomicCAS(arr + idx, assumed, assumed & ~(bit << v));
    } while (assumed != old);
}

/**
 * Returns whether or not bit at location h is nonzero in a one-hot
 * encoded 32-bit in array.
 */
 template<typename mask_type = std::uint32_t>
__device__ __host__ inline bool _get_bit(mask_type *arr, mask_type h) {
    int bit = h & (std::numeric_limits<mask_type>::digits-1);
    int idx = h / std::numeric_limits<mask_type>::digits;
    return (arr[idx] & (1 << bit)) > 0;
}
}; // END namespace detail


/**
 * Mask view to be used in device functions for reading and updating existing mask.
 * This assumes the appropriate masks (vertex/edge) have already been initialized,
 * since that will need to be done from the owning object.
 * @tparam vertex_t
 * @tparam edge_t
 * @tparam mask_t
 */
template<typename vertex_t, typename edge_t, typename mask_t = std::uint32_t>
struct graph_mask_view_t {

public:
    graph_mask_view_t() = delete;

    graph_mask_view_t(bool has_vertex_mask,
                      bool has_edge_mask,
                      vertex_t n_vertices,
                      edge_t n_edges,
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
        if(has_vertex_mask_) {
            // TODO: This is going to implicitly limit vertex_t to uint32_t. Need to support 64-bit types as well
            return detail::_get_bit<mask_t>(vertices_, static_cast<mask_t>(vertex));
        } else {
            return !complement_;
        }
    }

    /**
     * Return whether or not a specific edge is masked
     * @param edge_offset
     * @return
     */
    __device__ bool is_edge_masked(edge_t edge_offset) {
        if(has_edge_mask_) {
            // TODO: This is going to implicitly limit edge_t to uint32_t. Need to support 64-bit types as well
            return detail::_get_bit<mask_t>(edges_, static_cast<mask_t>(edge_offset));
        } else {
            return !complement_;
        }
    }

    /**
     * Add specific vertex to mask
     * @param vertex id of vertex to mask
     * @return
     */
    __device__ void mask_vertex(vertex_t vertex) {
        RAFT_EXPECTS(has_vertex_mask_, "Vertex mask needs to be initialized before it can be used");
        detail::_set_bit<mask_t>(vertices_, static_cast<mask_t>(vertex), true);
    }

    /**
     * Add specific edge to mask
     * @param edge_offset offset of edge to mask
     * @return
     */
    __device__ void mask_edge(edge_t edge_offset) {
        RAFT_EXPECTS(has_edge_mask_, "Edge mask needs to be initialized before it can be used");
        detail::_set_bit<mask_t>(edges_, static_cast<mask_t>(edge_offset), true);
    }

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
    __host__ __device__ bool has_edge_mask() const { return has_vertex_mask_; }

    /**
     * Has the vertex mask been initialized?
     */
    __host__ __device__ bool has_vertex_mask() const { return has_edge_mask_; }

    /**
     * Get the vertex mask
     */
    __host__ __device__ mask_t *get_vertex_mask() { return vertices_; }

    /**
     * Get the edge mask
     */
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
    std::optional<mask_t const*> get_edge_mask() const {
        return edges_.size() > 0 ? std::make_optional<mask_t const*>(edges_.data()) : std::nullopt;
    }

    /**
     * Retuns the vertex mask if it has been initialized on the instance
     * @return
     */
    std::optional<mask_t const*> get_vertex_mask() const {
        return vertices_.size() > 0 ? std::make_optional<mask_t const*>(vertices_.data()) : std::nullopt;
    }

    /**
     * Initializes an edge mask by allocating the device memory
     */
    void initialize_edge_mask() {
        if(edges_.size() == 0) {
            edges_.resize(n_edges_, handle.get_stream());
            RAFT_CUDA_TRY(cudaMemsetAsync(edges_.data(), edges_.size() * sizeof(mask_t), 0, handle.get_stream()));
        }
    }

    /**
     * Initializes a vertex mask by allocating the device memory
     */
    void initialize_vertex_mask() {
        if(vertices_.size() == 0) {
            vertices_.resize(n_vertices_, handle.get_stream());
            RAFT_CUDA_TRY(cudaMemsetAsync(vertices_.data(), vertices_.size() * sizeof(mask_t), 0, handle.get_stream()));
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
    auto view() {
        return graph_mask_view_t<vertex_t, edge_t, mask_t>(has_vertex_mask(),
                                                           has_edge_mask(),
                                                           n_vertices_,
                                                           n_edges_,
                                                           vertices_.get(),
                                                           n_edges_.get(),
                                                           complement_);
    }

private:
    raft::handle_t const &handle;
    vertex_t n_vertices_;
    edge_t n_edges_;
    bool complement_ = false;
    rmm::device_uvector<mask_t> vertices_;
    rmm::device_uvector<mask_t> edges_;

}; // end struct mask

/**
 * Device helper function to query whether a specific vertex
 * and/or specific edge mask have been applied.
 * @tparam vertex_t
 * @tparam edge_t
 * @param mask mask view object for which to query
 * @param adjacency_indptr
 * @param adjacency_indices
 * @param src_vertex
 * @param dst_idx_offset
 * @return
 */
template<typename vertex_t, typename edge_t>
__device__ bool is_masked(graph_mask_view_t<vertex_t, edge_t> &mask,
                     vertex_t *adjacency_indptr,
                     edge_t *adjacency_indices,
                     vertex_t src_vertex,
                     edge_t dst_idx_offset) {

    // TODO: Need to raise exception / print appropriate error when both vertex and edge masks have not
    // been initialized.

    // TODO: Should this be looked up once in each kernel thread and passed in?
    vertex_t start = adjacency_indptr[src_vertex];

    // TODO: Add function to mask
    return mask.is_vertex_masked(adjacency_indices[start+dst_idx_offset]) &&
           mask.is_edge_masked(start+dst_idx_offset);
}




}; // END namespace cugraph
