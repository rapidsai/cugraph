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

#include "graph.hpp"

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <cstdint>
#include <optional>
#include <vector>

namespace cugraph {
namespace detail {

    /**
     * Zeros the bit at location h in a one-hot encoded 32-bit int array
     */
    __device__ __host__ inline void _zero_bit(std::uint32_t* arr, std::uint32_t h)
    {
        int bit = h % 32;
        int idx = h / 32;

        std::uint32_t assumed;
        std::uint32_t old = arr[idx];
        do {
            assumed = old;
            old     = atomicCAS(arr + idx, assumed, assumed & ~(1 << bit));
        } while (assumed != old);
    }

    /**
     * Returns whether or not bit at location h is nonzero in a one-hot
     * encoded 32-bit in array.
     */
    __device__ __host__ inline bool _get_val(std::uint32_t* arr, std::uint32_t h)
    {
        int bit = h % 32;
        int idx = h / 32;
        return (arr[idx] & (1 << bit)) > 0;
    }

}


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
    };
}

