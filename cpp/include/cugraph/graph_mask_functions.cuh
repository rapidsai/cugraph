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

#include <cstdint>
#include <limits>
#include <cub/cub.cuh>
#include <raft/core/handle.hpp>

#include <cugraph/graph_mask.hpp>

namespace cugraph {
namespace detail {

/**
 * A simple block-level kernel for computing mask vertex degrees by using CUDA intrinsics
 * for counting the number of bits set in a mask and aggregating them across threads.
 *
 * This kernel is not load balanced and so it assumes the same block size
 * will work across all vertices. An optimization could be to use the global
 * (unmasked) vertex degrees to better spread the load of computing the masked
 * vertex degrees, and subtract their complements atomically from the global
 * vertex degrees.
 */
template<typename vertex_t, typename edge_t, typename mask_type, int tpb = 128>
__global__ void masked_degree_kernel(vertex_t *degrees_output,
                                     graph_mask_view_t <vertex_t, edge_t, mask_type> &mask,
                                     vertex_t *indptr) {

    /**
     *   1. For each vertex for each block, load the start and end offsets from indptr
     *   2. Compute start and end indices in the mask, along w/ the start and end masks
     *   3. For each element in the mask array, apply start or end mask if needed,
     *      compute popc (or popcll) and perform summed reduce of the result for each vertex
     */

    typedef cub::BlockReduce <vertex_t, tpb> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    int vertex = blockIdx.x;
    vertex_t start_offset = indptr[vertex];
    vertex_t stop_offset = indptr[vertex + 1];

    mask_type start_mask_offset = start_offset / std::numeric_limits<mask_type>::digits;
    mask_type stop_mask_offset = stop_offset / std::numeric_limits<mask_type>::digits;

    mask_type start_bit = start_offset & (std::numeric_limits<mask_type>::digits - 1);
    mask_type stop_bit = stop_offset & (std::numeric_limits<mask_type>::digits - 1);

    mask_type start_mask = (0xffffffff << start_bit) >> start_bit;
    mask_type stop_mask = (0xffffffff >> stop_bit) << stop_bit;

    mask_t *vertex_mask = mask.get_vertex_mask();

    vertex_t degree = 0;
    for (int i = threadIdx.x; i < (stop_mask_offset - start_mask_offset); i += tpb) {

        mask_t mask_elm = vertex_mask[i + start_mask_offset];

        // Apply start_mask to first element
        if (i == 0) {
            mask_elm &= start_mask;

            // Apply stop_mask to last element
        } else if (i == (stop_mask_offset - start_mask_offset) - 1) {
            mask_elm &= stop_mask;
        }

        degree += __popc(mask_elm);
    }

    degree = BlockReduce(temp_storage).Sum(degree);

    if (threadIdx.x == 0) {
        degrees_output[vertex] = degree;
    }
}
} // end namspace cugraph::detail

    template<typename vertex_t, typename edge_t, typename mask_type, int tpb = 128>
    __global__ void masked_degree_kernel(vertex_t *degrees_output,
                                         graph_mask_view_t <vertex_t, edge_t, mask_type> &mask,
                                         vertex_t major_range_first,
                                         vertex_t *vertex_ids,
                                         vertex_t *indptr) {

        /**
         *   1. For each vertex for each block, load the start and end offsets from indptr
         *   2. Compute start and end indices in the mask, along w/ the start and end masks
         *   3. For each element in the mask array, apply start or end mask if needed,
         *      compute popc (or popcll) and perform summed reduce of the result for each vertex
         */

        typedef cub::BlockReduce <vertex_t, tpb> BlockReduce;

        __shared__ typename BlockReduce::TempStorage temp_storage;

        int vertex = blockIdx.x;
        vertex_t start_offset = indptr[vertex];
        vertex_t stop_offset = indptr[vertex + 1];

        mask_type start_mask_offset = start_offset / std::numeric_limits<mask_type>::digits;
        mask_type stop_mask_offset = stop_offset / std::numeric_limits<mask_type>::digits;

        mask_type start_bit = start_offset & (std::numeric_limits<mask_type>::digits - 1);
        mask_type stop_bit = stop_offset & (std::numeric_limits<mask_type>::digits - 1);

        mask_type start_mask = (0xffffffff << start_bit) >> start_bit;
        mask_type stop_mask = (0xffffffff >> stop_bit) << stop_bit;

        mask_t *vertex_mask = mask.get_vertex_mask();

        vertex_t degree = 0;
        for (int i = threadIdx.x; i < (stop_mask_offset - start_mask_offset); i += tpb) {

            mask_t mask_elm = vertex_mask[i + start_mask_offset];

            // Apply start_mask to first element
            if (i == 0) {
                mask_elm &= start_mask;

                // Apply stop_mask to last element
            } else if (i == (stop_mask_offset - start_mask_offset) - 1) {
                mask_elm &= stop_mask;
            }

            degree += __popc(mask_elm);
        }

        degree = BlockReduce(temp_storage).Sum(degree);

        if (threadIdx.x == 0) {
            degrees_output[vertex_ids[vertex] - major_range_first] = degree;
        }
    }
} // end namspace cugraph::detail

template<typename vertex_t, typename edge_t, typename mask_type>
void masked_degrees(raft::handle_t const &handle,
                    vertex_t *degrees_output,
                    vertex_t size,
                    graph_mask_view_t <vertex_t, edge_t, mask_type> &mask,
                    vertex_t *indptr) {
    masked_degree_kernel<vertex_t, edge_t, mask_type, 128>
    <<<size, 128, 0, handle.get_stream()>>>(degrees_output, mask, indptr);
}

template<typename vertex_t, typename edge_t, typename mask_type>
void masked_degrees(raft::handle_t const &handle,
                    vertex_t *degrees_output,
                    vertex_t size,
                    graph_mask_view_t <vertex_t, edge_t, mask_type> &mask,
                    vertex_t major_range_first,
                    vertex_t *vertex_ids,
                    vertex_t *indptr) {
    masked_degree_kernel<vertex_t, edge_t, mask_type, 128>
    <<<size, 128, 0, handle.get_stream()>>>(degrees_output, mask, major_range_first, vertex_ids, indptr);
}


}; // end namespace cugraph
