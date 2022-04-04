/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

// Andrei Schaffer, aschaffer@nvidia.com
//
#pragma once

#include <cugraph/serialization/serializer.hpp>

#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <thrust/tuple.h>

namespace cugraph {
namespace broadcast {

/**
 * @brief broadcasts graph_t object (only the single GPU version).
 *
 * @tparam graph_t Type of graph (view).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_ptr pointer to graph object: not `nullptr` on send, `nullptr` (ignored) on receive.
 * @return graph_t object that was sent/received
 */
template <typename graph_t>
graph_t graph_broadcast(raft::handle_t const& handle, graph_t* graph_ptr)
{
  using namespace cugraph::serializer;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  if constexpr (!graph_t::is_multi_gpu) {
    if (handle.get_comms().get_rank() == 0) {
      CUGRAPH_EXPECTS(graph_ptr != nullptr, "Cannot serialize nullptr graph pointer.");

      auto pair = serializer_t::get_device_graph_sz_bytes(*graph_ptr);
      thrust::tuple<size_t, size_t> dev_sz_host_sz_bytes =
        thrust::make_tuple(pair.first, pair.second);

      auto total_graph_dev_sz = pair.first + pair.second;

      serializer_t ser(handle, total_graph_dev_sz);
      serializer_t::graph_meta_t<graph_t> graph_meta{};
      ser.serialize(*graph_ptr, graph_meta);

      int root{0};
      host_scalar_bcast(handle.get_comms(), dev_sz_host_sz_bytes, root, handle.get_stream());
      device_bcast(handle.get_comms(),
                   ser.get_storage(),
                   ser.get_storage(),
                   total_graph_dev_sz,
                   root,
                   handle.get_stream());

      return std::move(*graph_ptr);
    } else {
      thrust::tuple<size_t, size_t> dev_sz_host_sz_bytes(0, 0);

      int root{0};
      dev_sz_host_sz_bytes =
        host_scalar_bcast(handle.get_comms(), dev_sz_host_sz_bytes, root, handle.get_stream());
      //
      auto total_graph_dev_sz =
        thrust::get<0>(dev_sz_host_sz_bytes) + thrust::get<1>(dev_sz_host_sz_bytes);

      CUGRAPH_EXPECTS(total_graph_dev_sz > 0, "Graph size comm failure.");

      rmm::device_uvector<serializer_t::byte_t> data_buffer(total_graph_dev_sz,
                                                            handle.get_stream());

      device_bcast(handle.get_comms(),
                   data_buffer.data(),
                   data_buffer.data(),
                   total_graph_dev_sz,
                   root,
                   handle.get_stream());

      serializer_t ser(handle, data_buffer.data());
      auto graph = ser.unserialize<graph_t>(thrust::get<0>(dev_sz_host_sz_bytes),
                                            thrust::get<1>(dev_sz_host_sz_bytes));

      return graph;
    }
  } else {
    CUGRAPH_FAIL("Unsupported graph type for broadcasting.");

    return graph_t{handle};
  }
}

}  // namespace broadcast
}  // namespace cugraph
