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

namespace cugraph {
namespace broadcast {

using namespace cugraph::experimental;

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
      thrust::tuple<size_t, size_t> dev_sz_host_sz_bytes(pair);

      auto total_graph_dev_sz = pair.first + pair.second;

      serializer_t ser(handle, total_graph_dev_sz);
      graph_meta_t<graph_t> graph_meta{};
      ser.serialize(graph, graph_meta);

      // TODO:
      //
      // host_scalar_bcast(..., &dev_sz_host_sz_bytes, ...);
      //
      // device_bcast(..., ser.get_storage(), ...);

      return std::move(*graph_ptr);
    } else {
      thrust::tuple<size_t, size_t> dev_sz_host_sz_bytes;

      // TODO:
      //
      // host_scalar_bcast(..., &dev_sz_host_sz_bytes, ...);
      //
      auto total_graph_dev_sz =
        thrust::get<0>(dev_sz_host_sz_bytes) + thrust::get<1>(dev_sz_host_sz_bytes);

      rmm::device_uvector<std::byte> data_buffer(total_graph_dev_sz, handle.get_stream_view());

      // TODO:
      //
      // device_bcast(..., data_buffer.data(), ...);
      //
      serializer_t ser(handle, data_buffer.data());
      auto graph =
        ser.unserialize(thrust::get<0>(dev_sz_host_sz_bytes), thrust::get<1>(dev_sz_host_sz_bytes));

      return graph;
    }
  } else {
    CUGRAPH_FAIL("Unsupported graph type for broadcasting.");

    return graph_t{handle};
  }
}

}  // namespace broadcast
}  // namespace cugraph
