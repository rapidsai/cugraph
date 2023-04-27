/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>
#include <cugraph/mtmg/handle.hpp>

namespace cugraph {
namespace mtmg {

template <typename result_t>
class vertex_result_t : public detail::device_shared_wrapper_t<rmm::device_uvector<result_t>> {
 public:
  vertex_result_t() : detail::device_shared_wrapper_t<rmm::device_uvector<result_t>>() {}

  template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
  rmm::device_uvector<result_t> gather(
    handle_t const& handle,
    raft::device_span<vertex_t const> vertices,
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view)
  {
    rmm::device_uvector<result_t> result(0, handle.raft_handle().get_stream());
    return result;

    // Use this function to send vertex id/gpu rank to the right place
    // Then on each GPU do a thrust::gather
    // Then we need to shuffle each (vertex_id, value, gpu_rank) tuple back
    //    to the GPU rank it came from
    // Need to write some logic to put things in the proper order for the return value
    // Then we can copy to host

    // Should input/output be an std::vector/raft::host_span ?

    /*
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values,
  std::vector<vertex_t> const& vertex_partition_range_lasts);
    */
  }
};

}  // namespace mtmg
}  // namespace cugraph
