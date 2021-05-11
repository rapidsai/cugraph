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

#include <serialization/serializer.hpp>

#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

namespace cugraph {
namespace serializer {
template <typename value_t>
serializer::device_byte_it serializer::serialize(raft::handle_t const& handle,
                                                 rmm::device_uvector<value_t> const& src,
                                                 serializer::device_byte_it it_dev_dest) const
{
  auto byte_buff_sz       = src.size() * sizeof(value_t);
  auto it_end             = it_dev_dest + byte_buff_sz;
  byte_t const* byte_buff = reinterpret_cast<byte_t const*>(src.data());

  thrust::copy_n(rmm::exec_policy(handle.get_stream_view()), byte_buff, byte_buff_sz, it_dev_dest);

  return it_end;
}

template <typename value_t>
rmm::device_uvector<value_t> serializer::unserialize(raft::handle_t const& handle,
                                                     serializer::device_byte_it it_dev_src,
                                                     size_t size) const
{
  auto byte_buff_sz = size * sizeof(value_t);
  rmm::device_uvector<value_t> d_dest(size, handle.get_stream());
  byte_t* byte_buff = reinterpret_cast<byte_t*>(d_dest.data());

  thrust::copy_n(rmm::exec_policy(handle.get_stream_view()), it_dev_src, byte_buff_sz, byte_buff);

  return d_dest;
}

template <typename graph_view_t>
std::pair<serializer::device_byte_it, serializer::graph_meta_t<graph_view_t>> serializer::serialize(
  raft::handle_t const& handle,
  graph_view_t const& gview,
  serializer::device_byte_it it_dev_dest) const
{
  using vertex_t = typename graph_view_t::vertex_type;
  using edge_t   = typename graph_view_t::edge_type;
  using weight_t = typename graph_view_t::weight_type;

  if constexpr (!graph_view_t::is_multi_gpu) {
    graph_meta_t<graph_view_t> gvmeta{gview.get_number_of_vertices(),
                                      gview.get_number_of_edges(),
                                      gview.get_graph_properties(),
                                      gview.get_local_adj_matrix_partition_segment_offsets()};

    // TODO:
    //

    return std::make_pair(it_dev_dest, gvmeta);  // for now; FIXME

  } else {
    graph_meta_t<graph_view_t> gvmeta{};

    CUGRAPH_FAIL("Unsupported graph_view type for serialization.");

    return std::make_pair(it_dev_dest, gvmeta);  // for now; FIXME
  }
}

// Manual template instantiations (EIDir's):
//
template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& src,
  serializer::device_byte_it it_dev_dest) const;

template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& src,
  serializer::device_byte_it it_dev_dest) const;

template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<float> const& src,
  serializer::device_byte_it it_dev_dest) const;

template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<double> const& src,
  serializer::device_byte_it it_dev_dest) const;

template rmm::device_uvector<int32_t> serializer::unserialize(raft::handle_t const& handle,
                                                              serializer::device_byte_it it_dev_src,
                                                              size_t size) const;

template rmm::device_uvector<int64_t> serializer::unserialize(raft::handle_t const& handle,
                                                              serializer::device_byte_it it_dev_src,
                                                              size_t size) const;

template rmm::device_uvector<float> serializer::unserialize(raft::handle_t const& handle,
                                                            serializer::device_byte_it it_dev_src,
                                                            size_t size) const;

template rmm::device_uvector<double> serializer::unserialize(raft::handle_t const& handle,
                                                             serializer::device_byte_it it_dev_src,
                                                             size_t size) const;

}  // namespace serializer
}  // namespace cugraph
