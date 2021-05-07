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

#include <experimental/graph.hpp>

#include <rmm/device_uvector.hpp>

#include <raft/handle.hpp>

#include <memory>
#include <vector>

namespace cugraph {
namespace serializer {

class serializer {
 public:
  using byte_t = uint8_t;

  using device_byte_it = typename rmm::device_uvector<byte_t>::iterator;

  // device vector serialization:
  //
  template <typename value_t>
  device_byte_it serialize(
    raft::handle_t const& handle,
    rmm::device_uvector<value_t> const& src,  // serialization target
    device_byte_it it_dev_dest)               // device serialization destination: iterator
                                              // into pre-allocated device byte buffer
    const;                                    // append src_to_bytes to it_dest

  // device vector unserialization:
  //
  template <typename value_t>
  rmm::device_uvector<value_t> unserialize(raft::handle_t const& handle,
                                           device_byte_it it_dev_src,  // unserialization src
                                           size_t size)  // size of device vector to be unserialized
    const;  // extracts device_uvector of `size` bytes_to_value_t elements

  // more complex object (e.g., graph_view) serialization,
  // with device storage and host metadata:
  // (associated with target; e.g., num_vertices, etc.)
  //
  template <typename graph_view_t, typename metadata_t>
  std::pair<device_byte_it, metadata_t> serialize(
    raft::handle_t const& handle,
    graph_view_t const& gview,   // serialization target
    device_byte_it it_dev_dest)  // device serialization destination: iterator into
                                 // pre-allocated device byte buffer
    const;  // serialize more complex object that has both device and metadata host representation

  // more complex object (e.g., graph_view) unserialization,
  // with device storage and host metadata:
  // (associated with target; e.g., num_vertices, etc.)
  //
  template <typename graph_view_t, typename metadata_t>
  std::unique_ptr<graph_view_t> unserialize(
    raft::handle_t const& handle,
    device_byte_it it_dev_src,  // unserialization src
    metadata_t const& meta)     // associated with target; e.g., num_vertices, etc.
    const;
};

namespace detail {
}
}  // namespace serializer
}  // namespace cugraph
