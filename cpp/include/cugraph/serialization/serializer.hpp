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

#include <cugraph/experimental/graph.hpp>

#include <rmm/device_uvector.hpp>

#include <raft/handle.hpp>

#include <memory>
#include <vector>

namespace cugraph {
namespace serializer {

using namespace cugraph::experimental;

class serializer_t {
 public:
  using byte_t = uint8_t;

  using device_byte_it  = typename rmm::device_uvector<byte_t>::iterator;
  using device_byte_cit = typename rmm::device_uvector<byte_t>::const_iterator;

  // cnstr. for serialize() path:
  //
  serializer_t(raft::handle_t const& handle, size_t total_sz_bytes)
    : handle_(handle),
      d_storage_(total_sz_bytes, handle.get_stream()),
      begin_(d_storage_.begin()),
      cbegin_(d_storage_.begin())
  {
  }

  // cnstr. for unserialize() path:
  //
  serializer_t(raft::handle_t const& handle, byte_t const* ptr_d_storage)
    : handle_(handle), d_storage_(0, handle.get_stream()), cbegin_(ptr_d_storage)
  {
  }

  template <typename graph_t, typename Enable = void>
  struct graph_meta_t;

  template <typename graph_t>
  struct graph_meta_t<graph_t, std::enable_if_t<graph_t::is_multi_gpu>> {
    // purposely empty, for now;
    // FIXME: provide implementation for multi-gpu version
  };

  template <typename graph_t>
  struct graph_meta_t<graph_t, std::enable_if_t<!graph_t::is_multi_gpu>> {
    using vertex_t   = typename graph_t::vertex_type;
    using bool_ser_t = uint8_t;

    explicit graph_meta_t(graph_t const& graph)
      : num_vertices_(graph.get_number_of_vertices()),
        num_edges_(graph.get_number_of_edges()),
        properties_(graph.get_graph_properties()),
        segment_offsets_(graph.view().get_local_adj_matrix_partition_segment_offsets(0))
    {
    }

    size_t num_vertices_;
    size_t num_edges_;
    graph_properties_t properties_{};
    std::vector<vertex_t> segment_offsets_{};

    size_t get_device_sz_bytes(void) const
    {
      return (num_vertices_ + num_edges_) * sizeof(size_t) +
             segment_offsets_.size() * sizeof(vertex_t) + 3 * sizeof(bool_ser_t);
    }
  };

  // device array serialization:
  //
  template <typename value_t>
  void serialize(value_t const* p_d_src, size_t size);

  // device vector unserialization;
  // extracts device_uvector of `size` bytes_to_value_t elements:
  //
  template <typename value_t>
  rmm::device_uvector<value_t> unserialize(
    size_t size);  // size of device vector to be unserialized

  // more complex object (e.g., graph) serialization,
  // with device storage and host metadata:
  // (associated with target; e.g., num_vertices, etc.)
  //
  template <typename graph_t>
  graph_meta_t<graph_t> serialize(graph_t const& gview);  // serialization target

  // more complex object (e.g., graph) unserialization,
  // with device storage and host metadata:
  // (associated with target; e.g., num_vertices, etc.)
  //
  template <typename graph_t>
  graph_t unserialize(graph_meta_t<graph_t> const& meta);

  template <typename graph_t>
  static size_t get_device_graph_sz_bytes(graph_t const& graph)
  {
    using vertex_t = typename graph_t::vertex_type;
    using edge_t   = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;

    if constexpr (!graph_t::is_multi_gpu) {
      size_t num_edges     = graph.get_number_of_edges();
      size_t device_ser_sz = (graph.get_number_of_vertices() + 1) * sizeof(edge_t) +
                             num_edges * sizeof(vertex_t) + num_edges * sizeof(weight_t);

      return device_ser_sz;
    } else {
      CUGRAPH_FAIL("Unsupported graph type for un/serialization.");

      return 0;
    }
  }

  template <typename graph_t>
  static size_t get_device_graph_sz_bytes(graph_meta_t<graph_t> const& graph_meta)
  {
    using vertex_t = typename graph_t::vertex_type;
    using edge_t   = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;

    if constexpr (!graph_t::is_multi_gpu) {
      size_t num_vertices = graph_meta.num_vertices_;
      size_t num_edges    = graph_meta.num_edges_;

      size_t device_ser_sz = (num_vertices + 1) * sizeof(edge_t) + num_edges * sizeof(vertex_t) +
                             num_edges * sizeof(weight_t);

      return device_ser_sz;
    } else {
      CUGRAPH_FAIL("Unsupported graph type for un/serialization.");

      return 0;
    }
  }

 private:
  raft::handle_t const& handle_;
  rmm::device_uvector<byte_t> d_storage_;
  device_byte_it begin_{nullptr};    // advances on serialize()
  device_byte_cit cbegin_{nullptr};  // advances on unserialize()
};

}  // namespace serializer
}  // namespace cugraph
