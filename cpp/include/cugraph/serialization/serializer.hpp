/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>

#include <rmm/device_uvector.hpp>

#include <raft/handle.hpp>

#include <memory>
#include <vector>

namespace cugraph {
namespace serializer {

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

    graph_meta_t(void) {}

    explicit graph_meta_t(graph_t const& graph)
      : num_vertices_(graph.number_of_vertices()),
        num_edges_(graph.number_of_edges()),
        properties_(graph.graph_properties()),
        is_weighted_(graph.is_weighted()),
        segment_offsets_(graph.view().local_edge_partition_segment_offsets(0))
    {
    }

    graph_meta_t(size_t num_vertices,
                 size_t num_edges,
                 graph_properties_t const& properties,
                 bool is_weighted,
                 std::optional<std::vector<vertex_t>> const& segment_offsets)
      : num_vertices_(num_vertices),
        num_edges_(num_edges),
        properties_(properties),
        is_weighted_(is_weighted),
        segment_offsets_(segment_offsets)
    {
    }

    size_t num_vertices_;
    size_t num_edges_;
    graph_properties_t properties_{};
    bool is_weighted_{};
    std::optional<std::vector<vertex_t>> segment_offsets_{};

    size_t get_device_sz_bytes(void) const
    {
      return 2 * sizeof(size_t) +
             (segment_offsets_ ? (*segment_offsets_).size() : size_t{0}) * sizeof(vertex_t) +
             3 * sizeof(bool_ser_t);
    }
  };

  // POD-type serialization:
  //
  template <typename value_t>
  void serialize(value_t val);

  // POD-type unserialization:
  //
  template <typename value_t>
  value_t unserialize(void);

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

  // graph serialization,
  // with device storage and host metadata:
  // (associated with target; e.g., num_vertices, etc.)
  //
  template <typename graph_t>
  void serialize(graph_t const& graph, graph_meta_t<graph_t>& gmeta);  // serialization target

  // graph unserialization,
  // with device storage and host metadata:
  // (associated with target; e.g., num_vertices, etc.)
  //
  template <typename graph_t>
  graph_t unserialize(size_t device_sz_bytes, size_t host_sz_bytes);

  template <typename graph_t>
  static std::pair<size_t, size_t> get_device_graph_sz_bytes(
    graph_meta_t<graph_t> const& graph_meta)
  {
    using vertex_t = typename graph_t::vertex_type;
    using edge_t   = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;

    if constexpr (!graph_t::is_multi_gpu) {
      size_t num_vertices = graph_meta.num_vertices_;
      size_t num_edges    = graph_meta.num_edges_;

      size_t weight_storage_sz = graph_meta.is_weighted_ ? num_edges * sizeof(weight_t) : 0;

      size_t device_ser_sz =
        (num_vertices + 1) * sizeof(edge_t) + num_edges * sizeof(vertex_t) + weight_storage_sz;

      size_t host_ser_sz = graph_meta.get_device_sz_bytes();

      return std::make_pair(
        device_ser_sz,
        host_ser_sz);  // FIXME: remove when host_bcast() becomes available for host vectors

    } else {
      CUGRAPH_FAIL("Unsupported graph type for un/serialization.");

      return std::pair<size_t, size_t>{};
    }
  }

  template <typename graph_t>
  static std::pair<size_t, size_t> get_device_graph_sz_bytes(graph_t const& graph)
  {
    graph_meta_t<graph_t> gmeta{graph};
    return get_device_graph_sz_bytes(gmeta);
  }

  byte_t const* get_storage(void) const { return d_storage_.begin(); }
  byte_t* get_storage(void) { return d_storage_.begin(); }

 private:
  // serialization of graph metadata, via device orchestration:
  //
  template <typename graph_t>
  void serialize(graph_meta_t<graph_t> const& graph_meta);

  // unserialization of graph metadata, via device orchestration:
  //
  template <typename graph_t>
  graph_meta_t<graph_t> unserialize(
    size_t graph_meta_sz_bytes,
    graph_meta_t<graph_t> const& empty_meta);  // tag dispatching to avoid conflict with
                                               // `unserialize(size_t)` for device vectors

  raft::handle_t const& handle_;
  rmm::device_uvector<byte_t> d_storage_;
  device_byte_it begin_{nullptr};    // advances on serialize()
  device_byte_cit cbegin_{nullptr};  // advances on unserialize()
};

}  // namespace serializer
}  // namespace cugraph
