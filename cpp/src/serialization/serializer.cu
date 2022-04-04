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

#include <cugraph/serialization/serializer.hpp>

#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>

#include <thrust/copy.h>

#include <type_traits>

namespace cugraph {
namespace serializer {

template <typename value_t>
void serializer_t::serialize(value_t val)
{
  auto byte_buff_sz = sizeof(value_t);
  auto it_end       = begin_ + byte_buff_sz;

  raft::update_device(
    begin_, reinterpret_cast<byte_t const*>(&val), byte_buff_sz, handle_.get_stream());

  begin_ = it_end;
}

template <typename value_t>
value_t serializer_t::unserialize(void)
{
  value_t val{};
  auto byte_buff_sz = sizeof(value_t);

  raft::update_host(&val, reinterpret_cast<value_t const*>(cbegin_), 1, handle_.get_stream());

  cbegin_ += byte_buff_sz;
  return val;
}

template <typename value_t>
void serializer_t::serialize(value_t const* p_d_src, size_t size)
{
  auto byte_buff_sz       = size * sizeof(value_t);
  auto it_end             = begin_ + byte_buff_sz;
  byte_t const* byte_buff = reinterpret_cast<byte_t const*>(p_d_src);

  thrust::copy_n(handle_.get_thrust_policy(), byte_buff, byte_buff_sz, begin_);

  begin_ = it_end;
}

template <typename value_t>
rmm::device_uvector<value_t> serializer_t::unserialize(size_t size)
{
  auto byte_buff_sz = size * sizeof(value_t);
  rmm::device_uvector<value_t> d_dest(size, handle_.get_stream());
  byte_t* byte_buff = reinterpret_cast<byte_t*>(d_dest.data());

  thrust::copy_n(handle_.get_thrust_policy(), cbegin_, byte_buff_sz, byte_buff);

  cbegin_ += byte_buff_sz;
  return d_dest;
}

// serialization of graph metadata, via device orchestration:
//
template <typename graph_t>
void serializer_t::serialize(serializer_t::graph_meta_t<graph_t> const& gmeta)
{
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  if constexpr (!graph_t::is_multi_gpu) {
    using bool_t = typename graph_meta_t<graph_t>::bool_ser_t;

    serialize(gmeta.num_vertices_);
    serialize(gmeta.num_edges_);
    serialize(static_cast<bool_t>(gmeta.properties_.is_symmetric));
    serialize(static_cast<bool_t>(gmeta.properties_.is_multigraph));
    serialize(static_cast<bool_t>(gmeta.is_weighted_));

    auto seg_off_sz_bytes =
      (gmeta.segment_offsets_ ? (*(gmeta.segment_offsets_)).size() : size_t{0}) * sizeof(vertex_t);
    if (seg_off_sz_bytes > 0) {
      auto it_end = begin_ + seg_off_sz_bytes;

      raft::update_device(begin_,
                          reinterpret_cast<byte_t const*>((*(gmeta.segment_offsets_)).data()),
                          seg_off_sz_bytes,
                          handle_.get_stream());

      begin_ = it_end;
    }

  } else {
    CUGRAPH_FAIL("Unsupported graph type for serialization.");
  }
}

// unserialization of graph metadata, via device orchestration:
//
template <typename graph_t>
serializer_t::graph_meta_t<graph_t> serializer_t::unserialize(
  size_t graph_meta_sz_bytes,
  serializer_t::graph_meta_t<graph_t> const& empty_meta)  // tag dispatching parameter
{
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  if constexpr (!graph_t::is_multi_gpu) {
    using bool_t = typename graph_meta_t<graph_t>::bool_ser_t;

    CUGRAPH_EXPECTS(graph_meta_sz_bytes >= 2 * sizeof(size_t) + 3 * sizeof(bool_t),
                    "Un/serialization meta size mismatch.");

    size_t num_vertices  = unserialize<size_t>();
    size_t num_edges     = unserialize<size_t>();
    bool_t is_symmetric  = unserialize<bool_t>();
    bool_t is_multigraph = unserialize<bool_t>();
    bool_t is_weighted   = unserialize<bool_t>();

    graph_properties_t properties{static_cast<bool>(is_symmetric),
                                  static_cast<bool>(is_multigraph)};

    std::optional<std::vector<vertex_t>> segment_offsets{std::nullopt};

    size_t seg_off_sz_bytes = graph_meta_sz_bytes - 2 * sizeof(size_t) - 3 * sizeof(bool_t);

    if (seg_off_sz_bytes > 0) {
      segment_offsets = std::vector<vertex_t>(seg_off_sz_bytes / sizeof(vertex_t), vertex_t{0});
      raft::update_host((*segment_offsets).data(),
                        reinterpret_cast<vertex_t const*>(cbegin_),
                        seg_off_sz_bytes,
                        handle_.get_stream());

      cbegin_ += seg_off_sz_bytes;
    }

    return graph_meta_t<graph_t>{
      num_vertices, num_edges, properties, static_cast<bool>(is_weighted), segment_offsets};

  } else {
    CUGRAPH_FAIL("Unsupported graph type for unserialization.");
    return graph_meta_t<graph_t>{};
  }
}

// graph serialization:
// metadata argument (gvmeta) can be used for checking / testing;
//
template <typename graph_t>
void serializer_t::serialize(graph_t const& graph, serializer_t::graph_meta_t<graph_t>& gvmeta)
{
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  if constexpr (!graph_t::is_multi_gpu) {
    size_t num_vertices = graph.number_of_vertices();
    size_t num_edges    = graph.number_of_edges();
    auto&& gview        = graph.view();

    gvmeta = graph_meta_t<graph_t>{graph};

    auto offsets = gview.local_edge_partition_view().offsets();
    auto indices = gview.local_edge_partition_view().indices();
    auto weights = gview.local_edge_partition_view().weights();

    // FIXME: remove when host_bcast() becomes available for vectors;
    //
    // for now, this must come first, because unserialize()
    // needs it at the beginning to extract graph metadata
    // to be able to finish the rest of the graph unserialization;
    //
    serialize(gvmeta);

    serialize(offsets, num_vertices + 1);
    serialize(indices, num_edges);

    if (weights) serialize(*weights, num_edges);

  } else {
    CUGRAPH_FAIL("Unsupported graph type for serialization.");
  }
}

// graph unserialization:
//
template <typename graph_t>
graph_t serializer_t::unserialize(size_t device_sz_bytes, size_t host_sz_bytes)
{
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  if constexpr (!graph_t::is_multi_gpu) {
    graph_meta_t<graph_t> empty_meta{};  // tag-dispatching only

    // FIXME: remove when host_bcast() becomes available for vectors;
    //
    // for now, this must come first, because unserialize()
    // needs it at the beginning to extract graph metadata
    // to be able to finish the rest of the graph unserialization;
    //
    auto gvmeta = unserialize(host_sz_bytes, empty_meta);

    auto pair_sz = get_device_graph_sz_bytes(gvmeta);

    CUGRAPH_EXPECTS((pair_sz.first == device_sz_bytes) && (pair_sz.second == host_sz_bytes),
                    "Un/serialization size mismatch.");

    vertex_t num_vertices = gvmeta.num_vertices_;
    edge_t num_edges      = gvmeta.num_edges_;
    auto g_props          = gvmeta.properties_;
    auto is_weighted      = gvmeta.is_weighted_;
    auto seg_offsets      = gvmeta.segment_offsets_;

    auto d_offsets = unserialize<edge_t>(num_vertices + 1);
    auto d_indices = unserialize<vertex_t>(num_edges);

    return graph_t(
      handle_,
      num_vertices,
      num_edges,
      g_props,
      std::move(d_offsets),
      std::move(d_indices),
      is_weighted ? std::optional<rmm::device_uvector<weight_t>>{unserialize<weight_t>(num_edges)}
                  : std::nullopt,
      std::move(seg_offsets));  // RVO-ed
  } else {
    CUGRAPH_FAIL("Unsupported graph type for unserialization.");

    return graph_t{handle_};
  }
}

// Manual template instantiations (EIDir's):
//
template void serializer_t::serialize(int32_t const* p_d_src, size_t size);
template void serializer_t::serialize(int64_t const* p_d_src, size_t size);
template void serializer_t::serialize(float const* p_d_src, size_t size);
template void serializer_t::serialize(double const* p_d_src, size_t size);

template rmm::device_uvector<int32_t> serializer_t::unserialize(size_t size);
template rmm::device_uvector<int64_t> serializer_t::unserialize(size_t size);
template rmm::device_uvector<float> serializer_t::unserialize(size_t size);
template rmm::device_uvector<double> serializer_t::unserialize(size_t size);

// serialize graph:
//
template void serializer_t::serialize(
  graph_t<int32_t, int32_t, float, false, false> const& graph,
  serializer_t::graph_meta_t<graph_t<int32_t, int32_t, float, false, false>>&);

template void serializer_t::serialize(
  graph_t<int32_t, int64_t, float, false, false> const& graph,
  serializer_t::graph_meta_t<graph_t<int32_t, int64_t, float, false, false>>&);

template void serializer_t::serialize(
  graph_t<int64_t, int64_t, float, false, false> const& graph,
  serializer_t::graph_meta_t<graph_t<int64_t, int64_t, float, false, false>>&);

template void serializer_t::serialize(
  graph_t<int32_t, int32_t, double, false, false> const& graph,
  serializer_t::graph_meta_t<graph_t<int32_t, int32_t, double, false, false>>&);

template void serializer_t::serialize(
  graph_t<int32_t, int64_t, double, false, false> const& graph,
  serializer_t::graph_meta_t<graph_t<int32_t, int64_t, double, false, false>>&);

template void serializer_t::serialize(
  graph_t<int64_t, int64_t, double, false, false> const& graph,
  serializer_t::graph_meta_t<graph_t<int64_t, int64_t, double, false, false>>&);

// unserialize graph:
//
template graph_t<int32_t, int32_t, float, false, false> serializer_t::unserialize(size_t, size_t);

template graph_t<int32_t, int64_t, float, false, false> serializer_t::unserialize(size_t, size_t);

template graph_t<int64_t, int64_t, float, false, false> serializer_t::unserialize(size_t, size_t);

template graph_t<int32_t, int32_t, double, false, false> serializer_t::unserialize(size_t, size_t);

template graph_t<int32_t, int64_t, double, false, false> serializer_t::unserialize(size_t, size_t);

template graph_t<int64_t, int64_t, double, false, false> serializer_t::unserialize(size_t, size_t);

}  // namespace serializer
}  // namespace cugraph
