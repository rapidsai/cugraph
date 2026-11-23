/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Graph construction: owning COO producers, the explicit copying path, and create_graph().
 */
#include "cugraph_python.hpp"

#include <raft/util/cudart_utils.hpp>

#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cugraph_python {
namespace {

template <typename vertex_t>
__global__ void fill_cycle(vertex_t* sources, vertex_t* destinations, vertex_t n)
{
  auto i = static_cast<vertex_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    sources[i]      = i;
    destinations[i] = (i + 1 == n) ? vertex_t{0} : i + 1;
  }
}

std::uint64_t next_graph_identity()
{
  static std::atomic<std::uint64_t> counter{1};
  return counter.fetch_add(1);
}

template <typename value_t>
cugraph::arithmetic_device_uvector_t copy_property_buffer(raft::handle_t const& handle,
                                                          value_t const* values,
                                                          std::size_t num_edges)
{
  rmm::device_uvector<value_t> copied(num_edges, handle.get_stream());
  raft::copy(copied.data(), values, num_edges, handle.get_stream());
  return cugraph::arithmetic_device_uvector_t{std::move(copied)};
}

cugraph::arithmetic_device_uvector_t copy_property_input(raft::handle_t const& handle,
                                                         device_property_input_t const& input,
                                                         std::size_t num_edges)
{
  if (input.data == nullptr) {
    throw std::invalid_argument("copy_from_arrays: an edge property array is null");
  }
  switch (input.dtype) {
    case arithmetic_type_id_t::int32:
      return copy_property_buffer(handle, static_cast<std::int32_t const*>(input.data), num_edges);
    case arithmetic_type_id_t::int64:
      return copy_property_buffer(handle, static_cast<std::int64_t const*>(input.data), num_edges);
    case arithmetic_type_id_t::float32:
      return copy_property_buffer(handle, static_cast<float const*>(input.data), num_edges);
    case arithmetic_type_id_t::float64:
      return copy_property_buffer(handle, static_cast<double const*>(input.data), num_edges);
  }
  throw std::invalid_argument("copy_from_arrays: unsupported edge property dtype");
}

template <typename vertex_t>
std::unique_ptr<owned_coo_t> copy_from_arrays_impl(
  std::shared_ptr<context_t> context,
  vertex_t const* sources,
  vertex_t const* destinations,
  std::size_t num_edges,
  std::vector<device_property_input_t> const& properties)
{
  // The caller must have synchronized the producing stream. The source arrays are only read,
  // and remain valid after this call.
  auto stream = context->handle.get_stream();

  rmm::device_uvector<vertex_t> src(num_edges, stream);
  rmm::device_uvector<vertex_t> dst(num_edges, stream);
  raft::copy(src.data(), sources, num_edges, stream);
  raft::copy(dst.data(), destinations, num_edges, stream);

  std::vector<cugraph::arithmetic_device_uvector_t> edge_properties;
  edge_properties.reserve(properties.size());
  for (auto const& property : properties) {
    edge_properties.push_back(copy_property_input(context->handle, property, num_edges));
  }

  context->handle.sync_stream();
  return std::make_unique<owned_coo_t>(owned_coo_instance_t<vertex_t>(
    std::move(context), std::move(src), std::move(dst), std::move(edge_properties)));
}

// Selects graph_instance_t<vertex_t, vertex_t, store_transposed, multi_gpu>. Both orientations are
// compiled for single-GPU. multi_gpu=true is the extension point. The two orientations are
// different types, so each arm wraps its instance before returning.
template <typename vertex_t, typename functor_t>
std::tuple<std::unique_ptr<graph_t>, std::unique_ptr<edge_properties_t>>
dispatch_graph(bool store_transposed, bool multi_gpu, functor_t& functor)
{
  if (multi_gpu) {
    throw std::invalid_argument("create_graph: multi_gpu=true is not instantiated");
  }
  if (store_transposed) {
    auto [instance, properties] = functor.template operator()<vertex_t, vertex_t, true, false>();
    return {std::make_unique<graph_t>(std::move(instance)),
            std::make_unique<edge_properties_t>(std::move(properties))};
  }
  auto [instance, properties] = functor.template operator()<vertex_t, vertex_t, false, false>();
  return {std::make_unique<graph_t>(std::move(instance)),
          std::make_unique<edge_properties_t>(std::move(properties))};
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::pair<graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu>,
          std::shared_ptr<edge_properties_state_t<edge_t>>>
create_graph_impl(owned_coo_instance_t<vertex_t>& coo)
{
  // take() marks the COO consumed before any buffer moves. If anything below throws, the COO
  // stays consumed and cannot be retried.
  auto [sources, destinations, edge_properties] = coo.take();
  auto context                                  = coo.context();
  auto& handle                                  = context->handle;

  auto [native, properties, number_map] =
    cugraph::create_graph<vertex_t, edge_t, store_transposed, multi_gpu>(
      handle,
      std::nullopt,
      std::move(sources),
      std::move(destinations),
      std::move(edge_properties),
      cugraph::graph_properties_t{false, false},
      /*renumber=*/true,
      std::nullopt,
      std::nullopt,
      /*do_expensive_check=*/false);

  // Renumbering is always on in this prototype, so the number map is always present.
  std::optional<rmm::device_uvector<vertex_t>> owned_number_map{};
  if (number_map) { owned_number_map = number_map->release_buffer(); }

  handle.sync_stream();

  auto identity = next_graph_identity();
  auto properties_state = std::make_shared<edge_properties_state_t<edge_t>>();
  properties_state->context = context;
  properties_state->values = std::move(properties);
  properties_state->graph_id = identity;

  return {graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu>{
            std::move(context),
            std::move(native),
            std::move(owned_number_map),
            identity,
            identity,
            /*version=*/0},
          std::move(properties_state)};
}

template <typename vertex_t>
struct build_graph_t {
  owned_coo_instance_t<vertex_t>& coo;

  template <typename vertex_type_t, typename edge_t, bool store_transposed, bool multi_gpu>
  auto operator()()
  {
    static_assert(std::is_same_v<vertex_type_t, vertex_t>);
    return create_graph_impl<vertex_type_t, edge_t, store_transposed, multi_gpu>(coo);
  }
};

}  // namespace

std::unique_ptr<owned_coo_t> make_cycle_coo(std::shared_ptr<context_t> context, std::int64_t size)
{
  using vertex_t = std::int32_t;
  if (size < 2 || size > std::numeric_limits<vertex_t>::max()) {
    throw std::invalid_argument("make_cycle_coo: size must be in [2, INT32_MAX]");
  }
  auto n      = static_cast<vertex_t>(size);
  auto stream = context->handle.get_stream();

  rmm::device_uvector<vertex_t> sources(n, stream);
  rmm::device_uvector<vertex_t> destinations(n, stream);

  constexpr unsigned int block = 256;
  auto grid = static_cast<unsigned int>((static_cast<std::size_t>(n) + block - 1) / block);
  fill_cycle<vertex_t><<<grid, block, 0, stream.get()>>>(sources.data(), destinations.data(), n);
  RAFT_CHECK_CUDA(stream.get());

  context->handle.sync_stream();
  return std::make_unique<owned_coo_t>(
    owned_coo_instance_t<vertex_t>(std::move(context), std::move(sources), std::move(destinations)));
}

std::unique_ptr<owned_coo_t> copy_from_arrays(
  std::shared_ptr<context_t> context,
  device_property_input_t sources,
  device_property_input_t destinations,
  std::size_t num_edges,
  std::vector<device_property_input_t> const& edge_properties)
{
  if (sources.data == nullptr || destinations.data == nullptr) {
    throw std::invalid_argument("copy_from_arrays: sources and destinations must not be null");
  }
  if (sources.dtype != destinations.dtype) {
    throw std::invalid_argument("copy_from_arrays: sources and destinations differ in dtype");
  }
  switch (sources.dtype) {
    case arithmetic_type_id_t::int32:
      return copy_from_arrays_impl<std::int32_t>(std::move(context),
                                                 static_cast<std::int32_t const*>(sources.data),
                                                 static_cast<std::int32_t const*>(destinations.data),
                                                 num_edges,
                                                 edge_properties);
    case arithmetic_type_id_t::int64:
      return copy_from_arrays_impl<std::int64_t>(std::move(context),
                                                 static_cast<std::int64_t const*>(sources.data),
                                                 static_cast<std::int64_t const*>(destinations.data),
                                                 num_edges,
                                                 edge_properties);
    case arithmetic_type_id_t::float32:
    case arithmetic_type_id_t::float64:
      break;
  }
  throw std::invalid_argument("copy_from_arrays: endpoints must be int32 or int64");
}

std::tuple<std::unique_ptr<graph_t>, std::unique_ptr<edge_properties_t>>
create_graph(owned_coo_t& coo, bool store_transposed, bool multi_gpu)
{
  return std::visit(
    [&](auto& instance) -> std::tuple<std::unique_ptr<graph_t>, std::unique_ptr<edge_properties_t>> {
      using vertex_t = typename std::decay_t<decltype(instance)>::vertex_type;
      build_graph_t<vertex_t> builder{instance};
      return dispatch_graph<vertex_t>(store_transposed, multi_gpu, builder);
    },
    coo.impl);
}

}  // namespace cugraph_python
