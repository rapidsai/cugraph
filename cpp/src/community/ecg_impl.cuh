#pragma once
#include <cugraph/algorithms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>

namespace cugraph {

namespace detail {

//   * @return a tuple containing:
//   *    1) Device vector containing clustering result
//   *    2) number of levels of the returned clustering
//   *    3) modularity of the returned clustering

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
std::tuple<rmm::device_uvector<vertex_t>, size_t, weight_t> ecg(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::random::RngState& rng_state,
  weight_t min_weight,
  size_t ensemble_size,
  size_t max_level,
  weight_t threshold,
  weight_t resolution,
  weight_t theta)
{
  rmm::device_uvector<vertex_t> unique_cluster_ids(graph_view.number_of_vertices(),
                                                   handle.get_stream());

  return std::make_tuple(std::move(unique_cluster_ids), max_level, -1.0);
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
std::tuple<rmm::device_uvector<vertex_t>, size_t, weight_t> ecg(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::random::RngState& rng_state,
  weight_t min_weight,
  size_t ensemble_size,
  size_t max_level,
  weight_t threshold,
  weight_t resolution)
{
  return detail::ecg(handle,
                     graph_view,
                     edge_weight_view,
                     rng_state,
                     min_weight,
                     ensemble_size,
                     max_level,
                     threshold,
                     resolution,
                     1.0);
}

}  // namespace cugraph