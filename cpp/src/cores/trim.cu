#pragma once


#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
trim(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
       std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
       bool do_expensive_check)
{


  rmm::device_uvector<vertex_t> remaining_vertices(graph_view.local_vertex_partition_range_size(),  handle.get_stream());
  remaining_vertices.resize(
    thrust::distance(
      remaining_vertices.begin(),
      thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
        remaining_vertices.begin(),
        [core_numbers, v_first = graph_view.local_vertex_partition_range_first()] __device__(
          auto v) { return core_numbers[v - v_first] > edge_t{0}; })),
    handle.get_stream());

  thrust::for_each(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [k_first, core_numbers, v_first = graph_view.local_vertex_partition_range_first()] __device__(
        auto v) {
        if (core_numbers[v - v_first] < k_first) { core_numbers[v - v_first] = edge_t{0}; }
  });
  return remaining_vertices;
}
