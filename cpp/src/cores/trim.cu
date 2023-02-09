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
       bool do_expensive_check)
{

  
  auto in_degree = graph_view.compute_in_degree();
  auto out_degree = graph_view.compute_out_degree();
 
  rmm::device_uvector<vertex_t> remaining_vertices(graph_view.local_vertex_partition_range_size(),  handle.get_stream());

  remaining_vertices.resize(
    thrust::distance( remaining_vertices.begin(), thrust::copy_if(
        handle.get_thrust_policy(), thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),  remaining_vertices.begin(),
        [ in_degree ] __device__(
          auto in_degree) { return in_degree <= 1; })),
    handle.get_stream());

  return remaining_vertices;
}
}
