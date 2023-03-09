#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

template<typename vertex_t, typename edge_t>
std::vector<edge_t> trim_tests(edge_t const* offsets,
                               vertex_t const* indices,
                               vertex_t num_vertices){

  std::vector<bool> edge_valids(offsets[num_vertices], true);

  for (vertex_t i = 0; i < num_vertices; ++i) {
    for (edge_t j = offsets[i]; j < offsets[i + 1]; j++) {
      if (indices[j] == i) {
        edge_valids[j] = false;
      } else if ((j > offsets[i]) && (indices[j] == indices[j - 1])) {
        edge_valids[j] = false;
      }
    }
  }


  raft::handle_t handle;
  graph_view_t<vertex_t, edge_t, false, false> graph_view; 
  auto remain = cugraph::trim(handle, graph_view);

  auto in_degrees = remain.compute_in_degree();
  auto out_degrees = remain.compute_out_degree();
  auto min_in_degree = thrust::reduce(in_degrees.begin(), in_degrees.end(), thrust::minimum<vertex_t>{});
  auto min_out_degree = thrust::reduce(out_degrees.begin(), out_degrees.end(), thrust::minimum<vertex_t>{});

  ASSERT_TRUE(min_in_degree>0)<< "Triming remain indegree = 0";

  ASSERT_TRUE(min_out_degree>0)<< "Triming remain indegree = 0";



} 

