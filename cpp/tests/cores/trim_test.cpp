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

struct TRIM_Usecase {
  size_t source{0};
  bool check_correctness{true};
};
template <typename input_usecase_t>
class Tests_TRIM : public ::testing::TestWithParam<std::tuple<TRIM_Usecase, input_usecase_t>> {
 public:
  Tests_TRIM() {}
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template<typename vertex_t, typename edge_t, typename weight_t>
  std::vector<edge_t> run_current_test(edge_t const* offsets,
                               vertex_t const* indices,
                               vertex_t num_vertices, input_usecase_t const& input_usecase){

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
  constexpr bool renumber              = true;

  raft::handle_t handle;

  //cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);

  auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber, true, true);

  auto graph_view = graph.view();
  auto [remain, destination, edge_weitght] = trim(handle, graph_view);
  auto in_degrees = remain.compute_in_degree();
  auto out_degrees = remain.compute_out_degree();
  auto min_in_degree = thrust::reduce(in_degrees.begin(), in_degrees.end(), thrust::minimum<vertex_t>{});
  auto min_out_degree = thrust::reduce(out_degrees.begin(), out_degrees.end(), thrust::minimum<vertex_t>{});

  ASSERT_TRUE(min_in_degree > 1)<< "Triming remain indegree = 1";

  ASSERT_TRUE(min_out_degree > 1)<< "Triming remain indegree = 1";



  }
}; 

using Tests_TRIM_File = Tests_TRIM<cugraph::test::File_Usecase>;
INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_TRIM_File,
  // enable correctness checks
  ::testing::Values(
    std::make_tuple(TRIM_Usecase{0}, cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(TRIM_Usecase{0}, cugraph::test::File_Usecase("test/datasets/dblp.mtx")),
    std::make_tuple(TRIM_Usecase{1000}, cugraph::test::File_Usecase("test/datasets/wiki2003.mtx"))));

CUGRAPH_TEST_PROGRAM_MAIN()
