/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>

#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

struct WCC_Usecase {
  bool validate_results{true};
};

template <typename input_usecase_t>
class Tests_WCC : public ::testing::TestWithParam<std::tuple<WCC_Usecase, input_usecase_t>> {
 public:
  Tests_WCC() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> weakly_cc_time;

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(WCC_Usecase const& param, input_usecase_t const& input_usecase)
  {
    raft::handle_t handle{};

    cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);

    std::cout << "calling construct_graph" << std::endl;

    std::tie(graph, std::ignore) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, false);

    std::cout << "back from construct_graph" << std::endl;

    auto graph_view = graph.view();

    rmm::device_uvector<vertex_t> component_labels_v(graph_view.number_of_vertices(),
                                                     handle.get_stream());

    // cugraph::weakly_connected_components(handle, graph_view, component_labels_v.begin());

    // TODO: validate result
  }
};

using Tests_WCC_File      = Tests_WCC<cugraph::test::File_Usecase>;
using Tests_WCC_Rmat      = Tests_WCC<cugraph::test::Rmat_Usecase>;
using Tests_WCC_PathGraph = Tests_WCC<cugraph::test::PathGraph_Usecase>;

TEST_P(Tests_WCC_File, WCC)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}
TEST_P(Tests_WCC_Rmat, WCC)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}
TEST_P(Tests_WCC_PathGraph, WCC)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_WCC_File,
  ::testing::Values(
    std::make_tuple(WCC_Usecase{}, cugraph::test::File_Usecase("test/datasets/dolphins.mtx")),
    std::make_tuple(WCC_Usecase{}, cugraph::test::File_Usecase("test/datasets/coPapersDBLP.mtx")),
    std::make_tuple(WCC_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/coPapersCiteseer.mtx")),
    std::make_tuple(WCC_Usecase{}, cugraph::test::File_Usecase("test/datasets/hollywood.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  path_graph_test,
  Tests_WCC_PathGraph,
  ::testing::Values(std::make_tuple(WCC_Usecase{},
                                    cugraph::test::PathGraph_Usecase(
                                      std::vector<std::tuple<size_t, size_t>>({{1000, 0}}))),
                    std::make_tuple(WCC_Usecase{},
                                    cugraph::test::PathGraph_Usecase(
                                      std::vector<std::tuple<size_t, size_t>>({{100000, 0}})))));

CUGRAPH_TEST_PROGRAM_MAIN()
