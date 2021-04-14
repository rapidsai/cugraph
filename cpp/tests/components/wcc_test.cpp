/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <components/wcc_graphs.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <experimental/graph.hpp>

#include <raft/cudart_utils.h>
#include <rmm/device_uvector.hpp>

struct WCC_Usecase {
  bool validate_results{true};
};

template <typename input_usecase_t>
class Tests_WCC : public ::testing::TestWithParam<std::tuple<WCC_Usecase, input_usecase_t>> {
 public:
  Tests_WCC() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> weakly_cc_time;

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(WCC_Usecase const& param, input_usecase_t const& input_usecase)
  {
    raft::handle_t handle{};

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);

    std::tie(graph, std::ignore) =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, false, false);

    auto graph_view = graph.view();

    rmm::device_uvector<vertex_t> component_labels_v(graph_view.get_number_of_vertices(),
                                                     handle.get_stream());

    // cugraph::weakly_connected_components(handle, graph_view, component_labels_v.begin());

    // TODO: validate result
  }
};

using Tests_WCC_File      = Tests_WCC<cugraph::test::File_Usecase>;
using Tests_WCC_Rmat      = Tests_WCC<cugraph::test::Rmat_Usecase>;
using Tests_WCC_LineGraph = Tests_WCC<cugraph::test::LineGraph_Usecase>;

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
TEST_P(Tests_WCC_LineGraph, WCC)
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
  line_graph_test,
  Tests_WCC_LineGraph,
  ::testing::Values(std::make_tuple(WCC_Usecase{}, cugraph::test::LineGraph_Usecase(1000)),
                    std::make_tuple(WCC_Usecase{}, cugraph::test::LineGraph_Usecase(100000))));

CUGRAPH_TEST_PROGRAM_MAIN()
