/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

struct TopologicalSort_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTopologicalSort
  : public ::testing::TestWithParam<std::tuple<TopologicalSort_Usecase, input_usecase_t>> {
 public:
  Tests_MGTopologicalSort() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running topological sort on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(TopologicalSort_Usecase const& topological_sort_usecase,
                        input_usecase_t const& input_usecase)
  {
    // TODO: fill in test body
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTopologicalSort<input_usecase_t>::handle_ = nullptr;

using Tests_MGTopologicalSort_File = Tests_MGTopologicalSort<cugraph::test::File_Usecase>;
using Tests_MGTopologicalSort_Rmat = Tests_MGTopologicalSort<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTopologicalSort_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTopologicalSort_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTopologicalSort_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTopologicalSort_File,
  ::testing::Combine(
    ::testing::Values(TopologicalSort_Usecase{false}, TopologicalSort_Usecase{true}),
    ::testing::Values(
      cugraph::test::File_Usecase("test/datasets/dag_small.csv"))));  // TODO: replace with real DAG dataset

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGTopologicalSort_Rmat,
  ::testing::Values(
    std::make_tuple(TopologicalSort_Usecase{false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(TopologicalSort_Usecase{true},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_MGTopologicalSort_Rmat,
  ::testing::Values(
    std::make_tuple(TopologicalSort_Usecase{false, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(TopologicalSort_Usecase{true, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
