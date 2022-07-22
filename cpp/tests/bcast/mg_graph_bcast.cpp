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
#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#include <gtest/gtest.h>

#include <cugraph/utilities/path_retrieval.hpp>

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_SUITE_P()
//
struct GraphBcast_Usecase {
  std::string graph_file_full_path{};

  // FIXME:  We really should have a Graph_Testparms_Base class or something
  //         like that which can handle this graph_full_path thing.
  //
  explicit GraphBcast_Usecase(std::string const& graph_file_path)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
};

////////////////////////////////////////////////////////////////////////////////
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
class Tests_MGGraphBcast : public ::testing::TestWithParam<GraphBcast_Usecase> {
 public:
  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  // Run once for each test instance
  //
  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of broadcasting a graph,
  // by comparing the graph that was sent (`sg_graph`)
  // with th eone that was received (`graph-copy`):
  //
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_test(const GraphBcast_Usecase& param)
  {
    using sg_graph_t = cugraph::graph_t<vertex_t, edge_t, weight_t, false, false>;

    auto [sg_graph, d_renumber_map_labels] =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        *handle_, param.graph_file_full_path, true, /*renumber=*/false);

    if (handle_->get_comms().get_rank() == 0) {
      cugraph::broadcast::graph_broadcast(*handle_, &sg_graph);
    } else {
      sg_graph_t* g_ignore{nullptr};
      auto graph_copy       = cugraph::broadcast::graph_broadcast(*handle_, g_ignore);
      auto [same, str_fail] = cugraph::test::compare_graphs(*handle_, sg_graph, graph_copy);

      if (!same) std::cerr << "Graph comparison failed on " << str_fail << '\n';

      ASSERT_TRUE(same);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

std::unique_ptr<raft::handle_t> Tests_MGGraphBcast::handle_ = nullptr;

////////////////////////////////////////////////////////////////////////////////
TEST_P(Tests_MGGraphBcast, CheckInt32Int32Float) { run_test<int32_t, int32_t, float>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(simple_test,
                         Tests_MGGraphBcast,
                         ::testing::Values(GraphBcast_Usecase("test/datasets/karate.mtx")
                                           //,GraphBcast_Usecase("test/datasets/smallworld.mtx")
                                           ));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
