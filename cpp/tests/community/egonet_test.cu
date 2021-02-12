/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <experimental/graph.hpp>
#include <experimental/graph_view.hpp>
#include <graph.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <algorithm>
#include <tuple>
#include <vector>

typedef struct InducedEgo_Usecase_t {
  std::string graph_file_full_path{};
  std::vector<int32_t> ego_sources{};
  int32_t radius;
  bool test_weighted{false};

  InducedEgo_Usecase_t(std::string const& graph_file_path,
                       std::vector<int32_t> const& ego_sources,
                       int32_t radius,
                       bool test_weighted)
    : ego_sources(ego_sources), radius(radius), test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} InducedEgo_Usecase;

class Tests_InducedEgo : public ::testing::TestWithParam<InducedEgo_Usecase> {
 public:
  Tests_InducedEgo() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(InducedEgo_Usecase const& configuration)
  {
    int n_streams = 0;
    if (configuration.ego_sources.size() > 1)
      n_streams = std::min(configuration.ego_sources.size(), 128);
    raft::handle_t handle(n_streams);

    auto graph = cugraph::test::
      read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed>(
        handle, configuration.graph_file_full_path, configuration.test_weighted);
    auto graph_view = graph.view();

    rmm::device_uvector<vertex_t> d_ego_sources(configuration.ego_sources.size(),
                                                handle.get_stream());

    raft::update_device(d_ego_sources.data(),
                        configuration.ego_sources.data(),
                        configuration.ego_sources.size(),
                        handle.get_stream());

    rmm::device_uvector<vertex_t> d_ego_edgelist_src(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_ego_edgelist_dst(0, handle.get_stream());
    rmm::device_uvector<weight_t> d_ego_edgelist_weights(0, handle.get_stream());
    rmm::device_uvector<size_t> d_ego_edge_offsets(0, handle.get_stream());

    std::tie(d_ego_edgelist_src, d_ego_edgelist_dst, d_ego_edgelist_weights, d_ego_edge_offsets) =
      cugraph::experimental::extract_ego(handle,
                                         graph_view,
                                         d_ego_sources.data(),
                                         static_cast<vertex_t>(configuration.ego_sources.size()),
                                         configuration.radius);

    std::vector<size_t> h_cugraph_ego_edge_offsets(d_ego_edge_offsets.size());
    std::vector<vertex_t> h_cugraph_ego_edgelist_src(d_ego_edgelist_src.size());
    std::vector<vertex_t> h_cugraph_ego_edgelist_dst(d_ego_edgelist_dst.size());
    raft::update_host(h_cugraph_ego_edgelist_src.data(),
                      d_ego_edgelist_src.data(),
                      d_ego_edgelist_src.size(),
                      handle.get_stream());
    raft::update_host(h_cugraph_ego_edgelist_dst.data(),
                      d_ego_edgelist_dst.data(),
                      d_ego_edgelist_dst.size(),
                      handle.get_stream());
    raft::update_host(h_cugraph_ego_edge_offsets.data(),
                      d_ego_edge_offsets.data(),
                      d_ego_edge_offsets.size(),
                      handle.get_stream());
    ASSERT_TRUE(d_ego_edge_offsets.size() == (configuration.ego_sources.size() + 1));
    ASSERT_TRUE(d_ego_edgelist_src.size() == d_ego_edgelist_dst.size());
    if (configuration.test_weighted)
      ASSERT_TRUE(d_ego_edgelist_src.size() == d_ego_edgelist_weights.size());
    ASSERT_TRUE(h_cugraph_ego_edge_offsets[configuration.ego_sources.size()] ==
                d_ego_edgelist_src.size());
    for (size_t i = 0; i < configuration.ego_sources.size(); i++)
      ASSERT_TRUE(h_cugraph_ego_edge_offsets[i] < h_cugraph_ego_edge_offsets[i + 1]);
    auto n_vertices = graph_view.get_number_of_vertices();
    for (size_t i = 0; i < d_ego_edgelist_src.size(); i++) {
      ASSERT_TRUE(h_cugraph_ego_edgelist_src[i] >= 0);
      ASSERT_TRUE(h_cugraph_ego_edgelist_src[i] < n_vertices);
      ASSERT_TRUE(h_cugraph_ego_edgelist_dst[i] >= 0);
      ASSERT_TRUE(h_cugraph_ego_edgelist_dst[i] < n_vertices);
    }

    /*
    // For inspecting data
    std::vector<weight_t> h_cugraph_ego_edgelist_weights(d_ego_edgelist_weights.size());
    if (configuration.test_weighted) {
      raft::update_host(h_cugraph_ego_edgelist_weights.data(),
                        d_ego_edgelist_weights.data(),
                        d_ego_edgelist_weights.size(),
                        handle.get_stream());
    }
    raft::print_host_vector("offsets",
                            &h_cugraph_ego_edge_offsets[0],
                            h_cugraph_ego_edge_offsets.size(),
                            std::cout);
    raft::print_host_vector("src",
                            &h_cugraph_ego_edgelist_src[0],
                            h_cugraph_ego_edgelist_src.size(),
                            std::cout);
    raft::print_host_vector("dst",
                            &h_cugraph_ego_edgelist_dst[0],
                            h_cugraph_ego_edgelist_dst.size(),
                            std::cout);
    raft::print_host_vector("weights",
                            &h_cugraph_ego_edgelist_weights[0],
                            h_cugraph_ego_edgelist_weights.size(),
                            std::cout);
    */
  }
};

TEST_P(Tests_InducedEgo, CheckInt32Int32FloatUntransposed)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
}

INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_InducedEgo,
  ::testing::Values(
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{0}, 1, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{0}, 2, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{1}, 3, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{10, 0, 5}, 2, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{9, 3, 10}, 2, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{5, 12, 13}, 2, true)));

CUGRAPH_TEST_PROGRAM_MAIN()
