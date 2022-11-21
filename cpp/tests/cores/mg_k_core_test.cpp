/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

template <typename graph_view_t>
void check_correctness(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  rmm::device_uvector<typename graph_view_t::edge_type> const& d_core_numbers,
  std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
             rmm::device_uvector<typename graph_view_t::vertex_type>,
             std::optional<rmm::device_uvector<typename graph_view_t::weight_type>>> const&
    subgraph,
  size_t k)
{
#if 0
  auto [graph_src, graph_dst, graph_wgt] = cugraph::test::graph_to_host_coo(handle, graph_view);

  auto h_core_numbers = cugraph::test::to_host(handle, d_core_numbers.data(), d_core_number.size());

  auto [d_subgraph_src, d_subgraph_dst, d_subgraph_wgt] = subgraph;

  auto h_subgraph_src =
    cugraph::test::to_host(handle, d_subgraph_src.data(), d_subgraph_src.size());
  auto h_subgraph_dst =
    cugraph::test::to_host(handle, d_subgraph_dst.data(), d_subgraph_dst.size());

  std::optional<std::vector<weight_t>> h_subgraph_wgt{std::nullopt};
  if (d_subgraph_wgt)
    h_subgraph_wgt = std::make_optional(
      cugraph::test::to_host(handle, d_subgraph_wgt->data(), d_subgraph_wgt->size()));

  // Check that all edges in the subgraph are appropriate
  std::for_each(h_subgraph_src.begin(), h_subgraph_src.end(), [&h_core_numbers](auto v) {
    EXPECT_GE(h_core_numbers[v], k);
  });
  std::for_each(h_subgraph_dst.begin(), h_subgraph_dst.end(), [&h_core_numbers](auto v) {
    EXPECT_GE(h_core_numbers[v], k);
  });

  // Now we'll count how many edges should be in the subgraph
  size_t counter = 0;
  for (size_t i = 0; i < graph_src.size(); ++i)
    if ((h_core_numbers[graph_src[i]] >= k) && (h_core_numbers[graph_dst[i]] >= k)) ++counter;

  EXPECT_EQ(counter, h_subgraph.size());

#endif
}

struct KCore_Usecase {
  size_t k;
  cugraph::k_core_degree_type_t degree_type{cugraph::k_core_degree_type_t::OUT};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGKCore : public ::testing::TestWithParam<std::tuple<KCore_Usecase, input_usecase_t>> {
 public:
  Tests_MGKCore() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(std::tuple<KCore_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber              = true;
    auto [k_core_usecase, input_usecase] = param;

    using weight_t = float;

    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [sg_graph, sg_edge_weights, d_sg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        *handle_, input_usecase, false, renumber, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto sg_graph_view = sg_graph.view();
    auto sg_edge_weight_view =
      sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

    rmm::device_uvector<edge_t> d_core_numbers(sg_graph_view.number_of_vertices(),
                                               handle_->get_stream());

    cugraph::core_number(*handle_,
                         sg_graph_view,
                         d_core_numbers.data(),
                         k_core_usecase.degree_type,
                         k_core_usecase.k,
                         k_core_usecase.k);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    raft::device_span<edge_t const> core_number_span{d_core_numbers.data(), d_core_numbers.size()};

#if 0
    auto subgraph = cugraph::k_core(
                                    *handle_, sg_graph_view, k_core_usecase.k, std::nullopt, std::nullopt, std::make_optional(core_number_span));
#else
    EXPECT_THROW(cugraph::k_core(*handle_,
                                 sg_graph_view,
                                 sg_edge_weight_view,
                                 k_core_usecase.k,
                                 std::nullopt,
                                 std::make_optional(core_number_span)),
                 cugraph::logic_error);
#endif

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "Core Number took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (k_core_usecase.check_correctness) {
#if 0
      check_correctness(*handle_, graph_view, d_core_numbers, subgraph, k_core_usecase.k);
#endif
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGKCore<input_usecase_t>::handle_ = nullptr;

using Tests_MGKCore_File = Tests_MGKCore<cugraph::test::File_Usecase>;
using Tests_MGKCore_Rmat = Tests_MGKCore<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGKCore_File, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGKCore_Rmat, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGKCore_Rmat, CheckInt32Int64)
{
  run_current_test<int32_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGKCore_Rmat, CheckInt64Int64)
{
  run_current_test<int64_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGKCore_File,
  ::testing::Combine(
    // enable correctness checks
    testing::Values(KCore_Usecase{3, cugraph::k_core_degree_type_t::IN},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::OUT},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::INOUT}),
    testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx"),
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGKCore_Rmat,
  ::testing::Combine(
    // enable correctness checks
    testing::Values(KCore_Usecase{3, cugraph::k_core_degree_type_t::IN},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::OUT},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::INOUT}),
    testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGKCore_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    testing::Values(KCore_Usecase{3, cugraph::k_core_degree_type_t::OUT, false}),
    testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
