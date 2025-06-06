/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct SSSP_Usecase {
  size_t source{0};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGSSSP : public ::testing::TestWithParam<std::tuple<SSSP_Usecase, input_usecase_t>> {
 public:
  Tests_MGSSSP() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running SSSP on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(SSSP_Usecase const& sssp_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (sssp_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    ASSERT_TRUE(static_cast<vertex_t>(sssp_usecase.source) >= 0 &&
                static_cast<vertex_t>(sssp_usecase.source) < mg_graph_view.number_of_vertices())
      << "Invalid starting source.";

    // 2. run MG SSSP

    rmm::device_uvector<weight_t> d_mg_distances(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle_->get_stream());
    rmm::device_uvector<vertex_t> d_mg_predecessors(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG SSSP.");
    }

    cugraph::sssp(*handle_,
                  mg_graph_view,
                  *mg_edge_weight_view,
                  d_mg_distances.data(),
                  d_mg_predecessors.data(),
                  static_cast<vertex_t>(sssp_usecase.source),
                  std::numeric_limits<weight_t>::max());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. copmare SG & MG results

    if (sssp_usecase.check_correctness) {
      // 3-1. unrenumber & aggregate MG source & results

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_predecessors.data(),
        d_mg_predecessors.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      rmm::device_scalar<vertex_t> d_sg_source(static_cast<vertex_t>(sssp_usecase.source),
                                               handle_->get_stream());

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_sg_source.data(),
        size_t{1},
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      // 3-2. aggregate MG results

      rmm::device_uvector<weight_t> d_mg_aggregate_distances(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_distances) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<weight_t const>(d_mg_distances.data(), d_mg_distances.size()));

      rmm::device_uvector<vertex_t> d_mg_aggregate_predecessors(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_predecessors) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<vertex_t const>(d_mg_predecessors.data(), d_mg_predecessors.size()));

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<cugraph::edge_property_t<edge_t, weight_t>> sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          mg_edge_weight_view,
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-3. run SG SSSP

        auto sg_graph_view = sg_graph.view();
        auto sg_edge_weight_view =
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        rmm::device_uvector<weight_t> d_sg_distances(
          sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
        rmm::device_uvector<vertex_t> d_sg_predecessors(
          sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

        cugraph::sssp(*handle_,
                      sg_graph_view,
                      *sg_edge_weight_view,
                      d_sg_distances.data(),
                      d_sg_predecessors.data(),
                      d_sg_source.value(handle_->get_stream()),
                      std::numeric_limits<weight_t>::max());

        // 3-5. compare

        auto h_sg_offsets =
          cugraph::test::to_host(*handle_, sg_graph_view.local_edge_partition_view().offsets());
        auto h_sg_indices =
          cugraph::test::to_host(*handle_, sg_graph_view.local_edge_partition_view().indices());
        auto h_sg_weights = cugraph::test::to_host(
          *handle_,
          raft::device_span<weight_t const>((*sg_edge_weight_view).value_firsts()[0],
                                            (*sg_edge_weight_view).edge_counts()[0]));

        auto h_mg_aggregate_distances = cugraph::test::to_host(*handle_, d_mg_aggregate_distances);
        auto h_mg_aggregate_predecessors =
          cugraph::test::to_host(*handle_, d_mg_aggregate_predecessors);

        auto h_sg_distances    = cugraph::test::to_host(*handle_, d_sg_distances);
        auto h_sg_predecessors = cugraph::test::to_host(*handle_, d_sg_predecessors);

        auto max_weight_element = std::max_element(h_sg_weights.begin(), h_sg_weights.end());
        auto epsilon            = *max_weight_element * weight_t{1e-6};
        auto nearly_equal       = [epsilon](auto lhs, auto rhs) {
          return std::fabs(lhs - rhs) < epsilon;
        };

        ASSERT_TRUE(std::equal(h_mg_aggregate_distances.begin(),
                               h_mg_aggregate_distances.end(),
                               h_sg_distances.begin(),
                               nearly_equal));

        for (size_t i = 0; i < h_mg_aggregate_predecessors.size(); ++i) {
          if (h_mg_aggregate_predecessors[i] == cugraph::invalid_vertex_id<vertex_t>::value) {
            ASSERT_TRUE(h_sg_predecessors[i] == h_mg_aggregate_predecessors[i])
              << "vertex reachability does not match with the SG result.";
          } else {
            auto pred_distance = h_sg_distances[h_mg_aggregate_predecessors[i]];
            bool found{false};
            for (auto j = h_sg_offsets[h_mg_aggregate_predecessors[i]];
                 j < h_sg_offsets[h_mg_aggregate_predecessors[i] + 1];
                 ++j) {
              if (h_sg_indices[j] == i) {
                if (nearly_equal(pred_distance + h_sg_weights[j], h_sg_distances[i])) {
                  found = true;
                  break;
                }
              }
            }
            ASSERT_TRUE(found)
              << "no edge from the predecessor vertex to this vertex with the matching weight.";
          }
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGSSSP<input_usecase_t>::handle_ = nullptr;

using Tests_MGSSSP_File = Tests_MGSSSP<cugraph::test::File_Usecase>;
using Tests_MGSSSP_Rmat = Tests_MGSSSP<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGSSSP_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGSSSP_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGSSSP_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGSSSP_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(SSSP_Usecase{0, false},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(SSSP_Usecase{0, true}, cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(SSSP_Usecase{0, false}, cugraph::test::File_Usecase("test/datasets/dblp.mtx")),
    std::make_tuple(SSSP_Usecase{0, true}, cugraph::test::File_Usecase("test/datasets/dblp.mtx")),
    std::make_tuple(SSSP_Usecase{1000, false},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(SSSP_Usecase{1000, true},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGSSSP_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(SSSP_Usecase{0, false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(SSSP_Usecase{0, true},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGSSSP_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(SSSP_Usecase{0, false, false},
                    cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(SSSP_Usecase{0, true, false},
                    cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
