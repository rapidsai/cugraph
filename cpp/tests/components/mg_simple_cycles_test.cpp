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
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <vector>

// Canonicalize a cycle by rotating it to the vertex with the smallest index.
template <typename vertex_t>
std::vector<vertex_t> canonicalize_cycle(std::vector<vertex_t>&& cycle_vertices)
{
  if (cycle_vertices.size() <= 1) { return cycle_vertices; }
  auto min_it    = std::min_element(cycle_vertices.begin(), cycle_vertices.end());
  auto min_index = static_cast<size_t>(std::distance(cycle_vertices.begin(), min_it));
  std::rotate(cycle_vertices.begin(), cycle_vertices.begin() + min_index, cycle_vertices.end());
  return cycle_vertices;
}

template <typename vertex_t>
std::vector<std::vector<vertex_t>> canonicalize_cycles(std::vector<std::vector<vertex_t>>&& cycles)
{
  for (auto& cycle : cycles) {
    cycle = canonicalize_cycle(std::move(cycle));
  }
  // lexicographic sort of the rotated vertex sequences (e.g. [0, 2] < [0, 2, 1] < [1, 2])
  std::sort(cycles.begin(), cycles.end());
  return cycles;
}

template <typename vertex_t>
std::vector<std::vector<vertex_t>> cycles_from_vertices_and_offsets(
  std::vector<vertex_t> const& cycle_vertices, std::vector<size_t> const& cycle_offsets)
{
  std::vector<std::vector<vertex_t>> cycles{};
  if (cycle_offsets.size() == 0) { return cycles; }
  cycles.reserve(cycle_offsets.size() - 1);
  for (size_t i = 0; i + 1 < cycle_offsets.size(); ++i) {
    cycles.emplace_back(cycle_vertices.begin() + cycle_offsets[i],
                        cycle_vertices.begin() + cycle_offsets[i + 1]);
  }
  return cycles;
}

template <typename vertex_t>
std::vector<std::vector<vertex_t>> cycles_from_vertices_and_lengths(
  std::vector<vertex_t> const& cycle_vertices, std::vector<size_t> const& cycle_lengths)
{
  std::vector<std::vector<vertex_t>> cycles{};
  cycles.reserve(cycle_lengths.size());
  size_t offset{0};
  for (auto length : cycle_lengths) {
    cycles.emplace_back(cycle_vertices.begin() + offset, cycle_vertices.begin() + offset + length);
    offset += length;
  }
  return cycles;
}

struct SimpleCycles_Usecase {
  size_t k{10};
  double seed_ratio{1.0};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGSimpleCycles
  : public ::testing::TestWithParam<std::tuple<SimpleCycles_Usecase, input_usecase_t>> {
 public:
  Tests_MGSimpleCycles() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running simple_cycles on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(SimpleCycles_Usecase const& simple_cycles_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t    = float;
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    ASSERT_FALSE(mg_graph_view.is_symmetric())
      << "simple_cycles currently supports directed (asymmetric) graphs only.";

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (simple_cycles_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask(edge_mask->view());
    }

    ASSERT_TRUE(simple_cycles_usecase.seed_ratio > 0.0 && simple_cycles_usecase.seed_ratio <= 1.0)
      << "seed_ratio must be greater than 0.0 and less than or equal to 1.0.";
    std::optional<rmm::device_uvector<vertex_t>> d_mg_seed_vertices{std::nullopt};
    if (simple_cycles_usecase.seed_ratio < 1.0) {
      auto num_seeds = static_cast<size_t>(static_cast<double>(mg_graph_view.number_of_vertices()) *
                                           simple_cycles_usecase.seed_ratio);
      num_seeds      = std::max(num_seeds, size_t{1});
      num_seeds      = std::min(num_seeds, static_cast<size_t>(mg_graph_view.number_of_vertices()));
      raft::random::RngState rng_state(handle_->get_comms().get_rank());
      d_mg_seed_vertices = cugraph::select_random_vertices(
        *handle_,
        mg_graph_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        rng_state,
        num_seeds,
        false /* with_replacement */,
        true /* sort_vertices */);
    }

    // 2. run MG simple_cycles

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG simple_cycles");
    }

    auto [d_mg_cycle_vertices, d_mg_cycle_offsets] = cugraph::simple_cycles(
      *handle_,
      mg_graph_view,
      d_mg_seed_vertices ? std::make_optional(raft::device_span<vertex_t const>{
                             d_mg_seed_vertices->data(), d_mg_seed_vertices->size()})
                         : std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      static_cast<vertex_t>(simple_cycles_usecase.k));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (simple_cycles_usecase.check_correctness) {
      // 3-1. aggregate MG results (cycle offsets are local; gather cycle lengths instead)

      auto h_mg_cycle_offsets = cugraph::test::to_host(*handle_, d_mg_cycle_offsets);
      std::vector<size_t> h_mg_cycle_lengths(h_mg_cycle_offsets.size() - 1);
      std::adjacent_difference(
        h_mg_cycle_offsets.begin() + 1, h_mg_cycle_offsets.end(), h_mg_cycle_lengths.begin());
      auto d_mg_cycle_lengths = cugraph::test::to_device(*handle_, h_mg_cycle_lengths);

      auto d_mg_aggregate_cycle_vertices = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(d_mg_cycle_vertices.data(), d_mg_cycle_vertices.size()));
      auto d_mg_aggregate_cycle_lengths = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<size_t const>(d_mg_cycle_lengths.data(), d_mg_cycle_lengths.size()));

      std::optional<rmm::device_uvector<vertex_t>> d_mg_aggregate_seed_vertices{std::nullopt};
      if (d_mg_seed_vertices) {
        d_mg_aggregate_seed_vertices =
          cugraph::test::device_gatherv(*handle_,
                                        raft::device_span<vertex_t const>(
                                          d_mg_seed_vertices->data(), d_mg_seed_vertices->size()));
      }

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. run SG simple_cycles

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        if (d_mg_aggregate_seed_vertices) {
          *d_mg_aggregate_seed_vertices =
            cugraph::test::sort<vertex_t>(*handle_, std::move(*d_mg_aggregate_seed_vertices));
        }

        auto [d_sg_cycle_vertices, d_sg_cycle_offsets] = cugraph::simple_cycles(
          *handle_,
          sg_graph_view,
          d_mg_aggregate_seed_vertices
            ? std::make_optional(raft::device_span<vertex_t const>{
                d_mg_aggregate_seed_vertices->data(), d_mg_aggregate_seed_vertices->size()})
            : std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          static_cast<vertex_t>(simple_cycles_usecase.k));

        // 3-3. compare

        auto h_mg_cycles = canonicalize_cycles(cycles_from_vertices_and_lengths(
          cugraph::test::to_host(*handle_, d_mg_aggregate_cycle_vertices),
          cugraph::test::to_host(*handle_, d_mg_aggregate_cycle_lengths)));
        auto h_sg_cycles = canonicalize_cycles(
          cycles_from_vertices_and_offsets(cugraph::test::to_host(*handle_, d_sg_cycle_vertices),
                                           cugraph::test::to_host(*handle_, d_sg_cycle_offsets)));

        ASSERT_EQ(h_sg_cycles.size(), h_mg_cycles.size())
          << "number of simple cycles does not match the SG values.";
        ASSERT_TRUE(std::equal(h_sg_cycles.begin(), h_sg_cycles.end(), h_mg_cycles.begin()))
          << "simple cycles do not match the SG values.";
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGSimpleCycles<input_usecase_t>::handle_ = nullptr;

using Tests_MGSimpleCycles_File = Tests_MGSimpleCycles<cugraph::test::File_Usecase>;
using Tests_MGSimpleCycles_Rmat = Tests_MGSimpleCycles<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGSimpleCycles_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGSimpleCycles_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGSimpleCycles_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGSimpleCycles_File,
  ::testing::Combine(
    ::testing::Values(SimpleCycles_Usecase{size_t{3}, 0.5, false},
                      SimpleCycles_Usecase{size_t{3}, 0.5, true},
                      SimpleCycles_Usecase{size_t{3}, 1.0, false},
                      SimpleCycles_Usecase{size_t{3}, 1.0, true},
                      SimpleCycles_Usecase{size_t{6}, 0.5, false},
                      SimpleCycles_Usecase{size_t{6}, 0.5, true},
                      SimpleCycles_Usecase{size_t{6}, 1.0, false},
                      SimpleCycles_Usecase{size_t{6}, 1.0, true},
                      SimpleCycles_Usecase{size_t{10}, 0.5, false},
                      SimpleCycles_Usecase{size_t{10}, 0.5, true},
                      SimpleCycles_Usecase{size_t{10}, 1.0, false},
                      SimpleCycles_Usecase{size_t{10}, 1.0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate-asymmetric.csv"),
                      cugraph::test::File_Usecase("test/datasets/cage6.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGSimpleCycles_Rmat,
  ::testing::Combine(
    ::testing::Values(SimpleCycles_Usecase{size_t{3}, 0.5, false},
                      SimpleCycles_Usecase{size_t{3}, 0.5, true},
                      SimpleCycles_Usecase{size_t{3}, 1.0, false},
                      SimpleCycles_Usecase{size_t{3}, 1.0, true},
                      SimpleCycles_Usecase{size_t{6}, 0.5, false},
                      SimpleCycles_Usecase{size_t{6}, 0.5, true},
                      SimpleCycles_Usecase{size_t{6}, 1.0, false},
                      SimpleCycles_Usecase{size_t{6}, 1.0, true},
                      SimpleCycles_Usecase{size_t{10}, 0.5, false},
                      SimpleCycles_Usecase{size_t{10}, 0.5, true},
                      SimpleCycles_Usecase{size_t{10}, 1.0, false},
                      SimpleCycles_Usecase{size_t{10}, 1.0, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGSimpleCycles_Rmat,
  ::testing::Values(
    std::make_tuple(SimpleCycles_Usecase{size_t{10}, 1.0, false, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(SimpleCycles_Usecase{size_t{10}, 1.0, true, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
