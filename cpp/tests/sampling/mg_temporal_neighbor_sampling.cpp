/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "cuda_runtime_api.h"
#include "detail/nbr_sampling_validate.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <gtest/gtest.h>

struct Temporal_Neighbor_Sampling_Usecase {
  std::vector<int32_t> fanout{{-1}};
  int32_t batch_size{10};
  bool flag_replacement{true};
  bool biased{false};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTemporal_Neighbor_Sampling
  : public ::testing::TestWithParam<
      std::tuple<Temporal_Neighbor_Sampling_Usecase, input_usecase_t>> {
 public:
  Tests_MGTemporal_Neighbor_Sampling() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(
    std::tuple<Temporal_Neighbor_Sampling_Usecase const&, input_usecase_t const&> const& param)
  {
    using edge_time_t               = int32_t;
    using edge_type_t               = int32_t;
    constexpr bool store_transposed = false;
    constexpr bool renumber         = true;

    auto [temporal_neighbor_sampling_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MG construct graph");
    }

    std::optional<
      std::function<rmm::device_uvector<edge_time_t>(raft::handle_t const&, size_t, size_t)>>
      edge_start_times_functor{std::nullopt};
    std::optional<
      std::function<rmm::device_uvector<edge_time_t>(raft::handle_t const&, size_t, size_t)>>
      edge_end_times_functor{std::nullopt};

    // FIXME: Seed should be configurable in the test
    constexpr uint64_t seed{0};
    raft::random::RngState rng_state(seed + handle_->get_comms().get_rank());

    edge_start_times_functor = std::make_optional(
      [&rng_state](raft::handle_t const& handle, size_t size, size_t base_offset) {
        rmm::device_uvector<edge_time_t> result(size, handle.get_stream());

        cugraph::detail::uniform_random_fill(handle.get_stream(),
                                             result.data(),
                                             result.size(),
                                             edge_time_t{0},
                                             edge_time_t{20000},
                                             rng_state);

        return std::move(result);
      });

    auto [graph,
          edge_weights,
          edge_ids,
          edge_types,
          edge_start_times,
          edge_end_times,
          renumber_map] = cugraph::test::
      construct_graph<vertex_t, edge_t, weight_t, int32_t, int32_t, store_transposed, true>(
        *handle_,
        input_usecase,
        true,
        std::nullopt,
        std::nullopt,
        edge_start_times_functor,
        edge_end_times_functor,
        renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weights_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;
    auto edge_ids_view   = edge_ids ? std::make_optional((*edge_ids).view()) : std::nullopt;
    auto edge_types_view = edge_types ? std::make_optional((*edge_types).view()) : std::nullopt;
    auto edge_start_times_view =
      edge_start_times ? std::make_optional((*edge_start_times).view()) : std::nullopt;
    auto edge_end_times_view =
      edge_end_times ? std::make_optional((*edge_end_times).view()) : std::nullopt;

#if 0
    std::optional<cugraph::edge_property_t<decltype(graph_view), bool>> edge_mask{std::nullopt};
    if (temporal_neighbor_sampling_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(*handle_, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }
#endif

    constexpr float select_probability{0.05};

    auto random_sources = cugraph::select_random_vertices(
      *handle_,
      graph_view,
      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      rng_state,
      std::max(static_cast<size_t>(graph_view.number_of_vertices() * select_probability),
               std::min(static_cast<size_t>(graph_view.number_of_vertices()), size_t{1})),
      false,
      false);

    //
    //  Now we'll assign the vertices to batches
    //
    rmm::device_uvector<float> random_numbers(random_sources.size(), handle_->get_stream());

    cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                         random_numbers.data(),
                                         random_numbers.size(),
                                         float{0},
                                         float{1},
                                         rng_state);

    std::tie(random_numbers, random_sources) = cugraph::test::sort_by_key<float, vertex_t>(
      *handle_, std::move(random_numbers), std::move(random_sources));

    random_numbers.resize(0, handle_->get_stream());
    random_numbers.shrink_to_fit(handle_->get_stream());

    auto batch_number = std::make_optional<rmm::device_uvector<int32_t>>(0, handle_->get_stream());

    batch_number = cugraph::test::sequence(
      *handle_, random_sources.size(), temporal_neighbor_sampling_usecase.batch_size, int32_t{0});

    rmm::device_uvector<vertex_t> random_sources_copy(random_sources.size(), handle_->get_stream());

    raft::copy(random_sources_copy.data(),
               random_sources.data(),
               random_sources.size(),
               handle_->get_stream());

    std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank_mapping{std::nullopt};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MG Uniform temporal sampling");
    }

    cugraph::sampling_flags_t sampling_flags{};

    auto&& [src_out,
            dst_out,
            wgt_out,
            edge_id,
            edge_type,
            edge_start_time,
            edge_end_time,
            hop,
            offsets] =
      homogeneous_uniform_temporal_neighbor_sample(
        *handle_,
        rng_state,
        graph_view,
        edge_weights_view,
        edge_ids_view,
        edge_types_view,
        *edge_start_times_view,
        edge_end_times_view,
        raft::device_span<vertex_t const>{random_sources_copy.data(), random_sources.size()},
        batch_number ? std::make_optional(raft::device_span<int32_t const>{batch_number->data(),
                                                                           batch_number->size()})
                     : std::nullopt,
        label_to_output_comm_rank_mapping,
        raft::host_span<int32_t const>(temporal_neighbor_sampling_usecase.fanout.data(),
                                       temporal_neighbor_sampling_usecase.fanout.size()),
        sampling_flags);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (temporal_neighbor_sampling_usecase.check_correctness) {
      // Consolidate results on GPU 0
      auto mg_start_src = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{random_sources.data(), random_sources.size()});
      auto mg_aggregate_src = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{src_out.data(), src_out.size()});
      auto mg_aggregate_dst = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{dst_out.data(), dst_out.size()});
      auto mg_aggregate_wgt =
        wgt_out ? std::make_optional(cugraph::test::device_gatherv(
                    *handle_, raft::device_span<weight_t const>{wgt_out->data(), wgt_out->size()}))
                : std::nullopt;
      auto mg_aggregate_start_times = edge_start_time
                                        ? std::make_optional(cugraph::test::device_gatherv(
                                            *handle_,
                                            raft::device_span<edge_time_t const>{
                                              edge_start_time->data(), edge_start_time->size()}))
                                        : std::nullopt;
      auto mg_aggregate_end_times =
        edge_end_time
          ? std::make_optional(cugraph::test::device_gatherv(
              *handle_,
              raft::device_span<edge_time_t const>{edge_end_time->data(), edge_end_time->size()}))
          : std::nullopt;

      //  First validate that the extracted edges are actually a subset of the
      //  edges in the input graph
      rmm::device_uvector<vertex_t> vertices(2 * mg_aggregate_src.size(), handle_->get_stream());
      raft::copy(
        vertices.data(), mg_aggregate_src.data(), mg_aggregate_src.size(), handle_->get_stream());
      raft::copy(vertices.data() + mg_aggregate_src.size(),
                 mg_aggregate_dst.data(),
                 mg_aggregate_dst.size(),
                 handle_->get_stream());
      vertices = cugraph::test::sort<vertex_t>(*handle_, std::move(vertices));
      vertices = cugraph::test::unique<vertex_t>(*handle_, std::move(vertices));

      std::tie(vertices, std::ignore) =
        cugraph::shuffle_int_vertices(*handle_,
                                      std::move(vertices),
                                      std::vector<cugraph::arithmetic_device_uvector_t>{},
                                      graph_view.vertex_partition_range_lasts());

      rmm::device_uvector<size_t> d_subgraph_offsets(2, handle_->get_stream());
      std::vector<size_t> h_subgraph_offsets({0, vertices.size()});

      raft::update_device(d_subgraph_offsets.data(),
                          h_subgraph_offsets.data(),
                          h_subgraph_offsets.size(),
                          handle_->get_stream());

      rmm::device_uvector<vertex_t> src_compare(0, handle_->get_stream());
      rmm::device_uvector<vertex_t> dst_compare(0, handle_->get_stream());
      std::optional<rmm::device_uvector<weight_t>> wgt_compare{std::nullopt};

      std::tie(src_compare, dst_compare, wgt_compare, std::ignore) = extract_induced_subgraphs(
        *handle_,
        graph_view,
        edge_weights_view,
        raft::device_span<size_t const>(d_subgraph_offsets.data(), 2),
        raft::device_span<vertex_t const>(vertices.data(), vertices.size()),
        true);

      auto mg_aggregate_src_compare = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{src_compare.data(), src_compare.size()});
      auto mg_aggregate_dst_compare = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{dst_compare.data(), dst_compare.size()});
      auto mg_aggregate_wgt_compare =
        wgt_compare
          ? std::make_optional(cugraph::test::device_gatherv(
              *handle_,
              raft::device_span<weight_t const>{wgt_compare->data(), wgt_compare->size()}))
          : std::nullopt;

      if (handle_->get_comms().get_rank() == 0) {
        cugraph::test::validate_extracted_graph_is_subgraph(
          *handle_,
          raft::device_span<vertex_t const>{mg_aggregate_src_compare.data(),
                                            mg_aggregate_src_compare.size()},
          raft::device_span<vertex_t const>{mg_aggregate_dst_compare.data(),
                                            mg_aggregate_dst_compare.size()},
          mg_aggregate_wgt_compare
            ? std::make_optional(raft::device_span<weight_t const>{
                mg_aggregate_wgt_compare->data(), mg_aggregate_wgt_compare->size()})
            : std::nullopt,
          raft::device_span<vertex_t const>{mg_aggregate_src.data(), mg_aggregate_src.size()},
          raft::device_span<vertex_t const>{mg_aggregate_dst.data(), mg_aggregate_dst.size()},
          mg_aggregate_wgt ? std::make_optional(raft::device_span<weight_t const>{
                               mg_aggregate_wgt->data(), mg_aggregate_wgt->size()})
                           : std::nullopt);

        ASSERT_TRUE(cugraph::test::validate_temporal_integrity(
          *handle_,
          raft::device_span<vertex_t const>{mg_aggregate_src.data(), mg_aggregate_src.size()},
          raft::device_span<const vertex_t>{mg_aggregate_dst.data(), mg_aggregate_dst.size()},
          raft::device_span<const edge_time_t>{mg_aggregate_start_times->data(),
                                               mg_aggregate_start_times->size()},
          raft::device_span<const vertex_t>{mg_start_src.data(), mg_start_src.size()}));

        if (random_sources.size() < 100) {
          // This validation is too expensive for large number of vertices
          ASSERT_TRUE(cugraph::test::validate_sampling_depth(
            *handle_,
            std::move(mg_aggregate_src),
            std::move(mg_aggregate_dst),
            std::move(mg_start_src),
            temporal_neighbor_sampling_usecase.fanout.size()));
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTemporal_Neighbor_Sampling<input_usecase_t>::handle_ =
  nullptr;

using Tests_MGTemporal_Neighbor_Sampling_File =
  Tests_MGTemporal_Neighbor_Sampling<cugraph::test::File_Usecase>;

using Tests_MGTemporal_Neighbor_Sampling_Rmat =
  Tests_MGTemporal_Neighbor_Sampling<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTemporal_Neighbor_Sampling_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGTemporal_Neighbor_Sampling_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGTemporal_Neighbor_Sampling_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGTemporal_Neighbor_Sampling_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTemporal_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, false, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, false, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, false, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, false, true, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGTemporal_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, false, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, false, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, false, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, false, true, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGTemporal_Neighbor_Sampling_Rmat,
  ::testing::Combine(
    ::testing::Values(Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, false, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, false, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, false, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, false, true, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, -1, 10}, 128, true, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false, 0))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTemporal_Neighbor_Sampling_Rmat,
  ::testing::Combine(
    ::testing::Values(Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, false, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, false, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, false, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, false, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, false, true, true},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, true, false},
                      Temporal_Neighbor_Sampling_Usecase{{4, 10}, 128, true, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
