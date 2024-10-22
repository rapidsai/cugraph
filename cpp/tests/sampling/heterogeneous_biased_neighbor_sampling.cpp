/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "detail/nbr_sampling_validate.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/property_generator_utilities.hpp"

#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <gtest/gtest.h>

struct Heterogeneous_Biased_Neighbor_Sampling_Usecase {
  std::vector<int32_t> fanout{{-1}};
  int32_t batch_size{10};
  int32_t num_edge_types{1};
  bool flag_replacement{true};

  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Heterogeneous_Biased_Neighbor_Sampling
  : public ::testing::TestWithParam<
      std::tuple<Heterogeneous_Biased_Neighbor_Sampling_Usecase, input_usecase_t>> {
 public:
  Tests_Heterogeneous_Biased_Neighbor_Sampling() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(
    std::tuple<Heterogeneous_Biased_Neighbor_Sampling_Usecase const&, input_usecase_t const&> const& param)
  {
    using edge_type_t = int32_t;

    auto [heterogeneous_biased_neighbor_sampling_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<decltype(graph_view), bool>> edge_mask{std::nullopt};

    constexpr float select_probability{0.05};

    // FIXME:  Update the tests to initialize RngState and use it instead
    //         of seed...
    constexpr uint64_t seed{0};

    raft::random::RngState rng_state(seed);

    auto random_sources = cugraph::select_random_vertices(
      handle,
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

    auto batch_number = std::make_optional<rmm::device_uvector<int32_t>>(0, handle.get_stream());

    batch_number = cugraph::test::sequence(
      handle, random_sources.size(), heterogeneous_biased_neighbor_sampling_usecase.batch_size, int32_t{0});

    rmm::device_uvector<vertex_t> random_sources_copy(random_sources.size(), handle.get_stream());

    raft::copy(random_sources_copy.data(),
               random_sources.data(),
               random_sources.size(),
               handle.get_stream());

    std::optional<raft::device_span<int32_t const>>
      label_to_output_comm_rank_mapping{std::nullopt};
    
    // Generate the edge types

    std::optional<cugraph::edge_property_t<decltype(graph_view), edge_type_t>> edge_types{
      std::nullopt};

    if (heterogeneous_biased_neighbor_sampling_usecase.num_edge_types > 1) {
      edge_types = cugraph::test::generate<decltype(graph_view), edge_type_t>::edge_property(
          handle,
          graph_view,
          heterogeneous_biased_neighbor_sampling_usecase.num_edge_types);
    }

#ifdef NO_CUGRAPH_OPS
    EXPECT_THROW(
      cugraph::heterogeneous_biased_neighbor_sample(
        handle,
        rng_state,
        graph_view,
        edge_weight_view,
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        edge_types
            ? std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{(*edge_types)
                                                                                        .view()}
            : std::nullopt,
        *edge_weight_view,
        raft::device_span<vertex_t const>{random_sources_copy.data(), random_sources.size()},
        batch_number ? std::make_optional(raft::device_span<int32_t const>{batch_number->data(),
                                                                           batch_number->size()})
                     : std::nullopt,
        label_to_output_comm_rank_mapping,
        raft::host_span<int32_t const>(heterogeneous_biased_neighbor_sampling_usecase.fanout.data(),
                                       heterogeneous_biased_neighbor_sampling_usecase.fanout.size()),
        heterogeneous_biased_neighbor_sampling_usecase.num_edge_types,
        cugraph::sampling_flags_t{
            cugraph::prior_sources_behavior_t{0},
            true, // return_hops
            false, // dedupe_sources
            heterogeneous_biased_neighbor_sampling_usecase.flag_replacement
        }
        ),
      std::exception);
#else
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Biased neighbor sampling");
    }

    auto&& [src_out, dst_out, wgt_out, edge_id, edge_type, hop, offsets] =
      cugraph::heterogeneous_biased_neighbor_sample(
        handle,
        rng_state,
        graph_view,
        edge_weight_view,
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        edge_types
            ? std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{(*edge_types)
                                                                                        .view()}
            : std::nullopt,
        *edge_weight_view,
        raft::device_span<vertex_t const>{random_sources_copy.data(), random_sources.size()},
        batch_number ? std::make_optional(raft::device_span<int32_t const>{batch_number->data(),
                                                                           batch_number->size()})
                     : std::nullopt,
        label_to_output_comm_rank_mapping,
        raft::host_span<int32_t const>(heterogeneous_biased_neighbor_sampling_usecase.fanout.data(),
                                       heterogeneous_biased_neighbor_sampling_usecase.fanout.size()),
        heterogeneous_biased_neighbor_sampling_usecase.num_edge_types,
        cugraph::sampling_flags_t{
            cugraph::prior_sources_behavior_t{0},
            true, // return_hops
            false, // dedupe_sources
            heterogeneous_biased_neighbor_sampling_usecase.flag_replacement
        }
        );
    
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (heterogeneous_biased_neighbor_sampling_usecase.check_correctness) {
      //  First validate that the extracted edges are actually a subset of the
      //  edges in the input graph
      rmm::device_uvector<vertex_t> vertices(2 * src_out.size(), handle.get_stream());
      raft::copy(vertices.data(), src_out.data(), src_out.size(), handle.get_stream());
      raft::copy(
        vertices.data() + src_out.size(), dst_out.data(), dst_out.size(), handle.get_stream());
      vertices = cugraph::test::sort<vertex_t>(handle, std::move(vertices));
      vertices = cugraph::test::unique<vertex_t>(handle, std::move(vertices));

      rmm::device_uvector<size_t> d_subgraph_offsets(2, handle.get_stream());
      std::vector<size_t> h_subgraph_offsets({0, vertices.size()});

      raft::update_device(d_subgraph_offsets.data(),
                          h_subgraph_offsets.data(),
                          h_subgraph_offsets.size(),
                          handle.get_stream());

      rmm::device_uvector<vertex_t> src_compare(0, handle.get_stream());
      rmm::device_uvector<vertex_t> dst_compare(0, handle.get_stream());
      std::optional<rmm::device_uvector<weight_t>> wgt_compare{std::nullopt};

      std::tie(src_compare, dst_compare, wgt_compare, std::ignore) = extract_induced_subgraphs(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<size_t const>(d_subgraph_offsets.data(), 2),
        raft::device_span<vertex_t const>(vertices.data(), vertices.size()),
        true);
      
      
      

      ASSERT_TRUE(cugraph::test::validate_extracted_graph_is_subgraph(
        handle, src_compare, dst_compare, wgt_compare, src_out, dst_out, wgt_out));

      if (random_sources.size() < 100) {
        // This validation is too expensive for large number of vertices
        ASSERT_TRUE(
          cugraph::test::validate_sampling_depth(handle,
                                                 std::move(src_out),
                                                 std::move(dst_out),
                                                 std::move(wgt_out),
                                                 std::move(random_sources),
                                                 heterogeneous_biased_neighbor_sampling_usecase.fanout.size()));
      }
    }
#endif
  }
};

using Tests_Heterogeneous_Biased_Neighbor_Sampling_File =
  Tests_Heterogeneous_Biased_Neighbor_Sampling<cugraph::test::File_Usecase>;

using Tests_Heterogeneous_Biased_Neighbor_Sampling_Rmat =
  Tests_Heterogeneous_Biased_Neighbor_Sampling<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Heterogeneous_Biased_Neighbor_Sampling_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Heterogeneous_Biased_Neighbor_Sampling_File, CheckInt32Int64Float)
{
  run_current_test<int32_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Heterogeneous_Biased_Neighbor_Sampling_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Heterogeneous_Biased_Neighbor_Sampling_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Heterogeneous_Biased_Neighbor_Sampling_Rmat, CheckInt32Int64Float)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Heterogeneous_Biased_Neighbor_Sampling_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}


INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Heterogeneous_Biased_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, true},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_Heterogeneous_Biased_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, true},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Heterogeneous_Biased_Neighbor_Sampling_Rmat,
  ::testing::Combine(
    ::testing::Values(Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, true},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8}, 128, 2, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false, 0))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Heterogeneous_Biased_Neighbor_Sampling_Rmat,
  ::testing::Combine(
    ::testing::Values(Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8, 1, 9, 5, 12}, 1024, 4, false, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8, 1, 9, 5, 12}, 1024, 4, false, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8, 1, 9, 5, 12}, 1024, 4, true, false},
                      Heterogeneous_Biased_Neighbor_Sampling_Usecase{{4, 10, 7, 8, 1, 9, 5, 12}, 1024, 4, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0))));
//#endif

CUGRAPH_TEST_PROGRAM_MAIN()
