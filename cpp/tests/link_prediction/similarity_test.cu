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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#include "link_prediction/similarity_compare.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <gtest/gtest.h>

struct Similarity_Usecase {
  bool use_weights{false};
  bool check_correctness{true};
  bool all_pairs{false};
  std::optional<size_t> max_seeds{std::nullopt};
  std::optional<size_t> max_vertex_pairs_to_check{std::nullopt};
  std::optional<size_t> topk{std::nullopt};
};

template <typename input_usecase_t>
class Tests_Similarity
  : public ::testing::TestWithParam<std::tuple<Similarity_Usecase, input_usecase_t>> {
 public:
  Tests_Similarity() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename test_functor_t>
  void run_current_test(std::tuple<Similarity_Usecase const&, input_usecase_t const&> const& param,
                        test_functor_t const& test_functor)
  {
    constexpr bool renumber                  = true;
    auto [similarity_usecase, input_usecase] = param;

    // 1. initialize handle

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // 2. create SG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, similarity_usecase.use_weights, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. run similarity

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Similarity test");
    }

    rmm::device_uvector<vertex_t> v1(0, handle.get_stream());
    rmm::device_uvector<vertex_t> v2(0, handle.get_stream());
    rmm::device_uvector<weight_t> result_score(0, handle.get_stream());

    raft::random::RngState rng_state{0};

    rmm::device_uvector<vertex_t> sources(0, handle.get_stream());
    std::optional<raft::device_span<vertex_t const>> sources_span{std::nullopt};

    if (similarity_usecase.max_seeds) {
      sources = cugraph::select_random_vertices(
        handle,
        graph_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        rng_state,
        std::min(*similarity_usecase.max_seeds,
                 static_cast<size_t>(graph_view.number_of_vertices())),
        false,
        false);
      sources_span = raft::device_span<vertex_t const>{sources.data(), sources.size()};
    }

    if (similarity_usecase.all_pairs) {
      std::tie(v1, v2, result_score) = test_functor.run(handle,
                                                        graph_view,
                                                        edge_weight_view,
                                                        sources_span,
                                                        similarity_usecase.use_weights,
                                                        similarity_usecase.topk);
    } else {
      if (!sources_span) {
        sources.resize(graph_view.number_of_vertices(), handle.get_stream());
        thrust::sequence(handle.get_thrust_policy(), sources.begin(), sources.end(), vertex_t{0});
        sources_span = raft::device_span<vertex_t const>{sources.data(), sources.size()};
      }

      rmm::device_uvector<size_t> offsets(0, handle.get_stream());

      std::tie(offsets, v2) = k_hop_nbrs(handle, graph_view, *sources_span, 2, true);

      v1 = cugraph::detail::expand_sparse_offsets(
        raft::device_span<size_t const>{offsets.data(), offsets.size()},
        vertex_t{0},
        handle.get_stream());

      cugraph::unrenumber_local_int_vertices(handle,
                                             v1.data(),
                                             v1.size(),
                                             sources.data(),
                                             vertex_t{0},
                                             static_cast<vertex_t>(sources.size()),
                                             true);

      auto new_size = thrust::distance(
        thrust::make_zip_iterator(v1.begin(), v2.begin()),
        thrust::remove_if(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(v1.begin(), v2.begin()),
          thrust::make_zip_iterator(v1.end(), v2.end()),
          [] __device__(auto tuple) { return thrust::get<0>(tuple) == thrust::get<1>(tuple); }));

      v1.resize(new_size, handle.get_stream());
      v2.resize(new_size, handle.get_stream());

      // FIXME:  Need to add some tests that specify actual vertex pairs
      std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs{
        {v1.data(), v1.size()}, {v2.data(), v2.size()}};

      result_score = test_functor.run(
        handle, graph_view, edge_weight_view, vertex_pairs, similarity_usecase.use_weights);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (similarity_usecase.check_correctness) {
      auto [src, dst, wgt] = cugraph::test::graph_to_host_coo(
        handle,
        graph_view,
        edge_weight_view,
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));

      size_t check_size = similarity_usecase.max_vertex_pairs_to_check
                            ? std::min(v1.size(), *similarity_usecase.max_vertex_pairs_to_check)
                            : v1.size();

      //
      // FIXME: Need to reorder here.  thrust::shuffle on the tuples (vertex_pairs_1,
      // vertex_pairs_2, result_score) would
      //        be sufficient.
      //
      std::vector<vertex_t> h_vertex_pair_1(check_size);
      std::vector<vertex_t> h_vertex_pair_2(check_size);
      std::vector<weight_t> h_result_score(check_size);

      raft::update_host(h_vertex_pair_1.data(), v1.data(), check_size, handle.get_stream());
      raft::update_host(h_vertex_pair_2.data(), v2.data(), check_size, handle.get_stream());
      raft::update_host(
        h_result_score.data(), result_score.data(), check_size, handle.get_stream());

      if (similarity_usecase.use_weights) {
        weighted_similarity_compare(graph_view.number_of_vertices(),
                                    std::tie(src, dst, wgt),
                                    std::tie(h_vertex_pair_1, h_vertex_pair_2),
                                    h_result_score,
                                    test_functor);
      } else {
        similarity_compare(graph_view.number_of_vertices(),
                           std::tie(src, dst, wgt),
                           std::tie(h_vertex_pair_1, h_vertex_pair_2),
                           h_result_score,
                           test_functor);
      }
    }
  }
};

using Tests_Similarity_File = Tests_Similarity<cugraph::test::File_Usecase>;
using Tests_Similarity_Rmat = Tests_Similarity<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Similarity_File, CheckInt32Int32FloatJaccard)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int32FloatJaccard)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatJaccard)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_Similarity_File, CheckInt32Int32FloatSorensen)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int32FloatSorensen)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatSorensen)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_Similarity_File, CheckInt32Int32FloatOverlap)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int32FloatOverlap)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatOverlap)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_Similarity_File, CheckInt32Int32FloatCosine)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_cosine_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt32Int32FloatCosine)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_cosine_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatCosine)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_cosine_t{});
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Similarity_File,
  ::testing::Combine(::testing::Values(Similarity_Usecase{false, true, false, 20, 100},
                                       Similarity_Usecase{false, true, false, 20, 100},
                                       Similarity_Usecase{false, true, false, 20, 100, 10},
                                       Similarity_Usecase{false, true, true, 20, 100},
                                       Similarity_Usecase{false, true, true, 20, 100},
                                       Similarity_Usecase{false, true, true, 20, 100, 10}),
#if 0
                      // FIXME: See Issue #4132... these tests don't work for multi-graph right now
                                       Similarity_Usecase{true, true, false, 20, 100},
                                       Similarity_Usecase{true, true, false, 20, 100},
                                       Similarity_Usecase{true, true, false, 20, 100, 10},
                                       Similarity_Usecase{true, true, true, 20, 100},
                                       Similarity_Usecase{true, true, true, 20, 100},
                                       Similarity_Usecase{true, true, true, 20, 100, 10}),
#endif
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                                       cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Similarity_Rmat,
  ::testing::Combine(
    ::testing::Values(Similarity_Usecase{false, true, false, 20, 100},
                      Similarity_Usecase{false, true, false, 20, 100},
                      Similarity_Usecase{false, true, false, 1000, 100, 10},
                      Similarity_Usecase{false, true, true, 20, 100},
                      Similarity_Usecase{false, true, true, 20, 100},
                      Similarity_Usecase{false, true, true, 10000, 10000, 10},
#if 0
                      // FIXME: See Issue #4132... these tests don't work for multi-graph right now
                      Similarity_Usecase{true, true, true, 20, 100},
                      Similarity_Usecase{true, true, true, 20, 100},
                      Similarity_Usecase{true, true, false, 20, 100, 10},
                      Similarity_Usecase{true, true, false, 20, 100},
                      Similarity_Usecase{true, true, false, 20, 100},
                      Similarity_Usecase{true, true, true, 20, 100, 10},
#endif
                      Similarity_Usecase{false, true, true, std::nullopt, std::nullopt, 100},
                      Similarity_Usecase{false, true, true, std::nullopt, std::nullopt, 10}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Similarity_File,
  ::testing::Combine(
    // disable correctness checks
    // Disable weighted computation testing in 22.10
    //::testing::Values(Similarity_Usecase{false, false}, Similarity_Usecase{true, false}),
    ::testing::Values(Similarity_Usecase{false, false, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Similarity_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    //::testing::Values(Similarity_Usecase{false, false}, Similarity_Usecase{true, false}),
    ::testing::Values(Similarity_Usecase{false, false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
