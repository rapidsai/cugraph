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
#include <link_prediction/similarity_compare.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <gtest/gtest.h>

struct Similarity_Usecase {
  bool use_weights{false};
  bool check_correctness{true};
  size_t max_seeds{std::numeric_limits<size_t>::max()};
  size_t max_vertex_pairs_to_check{std::numeric_limits<size_t>::max()};
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

    //
    // FIXME:  Don't currently have an MG implementation of 2-hop neighbors.
    //         For now we'll do that on the CPU (really slowly, so keep max_seed
    //         small)
    //
    rmm::device_uvector<vertex_t> d_v1(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_v2(0, handle.get_stream());

    {
      auto [src, dst, wgt] = cugraph::test::graph_to_host_coo(handle, graph_view, edge_weight_view);

      size_t max_vertices = std::min(static_cast<size_t>(graph_view.number_of_vertices()),
                                     similarity_usecase.max_seeds);
      std::vector<vertex_t> h_v1;
      std::vector<vertex_t> h_v2;
      std::vector<vertex_t> one_hop_v1;
      std::vector<vertex_t> one_hop_v2;

      for (size_t seed = 0; seed < max_vertices; ++seed) {
        std::for_each(thrust::make_zip_iterator(src.begin(), dst.begin()),
                      thrust::make_zip_iterator(src.end(), dst.end()),
                      [&one_hop_v1, &one_hop_v2, seed](auto t) {
                        auto u = thrust::get<0>(t);
                        auto v = thrust::get<1>(t);
                        if (u == seed) {
                          one_hop_v1.push_back(u);
                          one_hop_v2.push_back(v);
                        }
                      });
      }

      std::for_each(thrust::make_zip_iterator(one_hop_v1.begin(), one_hop_v2.begin()),
                    thrust::make_zip_iterator(one_hop_v1.end(), one_hop_v2.end()),
                    [&](auto t1) {
                      auto seed     = thrust::get<0>(t1);
                      auto neighbor = thrust::get<1>(t1);
                      std::for_each(thrust::make_zip_iterator(src.begin(), dst.begin()),
                                    thrust::make_zip_iterator(src.end(), dst.end()),
                                    [&](auto t2) {
                                      auto u = thrust::get<0>(t2);
                                      auto v = thrust::get<1>(t2);
                                      if (u == neighbor) {
                                        h_v1.push_back(seed);
                                        h_v2.push_back(v);
                                      }
                                    });
                    });

      std::sort(thrust::make_zip_iterator(h_v1.begin(), h_v2.begin()),
                thrust::make_zip_iterator(h_v1.end(), h_v2.end()));

      auto end_iter = std::unique(thrust::make_zip_iterator(h_v1.begin(), h_v2.begin()),
                                  thrust::make_zip_iterator(h_v1.end(), h_v2.end()),
                                  [](auto t1, auto t2) {
                                    return (thrust::get<0>(t1) == thrust::get<0>(t2)) &&
                                           (thrust::get<1>(t1) == thrust::get<1>(t2));
                                  });

      h_v1.resize(
        thrust::distance(thrust::make_zip_iterator(h_v1.begin(), h_v2.begin()), end_iter));
      h_v2.resize(h_v1.size());

      d_v1.resize(h_v1.size(), handle.get_stream());
      d_v2.resize(h_v2.size(), handle.get_stream());

      raft::update_device(d_v1.data(), h_v1.data(), h_v1.size(), handle.get_stream());
      raft::update_device(d_v2.data(), h_v2.data(), h_v2.size(), handle.get_stream());
    }

    // FIXME:  Need to add some tests that specify actual vertex pairs
    // FIXME:  Need to a variation that calls call the two hop neighbors function
    // FIXME:  Debugging state as of EOD 9/28:
    //           1) Tested case of no vertex pairs... works great :-)
    //           2) Don't have a 2-hop on GPU yet.  Perhaps write a 2-hop on CPU
    //              for now?  We could then use that for testing the 2-hop function
    //              later.
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs{
      {d_v1.data(), d_v1.size()}, {d_v2.data(), d_v2.size()}};

    auto result_score = test_functor.run(
      handle, graph_view, edge_weight_view, vertex_pairs, similarity_usecase.use_weights);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (similarity_usecase.check_correctness) {
      auto [src, dst, wgt] = cugraph::test::graph_to_host_coo(handle, graph_view, edge_weight_view);

      size_t check_size = std::min(d_v1.size(), similarity_usecase.max_vertex_pairs_to_check);

      //
      // FIXME: Need to reorder here.  thrust::shuffle on the tuples (vertex_pairs_1,
      // vertex_pairs_2, result_score) would
      //        be sufficient.
      //
      std::vector<vertex_t> h_vertex_pair_1(check_size);
      std::vector<vertex_t> h_vertex_pair_2(check_size);
      std::vector<weight_t> h_result_score(check_size);

      raft::update_host(
        h_vertex_pair_1.data(), std::get<0>(vertex_pairs).data(), check_size, handle.get_stream());
      raft::update_host(
        h_vertex_pair_2.data(), std::get<1>(vertex_pairs).data(), check_size, handle.get_stream());
      raft::update_host(
        h_result_score.data(), result_score.data(), check_size, handle.get_stream());

      similarity_compare(graph_view.number_of_vertices(),
                         std::tie(src, dst, wgt),
                         std::tie(h_vertex_pair_1, h_vertex_pair_2),
                         h_result_score,
                         test_functor);
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

TEST_P(Tests_Similarity_Rmat, CheckInt32Int64FloatJaccard)
{
  run_current_test<int32_t, int64_t, float>(
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

TEST_P(Tests_Similarity_Rmat, CheckInt32Int64FloatSorensen)
{
  run_current_test<int32_t, int64_t, float>(
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

TEST_P(Tests_Similarity_Rmat, CheckInt32Int64FloatOverlap)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_Similarity_Rmat, CheckInt64Int64FloatOverlap)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Similarity_File,
  ::testing::Combine(
    // enable correctness checks
    // Disable weighted computation testing in 22.10
    //::testing::Values(Similarity_Usecase{true, true, 20, 100}, Similarity_Usecase{false, true, 20,
    // 100}),
    ::testing::Values(Similarity_Usecase{false, true, 20, 100}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Similarity_Rmat,
  ::testing::Combine(
    // enable correctness checks
    // Disable weighted computation testing in 22.10
    //::testing::Values(Similarity_Usecase{true, true, 20, 100}, Similarity_Usecase{false, true, 20,
    // 100}),
    ::testing::Values(Similarity_Usecase{false, true, 20, 100}),
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
    ::testing::Values(Similarity_Usecase{false, false}),
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
    ::testing::Values(Similarity_Usecase{false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
