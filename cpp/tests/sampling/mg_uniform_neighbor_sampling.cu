/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "detail/nbr_sampling_utils.cuh"

#include <utilities/mg_utilities.hpp>

#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <gtest/gtest.h>

struct Uniform_Neighbor_Sampling_Usecase {
  std::vector<int32_t> fanout{{-1}};
  int32_t batch_size{10};
  bool check_correctness{true};
  bool flag_replacement{true};
};

template <typename input_usecase_t>
class Tests_MGUniform_Neighbor_Sampling
  : public ::testing::TestWithParam<
      std::tuple<Uniform_Neighbor_Sampling_Usecase, input_usecase_t>> {
 public:
  Tests_MGUniform_Neighbor_Sampling() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Uniform_Neighbor_Sampling_Usecase const& uniform_neighbor_sampling_usecase,
                        input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG construct graph");
    }

    constexpr bool sort_adjacency_list = true;

    auto [mg_graph, mg_edge_weights, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, true, true, false, sort_adjacency_list);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    //
    // Test is designed like GNN sampling.  We'll select 90% of vertices
    // to be included in sampling batches
    //
    constexpr float select_probability{0.9};

    // FIXME:  Update the tests to initialize RngState and use it instead
    //         of seed...
    uint64_t seed{static_cast<uint64_t>(handle_->get_comms().get_rank())};

    raft::random::RngState rng_state(seed);

    rmm::device_uvector<float> random_numbers(mg_graph_view.local_vertex_partition_range_size(),
                                              handle_->get_stream());
    rmm::device_uvector<vertex_t> random_sources(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle_->get_stream());

    cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                         random_numbers.data(),
                                         random_numbers.size(),
                                         float{0},
                                         float{1},
                                         seed);

    auto random_sources_end = thrust::copy_if(
      handle_->get_thrust_policy(),
      thrust::make_counting_iterator(vertex_t{0}),
      thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_size()),
      random_sources.begin(),
      [d_random_number = random_numbers.data(), select_probability] __device__(vertex_t offset) {
        return d_random_number[offset] < select_probability;
      });

    random_sources.resize(thrust::distance(random_sources.begin(), random_sources_end),
                          handle_->get_stream());
    random_sources.shrink_to_fit(handle_->get_stream());

    random_numbers.resize(random_sources.size(), handle_->get_stream());
    random_numbers.shrink_to_fit(handle_->get_stream());

    if (mg_graph_view.local_vertex_partition_range_first() > 0)
      thrust::transform(
        handle_->get_thrust_policy(),
        random_sources.begin(),
        random_sources.end(),
        random_sources.begin(),
        [base_offset = mg_graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
          return v + base_offset;
        });

    //
    //  Now we'll assign the vertices to batches
    //
    cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                         random_numbers.data(),
                                         random_numbers.size(),
                                         float{0},
                                         float{1},
                                         seed);

    thrust::sort_by_key(handle_->get_thrust_policy(),
                        random_numbers.begin(),
                        random_numbers.end(),
                        random_sources.begin());

    random_numbers.resize(0, handle_->get_stream());
    random_numbers.shrink_to_fit(handle_->get_stream());

    rmm::device_uvector<int32_t> batch_number(random_sources.size(), handle_->get_stream());

    thrust::tabulate(handle_->get_thrust_policy(),
                     batch_number.begin(),
                     batch_number.end(),
                     [batch_size = uniform_neighbor_sampling_usecase.batch_size] __device__(
                       int32_t index) { return index / batch_size; });

    rmm::device_uvector<vertex_t> random_sources_copy(random_sources.size(), handle_->get_stream());

    raft::copy(random_sources_copy.data(),
               random_sources.data(),
               random_sources.size(),
               handle_->get_stream());

#ifdef NO_CUGRAPH_OPS
    EXPECT_THROW(cugraph::uniform_neighbor_sample(
                   *handle_,
                   handle,
                   mg_graph_view,
                   mg_edge_weight_view,
                   std::nullopt,
                   std::move(random_sources_copy),
                   std::move(batch_number),
                   raft::host_span<int32_t const>(uniform_neighbor_sampling_usecase.fanout.data(),
                                                  uniform_neighbor_sampling_usecase.fanout.size()),
                   rng_state,
                   uniform_neighbor_sampling_usecase.flag_replacement),
                 std::exception);
#else
    auto&& [src_out, dst_out, wgt_out, edge_id, edge_type, hop, labels] =
      cugraph::uniform_neighbor_sample(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<cugraph::edge_property_view_t<
          edge_t,
          thrust::zip_iterator<thrust::tuple<edge_t const*, int32_t const*>>>>{std::nullopt},
        std::move(random_sources_copy),
        std::move(batch_number),
        raft::host_span<int32_t const>(uniform_neighbor_sampling_usecase.fanout.data(),
                                       uniform_neighbor_sampling_usecase.fanout.size()),
        rng_state,
        uniform_neighbor_sampling_usecase.flag_replacement);

    if (uniform_neighbor_sampling_usecase.check_correctness) {
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

      //  First validate that the extracted edges are actually a subset of the
      //  edges in the input graph
      rmm::device_uvector<vertex_t> vertices(2 * mg_aggregate_src.size(), handle_->get_stream());
      raft::copy(
        vertices.data(), mg_aggregate_src.data(), mg_aggregate_src.size(), handle_->get_stream());
      raft::copy(vertices.data() + mg_aggregate_src.size(),
                 mg_aggregate_dst.data(),
                 mg_aggregate_dst.size(),
                 handle_->get_stream());
      thrust::sort(handle_->get_thrust_policy(), vertices.begin(), vertices.end());
      auto vertices_end =
        thrust::unique(handle_->get_thrust_policy(), vertices.begin(), vertices.end());
      vertices.resize(thrust::distance(vertices.begin(), vertices_end), handle_->get_stream());

      vertices = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        *handle_, std::move(vertices), mg_graph_view.vertex_partition_range_lasts());

      thrust::sort(handle_->get_thrust_policy(), vertices.begin(), vertices.end());

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
        mg_graph_view,
        mg_edge_weight_view,
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
        cugraph::test::validate_extracted_graph_is_subgraph(*handle_,
                                                            mg_aggregate_src_compare,
                                                            mg_aggregate_dst_compare,
                                                            mg_aggregate_wgt_compare,
                                                            mg_aggregate_src,
                                                            mg_aggregate_dst,
                                                            mg_aggregate_wgt);

        if (random_sources.size() < 100) {
          // This validation is too expensive for large number of vertices
          if (mg_aggregate_src.size() > 0) {
            cugraph::test::validate_sampling_depth(*handle_,
                                                   std::move(mg_aggregate_src),
                                                   std::move(mg_aggregate_dst),
                                                   std::move(mg_aggregate_wgt),
                                                   std::move(mg_start_src),
                                                   uniform_neighbor_sampling_usecase.fanout.size());
          }
        }
      }
    }
#endif
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGUniform_Neighbor_Sampling<input_usecase_t>::handle_ =
  nullptr;

using Tests_MGUniform_Neighbor_Sampling_File =
  Tests_MGUniform_Neighbor_Sampling<cugraph::test::File_Usecase>;

using Tests_MGUniform_Neighbor_Sampling_Rmat =
  Tests_MGUniform_Neighbor_Sampling<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGUniform_Neighbor_Sampling_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_File, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_File, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGUniform_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Uniform_Neighbor_Sampling_Usecase{{2}, 100, true, true},
                      Uniform_Neighbor_Sampling_Usecase{{2}, 100, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGUniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Uniform_Neighbor_Sampling_Usecase{{2}, 10, false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGUniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Uniform_Neighbor_Sampling_Usecase{{2}, 500, false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
