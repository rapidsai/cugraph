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

#include <cugraph/graph_functions.hpp>

#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <gtest/gtest.h>

struct Uniform_Neighbor_Sampling_Usecase {
  std::vector<int32_t> fanout{{-1}};
  int32_t batch_size{10};
  bool with_replacement{true};
  bool check_correctness{true};
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
  void run_current_test(std::tuple<Uniform_Neighbor_Sampling_Usecase, input_usecase_t> const& param)
  {
    auto [uniform_neighbor_sampling_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_,
        input_usecase,
        true /* test_weighted */,
        true /* renumber */,
        false /* drop_self_loops */,
        false /* drop_multi_edges */);

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
    // Test is designed like GNN sampling.  We'll select 5% of vertices to be included in sampling
    // batches
    //

    constexpr float select_probability{0.05};

    raft::random::RngState rng_state(handle_->get_comms().get_rank());

    auto random_sources = cugraph::select_random_vertices(
      *handle_,
      mg_graph_view,
      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      rng_state,
      std::max(static_cast<size_t>(mg_graph_view.number_of_vertices() * select_probability),
               std::min(static_cast<size_t>(mg_graph_view.number_of_vertices()), size_t{1})),
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

    thrust::sort_by_key(handle_->get_thrust_policy(),
                        random_numbers.begin(),
                        random_numbers.end(),
                        random_sources.begin());

    random_numbers.resize(0, handle_->get_stream());
    random_numbers.shrink_to_fit(handle_->get_stream());

    rmm::device_uvector<int32_t> batch_number(random_sources.size(), handle_->get_stream());

    auto seed_sizes = cugraph::host_scalar_allgather(
      handle_->get_comms(), random_sources.size(), handle_->get_stream());
    size_t num_seeds   = std::reduce(seed_sizes.begin(), seed_sizes.end());
    size_t num_batches = (num_seeds + uniform_neighbor_sampling_usecase.batch_size - 1) /
                         uniform_neighbor_sampling_usecase.batch_size;

    std::vector<size_t> seed_offsets(seed_sizes.size());
    std::exclusive_scan(seed_sizes.begin(), seed_sizes.end(), seed_offsets.begin(), size_t{0});

    thrust::tabulate(
      handle_->get_thrust_policy(),
      batch_number.begin(),
      batch_number.end(),
      [seed_offset = seed_offsets[handle_->get_comms().get_rank()],
       num_batches] __device__(int32_t index) { return (seed_offset + index) % num_batches; });

    rmm::device_uvector<int32_t> unique_batches(num_batches, handle_->get_stream());
    rmm::device_uvector<int32_t> comm_ranks(num_batches, handle_->get_stream());

    cugraph::detail::sequence_fill(
      handle_->get_stream(), unique_batches.data(), unique_batches.size(), int32_t{0});
    thrust::tabulate(handle_->get_thrust_policy(),
                     comm_ranks.begin(),
                     comm_ranks.end(),
                     [num_gpus = handle_->get_comms().get_size()] __device__(auto index) {
                       return index % num_gpus;
                     });

    rmm::device_uvector<vertex_t> random_sources_copy(random_sources.size(), handle_->get_stream());

    raft::copy(random_sources_copy.data(),
               random_sources.data(),
               random_sources.size(),
               handle_->get_stream());

#ifdef NO_CUGRAPH_OPS
    EXPECT_THROW(
      cugraph::uniform_neighbor_sample(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        raft::device_span<vertex_t const>{random_sources_copy.data(), random_sources.size()},
        std::make_optional(
          raft::device_span<int32_t const>{batch_number.data(), batch_number.size()}),
        std::make_optional(std::make_tuple(
          raft::device_span<int32_t const>{unique_batches.data(), unique_batches.size()},
          raft::device_span<int32_t const>{comm_ranks.data(), comm_ranks.size()})),
        raft::host_span<int32_t const>(uniform_neighbor_sampling_usecase.fanout.data(),
                                       uniform_neighbor_sampling_usecase.fanout.size()),
        rng_state,
        true,
        uniform_neighbor_sampling_usecase.with_replacement),
      std::exception);
#else
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG uniform_neighbor_sample");
    }

    auto&& [src_out, dst_out, wgt_out, edge_id, edge_type, hop, labels, offsets] =
      cugraph::uniform_neighbor_sample(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        raft::device_span<vertex_t const>{random_sources_copy.data(), random_sources.size()},
        std::make_optional(
          raft::device_span<int32_t const>{batch_number.data(), batch_number.size()}),
        std::make_optional(std::make_tuple(
          raft::device_span<int32_t const>{unique_batches.data(), unique_batches.size()},
          raft::device_span<int32_t const>{comm_ranks.data(), comm_ranks.size()})),
        raft::host_span<int32_t const>(uniform_neighbor_sampling_usecase.fanout.data(),
                                       uniform_neighbor_sampling_usecase.fanout.size()),
        rng_state,
        true,
        uniform_neighbor_sampling_usecase.with_replacement);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

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
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_Rmat, CheckInt32Int64Float)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGUniform_Neighbor_Sampling_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGUniform_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Uniform_Neighbor_Sampling_Usecase{{10, 25}, 128, false, true},
                      Uniform_Neighbor_Sampling_Usecase{{10, 25}, 128, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGUniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(
    ::testing::Values(Uniform_Neighbor_Sampling_Usecase{{10, 25}, 128, false, true},
                      Uniform_Neighbor_Sampling_Usecase{{10, 25}, 128, true, true}),
    ::testing::Values(
      // cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));
      cugraph::test::Rmat_Usecase(5, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGUniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(
    ::testing::Values(Uniform_Neighbor_Sampling_Usecase{{10, 25}, 128, false, false},
                      Uniform_Neighbor_Sampling_Usecase{{10, 25}, 128, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
