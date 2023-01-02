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
#include <structure/detail/structure_utils.cuh>
#include <structure/induced_subgraph_validate.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/sort.h>

#include <gtest/gtest.h>

#include <random>

struct InducedSubgraph_Usecase {
  std::vector<size_t> subgraph_sizes{};
  bool test_weighted{false};
  bool check_correctness{false};
};

template <typename input_usecase_t>
class Tests_MGInducedSubgraph
  : public ::testing::TestWithParam<std::tuple<InducedSubgraph_Usecase, input_usecase_t>> {
 public:
  Tests_MGInducedSubgraph() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(
    std::tuple<InducedSubgraph_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [induced_subgraph_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, induced_subgraph_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    int my_rank = handle_->get_comms().get_rank();

    // Construct random subgraph vertex lists
    std::vector<size_t> h_subgraph_offsets(induced_subgraph_usecase.subgraph_sizes.size() + 1, 0);
    std::vector<size_t> h_sg_subgraph_offsets(induced_subgraph_usecase.subgraph_sizes.size() + 1,
                                              0);

    size_t max_subgraph_vertices_size =
      std::accumulate(induced_subgraph_usecase.subgraph_sizes.begin(),
                      induced_subgraph_usecase.subgraph_sizes.end(),
                      size_t{0});

    rmm::device_uvector<vertex_t> all_vertices(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> d_subgraph_vertices(max_subgraph_vertices_size,
                                                      handle_->get_stream());
    rmm::device_uvector<vertex_t> d_sg_subgraph_vertices(max_subgraph_vertices_size,
                                                         handle_->get_stream());

    if (my_rank == 0) {
      // NOTE: This limits the size graph we can run in a test.  All of the
      // vertices must fit on one GPU.  Of course, we've already limited this
      // by the validation step which is going to run a single CPU version for
      // comparison.
      //
      // This is also inefficient if the number of randomly selected vertices
      // is much lass than the number of vertices in the graph.  But since we're
      // testing with modest size graphs we're not going to worry about this.
      //
      // Better would be to construct a mechanism to select from an iterator
      // rather than having to realize the entire sequence in a container.
      all_vertices.resize(mg_graph_view.number_of_vertices(), handle_->get_stream());
      cugraph::detail::sequence_fill(
        handle_->get_stream(), all_vertices.data(), all_vertices.size(), vertex_t{0});
    }

    for (size_t i = 0; i < induced_subgraph_usecase.subgraph_sizes.size(); ++i) {
      auto subgraph_size = induced_subgraph_usecase.subgraph_sizes[i];
      auto start         = h_subgraph_offsets[i];
      auto sg_start      = h_sg_subgraph_offsets[i];

      ASSERT_TRUE(subgraph_size <= mg_graph_view.number_of_vertices()) << "Invalid subgraph size.";
      rmm::device_uvector<vertex_t> vertices(0, handle_->get_stream());

      if (my_rank == 0) {
        vertices = cugraph::test::randomly_select(*handle_, all_vertices, subgraph_size, true);
        raft::copy(d_sg_subgraph_vertices.data() + sg_start,
                   vertices.data(),
                   vertices.size(),
                   handle_->get_stream());
        h_sg_subgraph_offsets[i + 1] = sg_start + vertices.size();
      }

      vertices = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        *handle_, std::move(vertices), mg_graph_view.vertex_partition_range_lasts());

      raft::copy(d_subgraph_vertices.data() + start,
                 vertices.data(),
                 vertices.size(),
                 handle_->get_stream());
      h_subgraph_offsets[i + 1] = start + vertices.size();
    }

    d_subgraph_vertices.resize(h_subgraph_offsets.back(), handle_->get_stream());
    d_subgraph_vertices.shrink_to_fit(handle_->get_stream());

    if (my_rank == 0) {
      d_sg_subgraph_vertices.resize(h_sg_subgraph_offsets.back(), handle_->get_stream());
      d_sg_subgraph_vertices.shrink_to_fit(handle_->get_stream());
      all_vertices.resize(0, handle_->get_stream());
      all_vertices.shrink_to_fit(handle_->get_stream());
    }

    // 3. run MG InducedSubgraph
    auto d_subgraph_offsets = cugraph::test::to_device(*handle_, h_subgraph_offsets);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG induced-subgraph");
    }

    auto [d_subgraph_edgelist_majors,
          d_subgraph_edgelist_minors,
          d_subgraph_edgelist_weights,
          d_subgraph_edge_offsets] =
      cugraph::extract_induced_subgraphs(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        raft::device_span<size_t const>(d_subgraph_offsets.data(), d_subgraph_offsets.size()),
        raft::device_span<vertex_t const>(d_subgraph_vertices.data(), d_subgraph_vertices.size()),
        false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (induced_subgraph_usecase.check_correctness) {
      d_subgraph_edgelist_majors = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(d_subgraph_edgelist_majors.data(),
                                          d_subgraph_edgelist_majors.size()));
      d_subgraph_edgelist_minors = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(d_subgraph_edgelist_minors.data(),
                                          d_subgraph_edgelist_minors.size()));

      if (d_subgraph_edgelist_weights)
        *d_subgraph_edgelist_weights = cugraph::test::device_gatherv(
          *handle_,
          raft::device_span<weight_t const>(d_subgraph_edgelist_weights->data(),
                                            d_subgraph_edgelist_weights->size()));

      auto graph_ids_v = cugraph::detail::expand_sparse_offsets(
        raft::device_span<size_t const>(d_subgraph_edge_offsets.data(),
                                        d_subgraph_edge_offsets.size()),
        vertex_t{0},
        handle_->get_stream());

      graph_ids_v = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(graph_ids_v.data(), graph_ids_v.size()));

      if (d_subgraph_edgelist_weights) {
        thrust::sort_by_key(
          handle_->get_thrust_policy(),
          thrust::make_zip_iterator(graph_ids_v.begin(),
                                    d_subgraph_edgelist_majors.begin(),
                                    d_subgraph_edgelist_minors.begin()),
          thrust::make_zip_iterator(
            graph_ids_v.end(), d_subgraph_edgelist_majors.end(), d_subgraph_edgelist_minors.end()),
          d_subgraph_edgelist_weights->begin());
      } else {
        thrust::sort(
          handle_->get_thrust_policy(),
          thrust::make_zip_iterator(graph_ids_v.begin(),
                                    d_subgraph_edgelist_majors.begin(),
                                    d_subgraph_edgelist_minors.begin()),
          thrust::make_zip_iterator(
            graph_ids_v.end(), d_subgraph_edgelist_majors.end(), d_subgraph_edgelist_minors.end()));
      }

      auto d_subgraph_edgelist_offsets =
        cugraph::detail::compute_sparse_offsets<size_t>(graph_ids_v.begin(),
                                                        graph_ids_v.end(),
                                                        size_t{0},
                                                        size_t{d_subgraph_offsets.size() - 1},
                                                        handle_->get_stream());

      auto [sg_graph, sg_edge_weights, sg_number_map] = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<rmm::device_uvector<vertex_t>>{std::nullopt},
        false);

      if (my_rank == 0) {
        auto d_sg_subgraph_offsets = cugraph::test::to_device(*handle_, h_sg_subgraph_offsets);

        auto [d_reference_subgraph_edgelist_majors,
              d_reference_subgraph_edgelist_minors,
              d_reference_subgraph_edgelist_weights,
              d_reference_subgraph_edge_offsets] =
          cugraph::extract_induced_subgraphs(
            *handle_,
            sg_graph.view(),
            sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt,
            raft::device_span<size_t const>(d_sg_subgraph_offsets.data(),
                                            d_sg_subgraph_offsets.size()),
            raft::device_span<vertex_t const>(d_sg_subgraph_vertices.data(),
                                              d_sg_subgraph_vertices.size()),
            false);

        induced_subgraph_validate(*handle_,
                                  d_subgraph_edgelist_majors,
                                  d_subgraph_edgelist_minors,
                                  d_subgraph_edgelist_weights,
                                  d_subgraph_edgelist_offsets,
                                  d_reference_subgraph_edgelist_majors,
                                  d_reference_subgraph_edgelist_minors,
                                  d_reference_subgraph_edgelist_weights,
                                  d_reference_subgraph_edge_offsets);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGInducedSubgraph<input_usecase_t>::handle_ = nullptr;

using Tests_MGInducedSubgraph_File = Tests_MGInducedSubgraph<cugraph::test::File_Usecase>;
using Tests_MGInducedSubgraph_Rmat = Tests_MGInducedSubgraph<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGInducedSubgraph_File, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGInducedSubgraph_Rmat, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGInducedSubgraph_Rmat, CheckInt32Int64)
{
  run_current_test<int32_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGInducedSubgraph_Rmat, CheckInt64Int64)
{
  run_current_test<int64_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  karate_test,
  Tests_MGInducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{0}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{1}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{10}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{34}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{10, 0, 5}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{9, 3, 10}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{5, 12, 13}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  web_google_test,
  Tests_MGInducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{250, 130, 15}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{125, 300, 70}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  ljournal_2008_test,
  Tests_MGInducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{300, 20, 400}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{9130, 1200, 300}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  webbase_1M_test,
  Tests_MGInducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{700}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{500}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
