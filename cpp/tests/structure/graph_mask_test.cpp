/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_mask.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_reference(vertex_t const* p_src_vertices,
                vertex_t const* p_dst_vertices,
                std::optional<weight_t const*> p_edge_weights,
                vertex_t number_of_vertices,
                edge_t number_of_edges)
{
  std::vector<edge_t> offsets(number_of_vertices + 1, edge_t{0});
  std::vector<vertex_t> indices(number_of_edges, vertex_t{0});
  auto weights = p_edge_weights
                   ? std::make_optional<std::vector<weight_t>>(number_of_edges, weight_t{0.0})
                   : std::nullopt;

  for (edge_t i = 0; i < number_of_edges; ++i) {
    auto major = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    offsets[1 + major]++;
  }
  std::partial_sum(offsets.begin() + 1, offsets.end(), offsets.begin() + 1);

  for (edge_t i = 0; i < number_of_edges; ++i) {
    auto major           = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    auto minor           = store_transposed ? p_src_vertices[i] : p_dst_vertices[i];
    auto start           = offsets[major];
    auto degree          = offsets[major + 1] - start;
    auto idx             = indices[start + degree - 1]++;
    indices[start + idx] = minor;
    if (p_edge_weights) { (*weights)[start + idx] = (*p_edge_weights)[i]; }
  }

  return std::make_tuple(std::move(offsets), std::move(indices), std::move(weights));
}

struct Graph_Usecase {
  bool test_weighted{false};
  bool multigraph{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_GraphMask
  : public ::testing::TestWithParam<std::tuple<Graph_Usecase, input_usecase_t>> {
 public:
  Tests_GraphMask() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(std::tuple<Graph_Usecase const&, input_usecase_t const&> const& param)
  {
    raft::handle_t handle{};
    auto [graph_usecase, input_usecase] = param;

    auto [d_srcs, d_dsts, d_weights, d_vertices, number_of_vertices, is_symmetric] =
      input_usecase
        .template construct_edgelist<vertex_t, edge_t, weight_t, store_transposed, false>(
          handle, graph_usecase.test_weighted);

    edge_t number_of_edges = static_cast<edge_t>(d_srcs.size());

    std::vector<vertex_t> h_srcs(number_of_edges);
    std::vector<vertex_t> h_dsts(number_of_edges);
    auto h_weights =
      d_weights ? std::make_optional<std::vector<weight_t>>(number_of_edges) : std::nullopt;

    raft::update_host(h_srcs.data(), d_srcs.data(), number_of_edges, handle.get_stream());
    raft::update_host(h_dsts.data(), d_dsts.data(), number_of_edges, handle.get_stream());
    if (h_weights) {
      raft::update_host(
        (*h_weights).data(), (*d_weights).data(), number_of_edges, handle.get_stream());
    }
    handle.sync_stream();

    auto [h_reference_offsets, h_reference_indices, h_reference_weights] =
      graph_reference<store_transposed>(
        h_srcs.data(),
        h_dsts.data(),
        h_weights ? std::optional<weight_t const*>{(*h_weights).data()} : std::nullopt,
        number_of_vertices,
        number_of_edges);

    cugraph::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_srcs.data(),
      d_dsts.data(),
      d_weights ? std::optional<weight_t const*>{(*d_weights).data()} : std::nullopt,
      number_of_edges};

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto graph = cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false>(
      handle,
      edgelist,
      cugraph::graph_meta_t<vertex_t, edge_t, false>{
        number_of_vertices,
        cugraph::graph_properties_t{is_symmetric, graph_usecase.multigraph},
        std::nullopt},
      true);

    auto graph_view = graph.view();

    handle.sync_stream();

    cugraph::graph_mask_t<vertex_t, edge_t> mask(handle, number_of_vertices, number_of_edges);

    auto mask_view = mask.view();
    graph_view.attach_mask_view(mask_view);

    auto mask_view_from_graph_view = *graph_view.get_mask_view();

    ASSERT_EQ(false, mask.has_vertex_mask());
    ASSERT_EQ(false, mask.has_edge_mask());
    ASSERT_EQ(false, mask_view_from_graph_view.has_vertex_mask());
    ASSERT_EQ(false, mask_view_from_graph_view.has_edge_mask());

    mask.initialize_vertex_mask();
    mask.initialize_edge_mask();

    // Need to create another view to reflect changes to the
    // owning object
    auto mask_view_from_graph_view2 = *graph_view.get_mask_view();

    ASSERT_EQ(true, mask.has_vertex_mask());
    ASSERT_EQ(true, mask.has_edge_mask());
    ASSERT_EQ(true, mask_view_from_graph_view2.has_vertex_mask());
    ASSERT_EQ(true, mask_view_from_graph_view2.has_edge_mask());
  }
};

using Tests_GraphMask_Rmat = Tests_GraphMask<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedFalse_32_32_float)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedFalse_32_64_float)
{
  run_current_test<int32_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedFalse_64_64_float)
{
  run_current_test<int64_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedFalse_32_32_double)
{
  run_current_test<int32_t, int32_t, double, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedFalse_32_64_double)
{
  run_current_test<int32_t, int64_t, double, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedFalse_64_64_double)
{
  run_current_test<int64_t, int64_t, double, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedTrue_32_32_float)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedTrue_32_64_float)
{
  run_current_test<int32_t, int64_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedTrue_64_64_float)
{
  run_current_test<int64_t, int64_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedTrue_32_32_double)
{
  run_current_test<int32_t, int32_t, double, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedTrue_32_64_double)
{
  run_current_test<int32_t, int64_t, double, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_GraphMask_Rmat, CheckStoreTransposedTrue_64_64_double)
{
  run_current_test<int64_t, int64_t, double, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_GraphMask_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Graph_Usecase{false, true}, Graph_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_GraphMask_Rmat,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(Graph_Usecase{false, true, false}, Graph_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
