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

#include <tuple>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_generators.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

// this function assumes that vertex IDs are not scrambled
template <typename vertex_t>
void validate_rmat_distribution(
  std::tuple<vertex_t, vertex_t>* edges,
  size_t num_edges,
  vertex_t src_first,
  vertex_t src_last,
  vertex_t dst_first,
  vertex_t dst_last,
  double a,
  double b,
  double c,
  bool clip_and_flip,
  size_t min_edges /* stop recursion if # edges < min_edges */,
  double error_tolerance /* (computed a|b|c - input a|b|c) shoud be smaller than error_tolerance*/)
{
  // we cannot expect the ratios of the edges in the four quadrants of the graph adjacency matrix to
  // converge close to a, b, c, d if num_edges is not large enough.
  if (num_edges < min_edges) { return; }

  auto src_threshold = (src_first + src_last) / 2;
  auto dst_threshold = (dst_first + dst_last) / 2;

  auto a_plus_b_last = std::partition(edges, edges + num_edges, [src_threshold](auto edge) {
    return std::get<0>(edge) < src_threshold;
  });
  auto a_last        = std::partition(
    edges, a_plus_b_last, [dst_threshold](auto edge) { return std::get<1>(edge) < dst_threshold; });
  auto c_last = std::partition(a_plus_b_last, edges + num_edges, [dst_threshold](auto edge) {
    return std::get<1>(edge) < dst_threshold;
  });

  ASSERT_TRUE(std::abs((double)std::distance(edges, a_last) / num_edges - a) < error_tolerance)
    << "# edges=" << num_edges << " computed a=" << (double)std::distance(edges, a_last) / num_edges
    << " iput a=" << a << " error tolerance=" << error_tolerance << ".";
  if (clip_and_flip && (src_first == dst_first) &&
      (src_last == dst_last)) {  // if clip_and_flip and in the diagonal
    ASSERT_TRUE(std::distance(a_last, a_plus_b_last) == 0);
    ASSERT_TRUE(std::abs((double)std::distance(a_plus_b_last, c_last) / num_edges - (b + c)) <
                error_tolerance)
      << "# edges=" << num_edges
      << " computed c=" << (double)std::distance(a_plus_b_last, c_last) / num_edges
      << " iput (b + c)=" << (b + c) << " error tolerance=" << error_tolerance << ".";
  } else {
    ASSERT_TRUE(std::abs((double)std::distance(a_last, a_plus_b_last) / num_edges - b) <
                error_tolerance)
      << "# edges=" << num_edges
      << " computed b=" << (double)std::distance(a_last, a_plus_b_last) / num_edges
      << " iput b=" << b << " error tolerance=" << error_tolerance << ".";
    ASSERT_TRUE(std::abs((double)std::distance(a_plus_b_last, c_last) / num_edges - c) <
                error_tolerance)
      << "# edges=" << num_edges
      << " computed c=" << (double)std::distance(a_plus_b_last, c_last) / num_edges
      << " iput c=" << c << " error tolerance=" << error_tolerance << ".";
  }

  validate_rmat_distribution(edges,
                             std::distance(edges, a_last),
                             src_first,
                             src_threshold,
                             dst_first,
                             dst_threshold,
                             a,
                             b,
                             c,
                             clip_and_flip,
                             min_edges,
                             error_tolerance);
  validate_rmat_distribution(a_last,
                             std::distance(a_last, a_plus_b_last),
                             src_first,
                             (src_first + src_last) / 2,
                             dst_threshold,
                             dst_last,
                             a,
                             b,
                             c,
                             clip_and_flip,
                             min_edges,
                             error_tolerance);
  validate_rmat_distribution(a_plus_b_last,
                             std::distance(a_plus_b_last, c_last),
                             src_threshold,
                             src_last,
                             dst_first,
                             dst_threshold,
                             a,
                             b,
                             c,
                             clip_and_flip,
                             min_edges,
                             error_tolerance);
  validate_rmat_distribution(c_last,
                             std::distance(c_last, edges + num_edges),
                             src_threshold,
                             src_last,
                             dst_threshold,
                             dst_last,
                             a,
                             b,
                             c,
                             clip_and_flip,
                             min_edges,
                             error_tolerance);

  return;
}

typedef struct GenerateRmat_Usecase_t {
  size_t scale{0};
  size_t edge_factor{0};
  double a{0.0};
  double b{0.0};
  double c{0.0};
  bool clip_and_flip{false};

  GenerateRmat_Usecase_t(
    size_t scale, size_t edge_factor, double a, double b, double c, bool clip_and_flip)
    : scale(scale), edge_factor(edge_factor), a(a), b(b), c(c), clip_and_flip(clip_and_flip){};
} GenerateRmat_Usecase;

class Tests_GenerateRmat : public ::testing::TestWithParam<GenerateRmat_Usecase> {
 public:
  Tests_GenerateRmat() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t>
  void run_current_test(GenerateRmat_Usecase const& configuration)
  {
    raft::handle_t handle{};

    auto num_vertices = static_cast<vertex_t>(size_t{1} << configuration.scale);
    std::vector<size_t> no_scramble_out_degrees(num_vertices, 0);
    std::vector<size_t> no_scramble_in_degrees(num_vertices, 0);
    std::vector<size_t> scramble_out_degrees(num_vertices, 0);
    std::vector<size_t> scramble_in_degrees(num_vertices, 0);
    for (size_t scramble = 0; scramble < 2; ++scramble) {
      rmm::device_uvector<vertex_t> d_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> d_dsts(0, handle.get_stream());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

      std::tie(d_srcs, d_dsts) = cugraph::generate_rmat_edgelist<vertex_t>(
        handle,
        configuration.scale,
        (size_t{1} << configuration.scale) * configuration.edge_factor,
        configuration.a,
        configuration.b,
        configuration.c,
        uint64_t{0},
        configuration.clip_and_flip);
      // static_cast<bool>(scramble));

      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

      auto h_cugraph_srcs = cugraph::test::to_host(handle, d_srcs);
      auto h_cugraph_dsts = cugraph::test::to_host(handle, d_dsts);

      ASSERT_TRUE(
        (h_cugraph_srcs.size() == (size_t{1} << configuration.scale) * configuration.edge_factor) &&
        (h_cugraph_dsts.size() == (size_t{1} << configuration.scale) * configuration.edge_factor))
        << "Returned an invalid number of R-mat graph edges.";
      ASSERT_TRUE(
        std::count_if(h_cugraph_srcs.begin(),
                      h_cugraph_srcs.end(),
                      [num_vertices = static_cast<vertex_t>(size_t{1} << configuration.scale)](
                        auto v) { return !cugraph::is_valid_vertex(num_vertices, v); }) == 0)
        << "Returned R-mat graph edges have invalid source vertex IDs.";
      ASSERT_TRUE(
        std::count_if(h_cugraph_dsts.begin(),
                      h_cugraph_dsts.end(),
                      [num_vertices = static_cast<vertex_t>(size_t{1} << configuration.scale)](
                        auto v) { return !cugraph::is_valid_vertex(num_vertices, v); }) == 0)
        << "Returned R-mat graph edges have invalid destination vertex IDs.";

      if (!scramble) {
        if (configuration.clip_and_flip) {
          for (size_t i = 0; i < h_cugraph_srcs.size(); ++i) {
            ASSERT_TRUE(h_cugraph_srcs[i] >= h_cugraph_dsts[i]);
          }
        }

        std::vector<std::tuple<vertex_t, vertex_t>> h_cugraph_edges(h_cugraph_srcs.size());
        for (size_t i = 0; i < h_cugraph_srcs.size(); ++i) {
          h_cugraph_edges[i] = std::make_tuple(h_cugraph_srcs[i], h_cugraph_dsts[i]);
        }

        validate_rmat_distribution(h_cugraph_edges.data(),
                                   h_cugraph_edges.size(),
                                   vertex_t{0},
                                   num_vertices,
                                   vertex_t{0},
                                   num_vertices,
                                   configuration.a,
                                   configuration.b,
                                   configuration.c,
                                   configuration.clip_and_flip,
                                   size_t{100000},
                                   0.01);
      }

      if (scramble) {
        std::for_each(h_cugraph_srcs.begin(),
                      h_cugraph_srcs.end(),
                      [&scramble_out_degrees](auto src) { scramble_out_degrees[src]++; });
        std::for_each(h_cugraph_dsts.begin(),
                      h_cugraph_dsts.end(),
                      [&scramble_in_degrees](auto dst) { scramble_in_degrees[dst]++; });
        std::sort(scramble_out_degrees.begin(), scramble_out_degrees.end());
        std::sort(scramble_in_degrees.begin(), scramble_in_degrees.end());
      } else {
        std::for_each(h_cugraph_srcs.begin(),
                      h_cugraph_srcs.end(),
                      [&no_scramble_out_degrees](auto src) { no_scramble_out_degrees[src]++; });
        std::for_each(h_cugraph_dsts.begin(),
                      h_cugraph_dsts.end(),
                      [&no_scramble_in_degrees](auto dst) { no_scramble_in_degrees[dst]++; });
        std::sort(no_scramble_out_degrees.begin(), no_scramble_out_degrees.end());
        std::sort(no_scramble_in_degrees.begin(), no_scramble_in_degrees.end());
      }
    }

    // this relies on the fact that the edge generator is deterministic.
    // ideally, we should test that the two graphs are isomorphic, but this is NP hard; insted, we
    // just check out-degree & in-degree distributions
    ASSERT_TRUE(std::equal(no_scramble_out_degrees.begin(),
                           no_scramble_out_degrees.end(),
                           scramble_out_degrees.begin()));
    ASSERT_TRUE(std::equal(
      no_scramble_in_degrees.begin(), no_scramble_in_degrees.end(), scramble_in_degrees.begin()));
  }
};

// FIXME: add tests for type combinations

TEST_P(Tests_GenerateRmat, CheckInt32) { run_current_test<int32_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(simple_test,
                         Tests_GenerateRmat,
                         ::testing::Values(GenerateRmat_Usecase(20, 16, 0.57, 0.19, 0.19, true),
                                           GenerateRmat_Usecase(20, 16, 0.57, 0.19, 0.19, false),
                                           GenerateRmat_Usecase(20, 16, 0.45, 0.22, 0.22, true),
                                           GenerateRmat_Usecase(20, 16, 0.45, 0.22, 0.22, false)));
typedef struct GenerateRmats_Usecase_t {
  size_t n_edgelists{0};
  size_t min_scale{0};
  size_t max_scale{0};
  size_t edge_factor{0};
  cugraph::generator_distribution_t component_distribution;
  cugraph::generator_distribution_t edge_distribution;

  GenerateRmats_Usecase_t(size_t n_edgelists,
                          size_t min_scale,
                          size_t max_scale,
                          size_t edge_factor,
                          cugraph::generator_distribution_t component_distribution,
                          cugraph::generator_distribution_t edge_distribution)
    : n_edgelists(n_edgelists),
      min_scale(min_scale),
      max_scale(max_scale),
      component_distribution(component_distribution),
      edge_distribution(edge_distribution),
      edge_factor(edge_factor){};
} GenerateRmats_Usecase;
class Tests_GenerateRmats : public ::testing::TestWithParam<GenerateRmats_Usecase> {
 public:
  Tests_GenerateRmats() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t>
  void run_current_test(GenerateRmats_Usecase const& configuration)
  {
    raft::handle_t handle{};

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto outputs = cugraph::generate_rmat_edgelists<vertex_t>(handle,
                                                              configuration.n_edgelists,
                                                              configuration.min_scale,
                                                              configuration.max_scale,
                                                              configuration.edge_factor,
                                                              configuration.component_distribution,
                                                              configuration.edge_distribution,
                                                              uint64_t{0});

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
    ASSERT_EQ(configuration.n_edgelists, outputs.size());
    for (auto i = outputs.begin(); i != outputs.end(); ++i) {
      ASSERT_EQ(std::get<0>(*i).size(), std::get<1>(*i).size());
      ASSERT_TRUE((configuration.min_scale * configuration.edge_factor) <= std::get<0>(*i).size());
      ASSERT_TRUE((configuration.max_scale * configuration.edge_factor) >= std::get<0>(*i).size());
    }
  }
};
TEST_P(Tests_GenerateRmats, CheckInt32) { run_current_test<int32_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_GenerateRmats,
  ::testing::Values(GenerateRmats_Usecase(8,
                                          1,
                                          16,
                                          32,
                                          cugraph::generator_distribution_t::UNIFORM,
                                          cugraph::generator_distribution_t::UNIFORM),
                    GenerateRmats_Usecase(8,
                                          1,
                                          16,
                                          32,
                                          cugraph::generator_distribution_t::UNIFORM,
                                          cugraph::generator_distribution_t::POWER_LAW),
                    GenerateRmats_Usecase(8,
                                          3,
                                          16,
                                          32,
                                          cugraph::generator_distribution_t::POWER_LAW,
                                          cugraph::generator_distribution_t::UNIFORM),
                    GenerateRmats_Usecase(8,
                                          3,
                                          16,
                                          32,
                                          cugraph::generator_distribution_t::POWER_LAW,
                                          cugraph::generator_distribution_t::POWER_LAW)));
CUGRAPH_TEST_PROGRAM_MAIN()
