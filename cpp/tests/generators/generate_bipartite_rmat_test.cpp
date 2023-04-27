/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cugraph/utilities/high_res_timer.hpp>

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
void validate_bipartite_rmat_distribution(
  std::tuple<vertex_t, vertex_t>* edges,
  size_t num_edges,
  vertex_t src_first,
  vertex_t src_last,
  vertex_t dst_first,
  vertex_t dst_last,
  double a,
  double b,
  double c,
  size_t min_edges /* stop recursion if # edges < min_edges */,
  double error_tolerance /* (computed a|b|c - input a|b|c) shoud be smaller than error_tolerance*/)
{
  // we cannot expect the ratios of the edges in the four quadrants of the graph adjacency matrix to
  // converge close to a, b, c, d if num_edges is not large enough.
  if (num_edges < min_edges) { return; }

  auto src_threshold = (src_first + src_last) / 2;
  auto dst_threshold = (dst_first + dst_last) / 2;

  if (src_last - src_first >= 2) {
    auto a_plus_b_last = std::partition(edges, edges + num_edges, [src_threshold](auto edge) {
      return std::get<0>(edge) < src_threshold;
    });
    if (dst_last - dst_first >= 2) {
      auto a_last = std::partition(edges, a_plus_b_last, [dst_threshold](auto edge) {
        return std::get<1>(edge) < dst_threshold;
      });
      auto c_last = std::partition(a_plus_b_last, edges + num_edges, [dst_threshold](auto edge) {
        return std::get<1>(edge) < dst_threshold;
      });

      ASSERT_TRUE(std::abs((double)std::distance(edges, a_last) / num_edges - a) < error_tolerance)
        << "# edges=" << num_edges
        << " computed a=" << (double)std::distance(edges, a_last) / num_edges << " iput a=" << a
        << " error tolerance=" << error_tolerance << ".";
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

      if ((src_threshold - src_first) * (dst_threshold - dst_first) >= 2) {
        validate_bipartite_rmat_distribution(edges,
                                             std::distance(edges, a_last),
                                             src_first,
                                             src_threshold,
                                             dst_first,
                                             dst_threshold,
                                             a,
                                             b,
                                             c,
                                             min_edges,
                                             error_tolerance);
      }
      if ((src_threshold - src_first) * (dst_last - dst_threshold) >= 2) {
        validate_bipartite_rmat_distribution(a_last,
                                             std::distance(a_last, a_plus_b_last),
                                             src_first,
                                             src_threshold,
                                             dst_threshold,
                                             dst_last,
                                             a,
                                             b,
                                             c,
                                             min_edges,
                                             error_tolerance);
      }
      if ((src_last - src_threshold) * (dst_threshold - dst_first) >= 2) {
        validate_bipartite_rmat_distribution(a_plus_b_last,
                                             std::distance(a_plus_b_last, c_last),
                                             src_threshold,
                                             src_last,
                                             dst_first,
                                             dst_threshold,
                                             a,
                                             b,
                                             c,
                                             min_edges,
                                             error_tolerance);
      }
      if ((src_last - src_threshold) * (dst_last - dst_threshold) >= 2) {
        validate_bipartite_rmat_distribution(c_last,
                                             std::distance(c_last, edges + num_edges),
                                             src_threshold,
                                             src_last,
                                             dst_threshold,
                                             dst_last,
                                             a,
                                             b,
                                             c,
                                             min_edges,
                                             error_tolerance);
      }
    } else {
      ASSERT_TRUE(std::abs((double)std::distance(edges, a_plus_b_last) / num_edges - (a + b)) <
                  error_tolerance)
        << "# edges=" << num_edges
        << " computed a+b=" << (double)std::distance(edges, a_plus_b_last) / num_edges
        << " iput a+b=" << (a + b) << " error tolerance=" << error_tolerance << ".";
      if (src_threshold - src_first >= 2) {
        validate_bipartite_rmat_distribution(edges,
                                             std::distance(edges, a_plus_b_last),
                                             src_first,
                                             src_threshold,
                                             dst_first,
                                             dst_last,
                                             a,
                                             b,
                                             c,
                                             min_edges,
                                             error_tolerance);
      }
      if (src_last - src_threshold >= 2) {
        validate_bipartite_rmat_distribution(edges,
                                             std::distance(a_plus_b_last, edges + num_edges),
                                             src_threshold,
                                             src_last,
                                             dst_first,
                                             dst_last,
                                             a,
                                             b,
                                             c,
                                             min_edges,
                                             error_tolerance);
      }
    }
  } else if (dst_last - dst_first >= 2) {
    auto a_plus_c_last = std::partition(edges, edges + num_edges, [dst_threshold](auto edge) {
      return std::get<1>(edge) < dst_threshold;
    });
    ASSERT_TRUE(std::abs((double)std::distance(edges, a_plus_c_last) / num_edges - (a + c)) <
                error_tolerance)
      << "# edges=" << num_edges
      << " computed a+c=" << (double)std::distance(edges, a_plus_c_last) / num_edges
      << " iput a+c=" << (a + c) << " error tolerance=" << error_tolerance << ".";
    if (dst_threshold - dst_first >= 2) {
      validate_bipartite_rmat_distribution(edges,
                                           std::distance(edges, a_plus_c_last),
                                           src_first,
                                           src_last,
                                           dst_first,
                                           dst_threshold,
                                           a,
                                           b,
                                           c,
                                           min_edges,
                                           error_tolerance);
    }
    if (dst_last - dst_threshold >= 2) {
      validate_bipartite_rmat_distribution(edges,
                                           std::distance(a_plus_c_last, edges + num_edges),
                                           src_first,
                                           src_last,
                                           dst_threshold,
                                           dst_last,
                                           a,
                                           b,
                                           c,
                                           min_edges,
                                           error_tolerance);
    }
  }

  return;
}

struct GenerateBipartiteRmat_Usecase {
  size_t src_scale{0};
  size_t dst_scale{0};
  size_t src_edge_factor{0};  // # edges = 2^src_scale * src_edge_factor
  double a{0.0};
  double b{0.0};
  double c{0.0};

  GenerateBipartiteRmat_Usecase(
    size_t src_scale, size_t dst_scale, size_t src_edge_factor, double a, double b, double c)
    : src_scale(src_scale),
      dst_scale(dst_scale),
      src_edge_factor(src_edge_factor),
      a(a),
      b(b),
      c(c){};
};

class Tests_GenerateBipartiteRmat : public ::testing::TestWithParam<GenerateBipartiteRmat_Usecase> {
 public:
  Tests_GenerateBipartiteRmat() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t>
  void run_current_test(GenerateBipartiteRmat_Usecase const& configuration)
  {
    raft::handle_t handle{};
    HighResTimer hr_timer{};

    auto num_src_vertices = static_cast<vertex_t>(size_t{1} << configuration.src_scale);
    auto num_dst_vertices = static_cast<vertex_t>(size_t{1} << configuration.dst_scale);

    std::vector<size_t> no_scramble_out_degrees(num_src_vertices, 0);
    std::vector<size_t> no_scramble_in_degrees(num_dst_vertices, 0);
    std::vector<size_t> scramble_out_degrees(num_src_vertices, 0);
    std::vector<size_t> scramble_in_degrees(num_dst_vertices, 0);
    for (size_t scramble = 0; scramble < 2; ++scramble) {
      raft::random::RngState rng_state(0);

      rmm::device_uvector<vertex_t> d_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> d_dsts(0, handle.get_stream());

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.start("Generate edge list");
      }

      std::tie(d_srcs, d_dsts) = cugraph::generate_bipartite_rmat_edgelist<vertex_t>(
        handle,
        rng_state,
        configuration.src_scale,
        configuration.dst_scale,
        (size_t{1} << configuration.src_scale) * configuration.src_edge_factor,
        configuration.a,
        configuration.b,
        configuration.c);

      if (scramble == 1) {
        d_srcs = cugraph::scramble_vertex_ids(handle, std::move(d_srcs), configuration.src_scale);
        d_dsts = cugraph::scramble_vertex_ids(handle, std::move(d_dsts), configuration.dst_scale);
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      auto h_cugraph_srcs = cugraph::test::to_host(handle, d_srcs);
      auto h_cugraph_dsts = cugraph::test::to_host(handle, d_dsts);

      ASSERT_TRUE((h_cugraph_srcs.size() ==
                   (size_t{1} << configuration.src_scale) * configuration.src_edge_factor) &&
                  (h_cugraph_dsts.size() ==
                   (size_t{1} << configuration.src_scale) * configuration.src_edge_factor))
        << "Returned an invalid number of bipartite R-mat graph edges.";
      ASSERT_TRUE(std::count_if(h_cugraph_srcs.begin(),
                                h_cugraph_srcs.end(),
                                [num_src_vertices](auto v) {
                                  return !cugraph::is_valid_vertex(num_src_vertices, v);
                                }) == 0)
        << "Returned bipartite R-mat graph edges have invalid source vertex IDs.";
      ASSERT_TRUE(std::count_if(h_cugraph_dsts.begin(),
                                h_cugraph_dsts.end(),
                                [num_dst_vertices](auto v) {
                                  return !cugraph::is_valid_vertex(num_dst_vertices, v);
                                }) == 0)
        << "Returned bipartite R-mat graph edges have invalid destination vertex IDs.";

      if (!scramble) {
        std::vector<std::tuple<vertex_t, vertex_t>> h_cugraph_edges(h_cugraph_srcs.size());
        for (size_t i = 0; i < h_cugraph_srcs.size(); ++i) {
          h_cugraph_edges[i] = std::make_tuple(h_cugraph_srcs[i], h_cugraph_dsts[i]);
        }

        validate_bipartite_rmat_distribution(h_cugraph_edges.data(),
                                             h_cugraph_edges.size(),
                                             vertex_t{0},
                                             num_src_vertices,
                                             vertex_t{0},
                                             num_dst_vertices,
                                             configuration.a,
                                             configuration.b,
                                             configuration.c,
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
    // ideally, we should test that the two graphs are isomorphic, but this is NP hard; instead, we
    // just check out-degree & in-degree distributions
    ASSERT_TRUE(std::equal(no_scramble_out_degrees.begin(),
                           no_scramble_out_degrees.end(),
                           scramble_out_degrees.begin()));
    ASSERT_TRUE(std::equal(
      no_scramble_in_degrees.begin(), no_scramble_in_degrees.end(), scramble_in_degrees.begin()));
  }
};

TEST_P(Tests_GenerateBipartiteRmat, CheckInt32) { run_current_test<int32_t>(GetParam()); }
TEST_P(Tests_GenerateBipartiteRmat, CheckInt64) { run_current_test<int64_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_GenerateBipartiteRmat,
  ::testing::Values(GenerateBipartiteRmat_Usecase(20, 10, 16, 0.57, 0.19, 0.19),
                    GenerateBipartiteRmat_Usecase(10, 20, 16, 0.57, 0.19, 0.19),
                    GenerateBipartiteRmat_Usecase(20, 10, 16, 0.45, 0.22, 0.22),
                    GenerateBipartiteRmat_Usecase(10, 20, 16, 0.45, 0.22, 0.22)));

CUGRAPH_TEST_PROGRAM_MAIN()
