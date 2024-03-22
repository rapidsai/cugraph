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

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <thrust/set_operations.h>

#include <gtest/gtest.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

struct KTruss_Usecase {
  int32_t k{3};
  bool check_correctness{true};
  bool test_weighted_{false};
};

template <typename input_usecase_t>
class Tests_KTruss : public ::testing::TestWithParam<std::tuple<KTruss_Usecase, input_usecase_t>> {
 public:
  Tests_KTruss() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>>
  k_truss_reference(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    edge_t k)
  {
    rmm::device_uvector<vertex_t> d_src(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dst(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> d_wgt{std::nullopt};

    // Decompress the edgelist - For Rmat generated graph, remove self loop and multi edges
    std::tie(d_src, d_dst, d_wgt, std::ignore) = decompress_to_edgelist(
      handle,
      graph_view,
      edge_weight_view ? std::make_optional(*edge_weight_view) : std::nullopt,
      std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      std::optional<raft::device_span<vertex_t const>>(std::nullopt));

    // get host vectors

    std::vector<vertex_t> h_src(d_src.size());
    raft::update_host(h_src.data(), d_src.data(), d_src.size(), handle.get_stream());
    std::vector<vertex_t> h_dst(d_src.size());
    raft::update_host(h_dst.data(), d_dst.data(), d_dst.size(), handle.get_stream());

    std::vector<weight_t> h_wgt(d_src.size());
    if (edge_weight_view) {
      raft::update_host(h_wgt.data(), (*d_wgt).data(), (*d_wgt).size(), handle.get_stream());
    }

    std::vector<vertex_t> h_src_(h_src.size());
    std::vector<vertex_t> h_dst_(h_dst.size());
    // FIXME: Instead extract the vertices from the graph
    std::vector<vertex_t> dup_vertices(2 * h_src.size());

    thrust::copy(thrust::host, h_src.begin(), h_src.end(), dup_vertices.begin());
    thrust::copy(thrust::host, h_dst.begin(), h_dst.end(), dup_vertices.begin() + h_src.size());

    std::set<vertex_t> vertices(dup_vertices.begin(), dup_vertices.end());

    auto n_dropped = 1;

    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(h_src.begin(), h_dst.begin()));

    auto weighted_edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(h_src.begin(), h_dst.begin(), h_wgt.begin()));

    thrust::sort(thrust::host, edge_first, edge_first + h_src.size());

    while (n_dropped > 0) {
      n_dropped = 0;
      std::set<vertex_t> seen;
      // Go over all the vertices
      for (auto u = vertices.begin(); u != vertices.end(); ++u) {
        // Find all neighbours of u
        vertex_t idx_start   = 0;
        vertex_t idx_end     = 0;
        auto itr_lower_range = thrust::lower_bound(thrust::host, h_src.begin(), h_src.end(), *u);

        idx_start = thrust::distance(h_src.begin(), itr_lower_range);

        if (*itr_lower_range == *u) {
          auto itr_upper_range =
            thrust::upper_bound(thrust::host, itr_lower_range, h_src.end(), *u);

          idx_end = thrust::distance(itr_lower_range, itr_upper_range);
        }

        std::set<vertex_t> nbrs_u;

        for (edge_t i = idx_start; i < idx_start + idx_end; ++i) {
          nbrs_u.insert(*(h_dst.begin() + i));
        }

        seen.insert(*u);

        std::set<vertex_t> new_nbrs;

        std::set_difference(nbrs_u.begin(),
                            nbrs_u.end(),
                            seen.begin(),
                            seen.end(),
                            std::inserter(new_nbrs, new_nbrs.end()));

        // Finding the neighbors of v
        for (auto v = new_nbrs.begin(); v != new_nbrs.end(); ++v) {
          auto itr_lower_range = thrust::lower_bound(thrust::host, h_src.begin(), h_src.end(), *v);

          idx_start = thrust::distance(h_src.begin(), itr_lower_range);

          if (*itr_lower_range == *v) {
            auto itr_upper_range =
              thrust::upper_bound(thrust::host, itr_lower_range, h_src.end(), *v);

            idx_end = thrust::distance(itr_lower_range, itr_upper_range);
          }
          std::set<vertex_t> nbrs_v;

          for (edge_t i = idx_start; i < idx_start + idx_end; ++i) {
            nbrs_v.insert(*(h_dst.begin() + i));
          }

          std::set<vertex_t> nbr_intersection_u_v;
          // Find the intersection of nbr_u and nbr_v
          std::set_intersection(nbrs_u.begin(),
                                nbrs_u.end(),
                                nbrs_v.begin(),
                                nbrs_v.end(),
                                std::inserter(nbr_intersection_u_v, nbr_intersection_u_v.end()));

          if (nbr_intersection_u_v.size() < (k - 2)) {
            auto edge = thrust::make_tuple(vertex_t{*u}, vertex_t{*v});

            if (edge_weight_view) {
              auto weighted_edge_last = thrust::remove_if(
                thrust::host,
                weighted_edge_first,
                weighted_edge_first + h_src.size(),
                [edge](auto e) {
                  auto src         = thrust::get<0>(e);
                  auto dst         = thrust::get<1>(e);
                  auto remove_edge = thrust::make_tuple(src, dst);
                  return (remove_edge == edge || thrust::make_tuple(dst, src) == edge);
                }

              );

              h_src.resize(thrust::distance(weighted_edge_first, weighted_edge_last));
              h_dst.resize(thrust::distance(weighted_edge_first, weighted_edge_last));
              h_wgt.resize(thrust::distance(weighted_edge_first, weighted_edge_last));
            }
            auto edge_last = thrust::remove_if(
              thrust::host,
              edge_first,
              edge_first + h_src.size(),
              [edge](auto e) {
                auto src         = thrust::get<0>(e);
                auto dst         = thrust::get<1>(e);
                auto remove_edge = thrust::make_tuple(src, dst);
                return (remove_edge == edge || thrust::make_tuple(dst, src) == edge);
              }

            );

            h_src.resize(thrust::distance(edge_first, edge_last));
            h_dst.resize(thrust::distance(edge_first, edge_last));

            n_dropped += 1;
          }
        }
      }
    }

    // convert back to device_vector
    rmm::device_uvector<vertex_t> d_reference_src(h_src.size(), handle.get_stream());

    raft::update_device(d_reference_src.data(), h_src.data(), h_src.size(), handle.get_stream());

    rmm::device_uvector<vertex_t> d_reference_dst(h_src.size(), handle.get_stream());

    raft::update_device(d_reference_dst.data(), h_dst.data(), h_dst.size(), handle.get_stream());

    auto d_reference_wgt =
      std::make_optional<rmm::device_uvector<weight_t>>(h_src.size(), handle.get_stream());

    if (edge_weight_view) {
      raft::update_device(
        (*d_reference_wgt).data(), h_wgt.data(), h_wgt.size(), handle.get_stream());
    }

    return std::make_tuple(std::move(d_reference_src),
                           std::move(d_reference_dst),
                           edge_weight_view ? std::move(d_reference_wgt) : std::nullopt);
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<KTruss_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber = false;

    auto [k_truss_usecase, input_usecase] = param;

    raft::handle_t handle{};

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MG Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, k_truss_usecase.test_weighted_, renumber, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("K-truss");
    }

    auto [d_cugraph_src, d_cugraph_dst, d_cugraph_wgt] =
      cugraph::k_truss<vertex_t, edge_t, weight_t, false>(
        handle, graph_view, edge_weight_view, k_truss_usecase.k, false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (k_truss_usecase.check_correctness) {
      auto [d_reference_src, d_reference_dst, d_reference_wgt] =
        k_truss_reference<vertex_t, edge_t, weight_t>(
          handle, graph_view, edge_weight_view, k_truss_usecase.k);
      if (edge_weight_view) {
        auto d_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
          d_cugraph_src.begin(), d_cugraph_dst.begin(), (*d_cugraph_wgt).begin()));

        thrust::sort(handle.get_thrust_policy(), d_edge_first, d_edge_first + d_cugraph_src.size());
      } else {
        auto d_edge_first = thrust::make_zip_iterator(
          thrust::make_tuple(d_cugraph_src.begin(), d_cugraph_dst.begin()));

        thrust::sort(handle.get_thrust_policy(), d_edge_first, d_edge_first + d_cugraph_src.size());
      }

      EXPECT_EQ(d_cugraph_src.size(), d_reference_src.size());

      ASSERT_TRUE(thrust::equal(handle.get_thrust_policy(),
                                d_cugraph_src.begin(),
                                d_cugraph_src.end(),
                                d_reference_src.begin()));

      ASSERT_TRUE(thrust::equal(handle.get_thrust_policy(),
                                d_cugraph_dst.begin(),
                                d_cugraph_dst.end(),
                                d_reference_dst.begin()));

      if (edge_weight_view) {
        auto compare_functor = cugraph::test::device_nearly_equal<weight_t>{
          weight_t{1e-3},
          weight_t{(weight_t{1} / static_cast<weight_t>((*d_cugraph_wgt).size())) *
                   weight_t{1e-3}}};
        EXPECT_TRUE(thrust::equal(handle.get_thrust_policy(),
                                  (*d_cugraph_wgt).begin(),
                                  (*d_cugraph_wgt).end(),
                                  (*d_reference_wgt).begin(),
                                  compare_functor));
      }
    }
  }
};

using Tests_KTruss_File = Tests_KTruss<cugraph::test::File_Usecase>;

TEST_P(Tests_KTruss_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_KTruss_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_KTruss_File, CheckInt32Int64Float)
{
  run_current_test<int32_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_KTruss_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KTruss_Usecase{5, true, false},
                      KTruss_Usecase{4, true, true},
                      KTruss_Usecase{9, true, true},
                      KTruss_Usecase{7, true, true}),
    ::testing::Values(
      cugraph::test::File_Usecase("/home/nfs/jnke/ktruss/cugraph/datasets/dolphins.mtx"),
      cugraph::test::File_Usecase("/home/nfs/jnke/ktruss/cugraph/datasets/karate.mtx"))));

CUGRAPH_TEST_PROGRAM_MAIN()
