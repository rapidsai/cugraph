/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuda_profiler_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

struct MsBfs_Usecase {
  size_t n_edgelists;
  int  radius;
  bool test_weighted_{false};
  bool check_correctness_{true};
};

template <typename input_usecase_t>
class Tests_MsBfs
    : public ::testing::TestWithParam<std::tuple<MsBfs_Usecase, input_usecase_t>> {
 public:
  Tests_MsBfs() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<MsBfs_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber               = false;
    auto [msbfs_usecase, input_usecase] = param;
    raft::handle_t handle{};

    HighResTimer hr_timer{};


    rmm::device_uvector<vertex_t> d_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dsts(0, handle.get_stream());

    vertex_t n_edges = 0, offset = 0, n_vertices = 0;
    std::vector<vertex_t> h_sources;
    rmm::device_uvector<vertex_t> d_sources(h_sources.size(), handle.get_stream());
    {

        std::vector<rmm::device_uvector<vertex_t>> src_chunks{};
        std::vector<rmm::device_uvector<vertex_t>> dst_chunks{};

        std::vector<rmm::device_uvector<vertex_t>> cur_src_chunks{};
        std::vector<rmm::device_uvector<vertex_t>> cur_dst_chunks{};

        for (auto i = 0; i < msbfs_usecase.n_edgelists; ++i) {

            h_sources.push_back(offset);
            std::tie(cur_src_chunks, cur_dst_chunks, std::ignore, std::ignore, std::ignore) =
            input_usecase.template construct_edgelist<vertex_t, weight_t>(handle, false, false, false, true, i);

            // v offset is max of src/dst
            auto max_src = thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                        cur_src_chunks[0].begin(),
                                        cur_src_chunks[0].end(),
                                        static_cast<vertex_t>(0),
                                        thrust::maximum<vertex_t>());

            auto max_dst = thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                        cur_dst_chunks[0].begin(),
                                        cur_dst_chunks[0].end(),
                                        static_cast<vertex_t>(0),
                                        thrust::maximum<vertex_t>());
            
            //std::cout<<"max_src = " << max_src << " max_dst = " << max_dst << std::endl;

            offset = std::max(max_src, max_dst) + 1;

            //std::cout<<"max_vertex = " << offset << std::endl;
            
            //std::cout<<"i = " << i << " num_edges = " << cur_src_chunks[0].size() << " max_vertex = " << offset << std::endl;

            //raft::print_device_vector("d_sources", cur_src_chunks[0].data(), 10, std::cout);
            //raft::print_device_vector("d_sources", cur_dst_chunks[0].data(), 10, std::cout);
            src_chunks.push_back(std::move(cur_src_chunks[0]));
            dst_chunks.push_back(std::move(cur_dst_chunks[0]));

        }

        raft::copy(d_sources.data(), h_sources.data(), h_sources.size(), handle.get_stream());

        std::cout<<"src_chunks size = " << src_chunks.size() << std::endl;

        std::tie(d_srcs, d_dsts, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
            cugraph::test::detail::
                concatenate_edge_chunks<vertex_t, edge_t, weight_t, int32_t, int32_t>(
                handle,
                std::move(src_chunks),
                std::move(dst_chunks),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt);
            
        n_edges = d_srcs.size(); // As we count the number of edges, increase this counter
    }

    n_vertices = offset;

    std::cout<<"n_vertices = " << n_vertices << " num_edges = " << d_srcs.size() << std::endl;

    rmm::device_uvector<vertex_t> d_vertices(n_vertices, handle.get_stream());

    thrust::sequence(
      rmm::exec_policy(handle.get_stream()), d_vertices.begin(), d_vertices.end(), vertex_t{0});

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("SG Construct graph");
    }

    // create the graph
    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);

    std::tie(graph, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
      cugraph::
        create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, int32_t, false, false>(
          handle,
          std::move(d_vertices),
          std::move(d_srcs),
          std::move(d_dsts),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cugraph::graph_properties_t{false, true},
          false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    //std::cout<<"number of vertices = " << graph_view.number_of_vertices() << std::endl;
    
    
    std::vector<rmm::device_uvector<vertex_t>> d_distances_ref{};
    std::vector<rmm::device_uvector<vertex_t>> d_predecessors_ref{};
    std::vector<std::vector<vertex_t>> h_distances_ref(h_sources.size());
    std::vector<std::vector<vertex_t>> h_predecessors_ref(h_sources.size());

    d_distances_ref.reserve(h_sources.size());
    d_predecessors_ref.reserve(h_sources.size());
    for (size_t i = 0; i < h_sources.size(); i++) {
      rmm::device_uvector<vertex_t> tmp_distances(graph_view.number_of_vertices(),
                                                  handle.get_next_usable_stream(i));
      rmm::device_uvector<vertex_t> tmp_predecessors(graph_view.number_of_vertices(),
                                                     handle.get_next_usable_stream(i));

      d_distances_ref.push_back(std::move(tmp_distances));
      d_predecessors_ref.push_back(std::move(tmp_predecessors));
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MsBFS");
    }

    bool direction_optimizing = false;

    vertex_t source = h_sources[0];


    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());
    
    for (size_t i = 0; i < h_sources.size(); i++) {
      source = h_sources[i];
      rmm::device_scalar<vertex_t> const d_source_i(source, handle.get_stream());
      cugraph::bfs(handle,
                   graph_view,
                   d_distances_ref[i].begin(),
                   d_predecessors_ref[i].begin(),
                   d_source_i.data(),
                   size_t{1},
                   direction_optimizing,
                   msbfs_usecase.radius);
    }


    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }


    // checksum
    vertex_t ref_sum = 0;
    for (size_t i = 0; i < h_sources.size(); i++) {
      thrust::replace(rmm::exec_policy(handle.get_stream()),
                      d_distances_ref[i].begin(),
                      d_distances_ref[i].end(),
                      std::numeric_limits<vertex_t>::max(),
                      static_cast<vertex_t>(0));
      ref_sum += thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                d_distances_ref[i].begin(),
                                d_distances_ref[i].end(),
                                static_cast<vertex_t>(0));
    }
    thrust::replace(rmm::exec_policy(handle.get_stream()),
                    d_distances.begin(),
                    d_distances.end(),
                    std::numeric_limits<vertex_t>::max(),
                    static_cast<vertex_t>(0));
    vertex_t ms_sum = thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                     d_distances.begin(),
                                     d_distances.end(),
                                     static_cast<vertex_t>(0));

    auto d_vertex_degree = graph_view.compute_out_degrees(handle);

    thrust::sort(rmm::exec_policy(handle.get_stream()), d_sources.begin(), d_sources.end());

    d_vertices.resize(graph_view.number_of_vertices(), handle.get_stream());

    thrust::sequence(
      rmm::exec_policy(handle.get_stream()), d_vertices.begin(), d_vertices.end(), vertex_t{0});

    auto seeds_degree_last = thrust::partition(
      rmm::exec_policy(handle.get_stream()),
      thrust::make_zip_iterator(d_vertex_degree.begin(), d_vertices.begin()),
      thrust::make_zip_iterator(d_vertex_degree.end(), d_vertices.end()),
      [d_sources = raft::device_span<vertex_t>(d_sources.data(), d_sources.size())] __device__(
        auto pair_vertex_degree) {
        return thrust::binary_search(
          thrust::seq, d_sources.begin(), d_sources.end(), thrust::get<1>(pair_vertex_degree));
      });

    auto seeds_degree_size = thrust::distance(
      thrust::make_zip_iterator(d_vertex_degree.begin(), d_vertices.begin()), seeds_degree_last);

    d_vertex_degree.resize(seeds_degree_size, handle.get_stream());

    auto seeds_degree_sum = thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                           d_vertex_degree.begin(),
                                           d_vertex_degree.end(),
                                           static_cast<vertex_t>(0),
                                           thrust::plus<vertex_t>());

    // Check the degree of the seed vertices
    // If degree of all seeds is zero hence ref_sum is zero otherwise ref_sum > 0
    ASSERT_TRUE(seeds_degree_sum ? ref_sum > 0 : ref_sum == 0);
    ASSERT_TRUE(ref_sum < std::numeric_limits<vertex_t>::max());
    ASSERT_TRUE(ref_sum == ms_sum);


    
    
    
    
    //"""
    /*
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("K-truss");
    }

    auto [d_cugraph_srcs, d_cugraph_dsts, d_cugraph_wgts] =
      cugraph::k_truss<vertex_t, edge_t, weight_t, false>(
        handle,
        graph_view,
        edge_weight ? std::make_optional((*edge_weight).view()) : std::nullopt,
        msbfs_usecase.k_,
        false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (msbfs_usecase.check_correctness_) {
      std::optional<cugraph::graph_t<vertex_t, edge_t, false, false>> modified_graph{std::nullopt};
      auto [h_offsets, h_indices, h_values] = cugraph::test::graph_to_host_csr(
        handle,
        graph_view,
        edge_weight ? std::make_optional((*edge_weight).view()) : std::nullopt,
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));

      rmm::device_uvector<weight_t> d_sorted_cugraph_wgts{0, handle.get_stream()};
      rmm::device_uvector<vertex_t> d_sorted_cugraph_srcs{0, handle.get_stream()};
      rmm::device_uvector<vertex_t> d_sorted_cugraph_dsts{0, handle.get_stream()};

      if (edge_weight) {
        std::tie(d_sorted_cugraph_srcs, d_sorted_cugraph_dsts, d_sorted_cugraph_wgts) =
          cugraph::test::sort_by_key<vertex_t, weight_t>(
            handle, d_cugraph_srcs, d_cugraph_dsts, *d_cugraph_wgts);
      } else {
        std::tie(d_sorted_cugraph_srcs, d_sorted_cugraph_dsts) =
          cugraph::test::sort<vertex_t>(handle, d_cugraph_srcs, d_cugraph_dsts);
      }

      auto h_cugraph_srcs = cugraph::test::to_host(handle, d_sorted_cugraph_srcs);

      auto h_cugraph_dsts = cugraph::test::to_host(handle, d_sorted_cugraph_dsts);

      auto [h_reference_srcs, h_reference_dsts, h_reference_wgts] =
        k_truss_reference<vertex_t, edge_t, weight_t>(
          h_offsets, h_indices, h_values, msbfs_usecase.k_);

      EXPECT_EQ(h_cugraph_srcs.size(), h_reference_srcs.size());
      ASSERT_TRUE(
        std::equal(h_cugraph_srcs.begin(), h_cugraph_srcs.end(), h_reference_srcs.begin()));

      ASSERT_TRUE(
        std::equal(h_cugraph_dsts.begin(), h_cugraph_dsts.end(), h_reference_dsts.begin()));

      if (edge_weight) {
        auto h_cugraph_wgts  = cugraph::test::to_host(handle, d_sorted_cugraph_wgts);
        auto compare_functor = host_nearly_equal<weight_t>{
          weight_t{1e-3},
          weight_t{(weight_t{1} / static_cast<weight_t>((h_cugraph_wgts).size())) *
                   weight_t{1e-3}}};
        EXPECT_TRUE(std::equal((h_cugraph_wgts).begin(),
                               (h_cugraph_wgts).end(),
                               (*h_reference_wgts).begin(),
                               compare_functor));
      }
    }
    */
    //"""
  }
};

//using Tests_MsBfs_File = Tests_MsBfs<cugraph::test::File_Usecase>;
using Tests_MsBfs_Rmat = Tests_MsBfs<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MsBfs_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}
/*
TEST_P(Tests_MsBfs_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}
*/


INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MsBfs_Rmat,
                         // enable correctness checks
                         ::testing::Combine(::testing::Values(MsBfs_Usecase{8, 2, false, false}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false))));



CUGRAPH_TEST_PROGRAM_MAIN()
