/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

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

#include <cuda_profiler_api.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <tuple>
#include <utilities/thrust_wrapper.hpp>
#include <vector>

struct MsBfs_Usecase {
  size_t n_edgelists;
  size_t min_scale;
  size_t max_scale;
  size_t edge_factor;
  int radius;
};

template <typename vertex_t>
void translate_vertex_ids(raft::handle_t const& handle,
                          rmm::device_uvector<vertex_t>& d_src_v,
                          rmm::device_uvector<vertex_t>& d_dst_v,
                          vertex_t vertex_id_offset)
{
  thrust::transform(rmm::exec_policy(handle.get_stream()),
                    d_src_v.begin(),
                    d_src_v.end(),
                    d_src_v.begin(),
                    [offset = vertex_id_offset] __device__(vertex_t v) { return offset + v; });

  thrust::transform(rmm::exec_policy(handle.get_stream()),
                    d_dst_v.begin(),
                    d_dst_v.end(),
                    d_dst_v.begin(),
                    [offset = vertex_id_offset] __device__(vertex_t v) { return offset + v; });
}

class Tests_MsBfs : public ::testing::TestWithParam<MsBfs_Usecase> {
 public:
  Tests_MsBfs() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(MsBfs_Usecase const& configuration)
  {
    using weight_t   = float;
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(16);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

    auto edgelists =
      cugraph::generate_rmat_edgelists<vertex_t>(handle,
                                                 configuration.n_edgelists,
                                                 configuration.min_scale,
                                                 configuration.max_scale,
                                                 configuration.edge_factor,
                                                 cugraph::generator_distribution_t::POWER_LAW,
                                                 cugraph::generator_distribution_t::UNIFORM,
                                                 uint64_t{0});
    // form aggregated edge list
    vertex_t n_edges = 0, offset = 0, n_vertices = 0;
    std::vector<vertex_t> h_sources;
    for (auto i = edgelists.begin(); i != edgelists.end(); ++i) {
      // translate
      translate_vertex_ids(handle, std::get<0>(*i), std::get<1>(*i), offset);

      n_edges += std::get<0>(*i).size();
      // populating sources with the smallest v_id in the component
      h_sources.push_back(offset);

      // v offset is max of src/dst
      auto max_src = thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                    std::get<0>(*i).begin(),
                                    std::get<0>(*i).end(),
                                    static_cast<vertex_t>(0),
                                    thrust::maximum<vertex_t>());

      auto max_dst = thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                    std::get<1>(*i).begin(),
                                    std::get<1>(*i).end(),
                                    static_cast<vertex_t>(0),
                                    thrust::maximum<vertex_t>());

      offset = std::max(max_src, max_dst) + 1;
    }
    n_vertices = offset;
    std::cout << n_vertices << std::endl;
    std::cout << n_edges << std::endl;

    rmm::device_uvector<vertex_t> d_srcs(n_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dst(n_edges, handle.get_stream());
    auto it_src = d_srcs.begin();
    auto it_dst = d_dst.begin();
    for (auto i = edgelists.begin(); i != edgelists.end(); ++i) {
      it_src = thrust::copy(rmm::exec_policy(handle.get_stream()),
                            std::get<0>(*i).begin(),
                            std::get<0>(*i).end(),
                            it_src);
      it_dst = thrust::copy(rmm::exec_policy(handle.get_stream()),
                            std::get<1>(*i).begin(),
                            std::get<1>(*i).end(),
                            it_dst);
    }

    rmm::device_uvector<vertex_t> d_sources(h_sources.size(), handle.get_stream());
    raft::copy(d_sources.data(), h_sources.data(), h_sources.size(), handle.get_stream());

    // create the graph
    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_vertices(n_vertices, handle.get_stream());
    rmm::device_uvector<weight_t> d_weights(n_edges, handle.get_stream());
    thrust::sequence(
      rmm::exec_policy(handle.get_stream()), d_vertices.begin(), d_vertices.end(), vertex_t{0});

    std::tie(graph, std::ignore, std::ignore, std::ignore) =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, false, false>(
        handle,
        std::move(d_vertices),
        std::move(d_srcs),
        std::move(d_dst),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{false, true},
        false);

    auto graph_view = graph.view();

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

    // warm up
    bool direction_optimizing = false;

    vertex_t source = h_sources[0];
    rmm::device_scalar<vertex_t> const d_source_0(source, handle.get_stream());
    cugraph::bfs(handle,
                 graph_view,
                 d_distances_ref[0].begin(),
                 d_predecessors_ref[0].begin(),
                 d_source_0.data(),
                 size_t{1},
                 direction_optimizing,
                 configuration.radius);

    // one by one
    HighResTimer hr_timer;
    hr_timer.start("bfs");
    cudaProfilerStart();
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
                   configuration.radius);
    }
    cudaProfilerStop();
    hr_timer.stop();

    // ms
    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    hr_timer.start("msbfs");
    cudaProfilerStart();
    cugraph::bfs(handle,
                 graph_view,
                 d_distances.begin(),
                 d_predecessors.begin(),
                 d_sources.data(),
                 h_sources.size(),
                 direction_optimizing,
                 configuration.radius);

    cudaProfilerStop();
    hr_timer.stop();
    hr_timer.display_and_clear(std::cout);

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
    ASSERT_TRUE(ref_sum > 0);
    ASSERT_TRUE(ref_sum < std::numeric_limits<vertex_t>::max());
    ASSERT_TRUE(ref_sum == ms_sum);
  }
};

TEST_P(Tests_MsBfs, CheckInt32Int32) { run_current_test<int32_t, int32_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MsBfs,
                         ::testing::Values(MsBfs_Usecase{8, 10, 16, 32, 2},
                                           MsBfs_Usecase{512, 10, 16, 32, 3},
                                           MsBfs_Usecase{512, 10, 16, 32, 100}));

CUGRAPH_TEST_PROGRAM_MAIN()
