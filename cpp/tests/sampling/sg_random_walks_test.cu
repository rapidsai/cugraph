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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <gtest/gtest.h>

struct UniformRandomWalks_Usecase {
  bool test_weighted{false};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t num_paths)
  {
    return cugraph::uniform_random_walks(handle, graph_view, start_vertices, num_paths, seed);
  }
};

struct BiasedRandomWalks_Usecase {
  bool test_weighted{true};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t num_paths)
  {
    return cugraph::biased_random_walks(handle, graph_view, start_vertices, num_paths, seed);
  }
};

struct Node2VecRandomWalks_Usecase {
  double p{1};
  double q{1};
  bool test_weighted{false};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t num_paths)
  {
    return cugraph::node2vec_random_walks(
      handle, graph_view, start_vertices, num_paths, p, q, seed);
  }
};

template <typename tuple_t>
class Tests_RandomWalks : public ::testing::TestWithParam<tuple_t> {
 public:
  Tests_RandomWalks() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(tuple_t const& param)
  {
    raft::handle_t handle{};
    HighResClock hr_clock{};

    auto [randomwalks_usecase, input_usecase] = param;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    // TODO: Do I need to turn renumber off?  It's off in the old test
    bool renumber{true};
    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, randomwalks_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto graph_view = graph.view();

    // FIXME: The sampling functions should use the standard version, not number_of_vertices
    vertex_t invalid_vertex_id = graph_view.number_of_vertices();

    std::cout << "invalid_vertex_id = " << invalid_vertex_id << std::endl;

    edge_t num_paths = 10;
    rmm::device_uvector<vertex_t> d_start(num_paths, handle.get_stream());

    thrust::tabulate(handle.get_thrust_policy(),
                     d_start.begin(),
                     d_start.end(),
                     [num_vertices = graph_view.number_of_vertices()] __device__(auto idx) {
                       return (idx % num_vertices);
                     });

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

#if 1
    auto [d_vertices, d_weights] =
      randomwalks_usecase(handle,
                          graph_view,
                          raft::device_span<vertex_t const>{d_start.data(), d_start.size()},
                          num_paths);

    std::cout << "num_paths = " << num_paths << std::endl;
    raft::print_device_vector("d_start", d_start.data(), d_start.size(), std::cout);
#else
    EXPECT_THROW(
      randomwalks_usecase(handle,
                          graph_view,
                          raft::device_span<vertex_t const>{d_start.data(), d_start.size()},
                          num_paths),
      std::exception);
#endif

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "RandomWalks took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (randomwalks_usecase.check_correctness) {
      // FIXME: This maybe should be a general purpose test utility - to convert
      //    a graph_view to an std::map for validation.
      // Actually... very slow on CPU for large graphs.
      // Perhaps I can decompress, sort and search for edges in the sorted list?
      std::map<std::pair<vertex_t, vertex_t>, weight_t> h_edge_map;

      auto [d_src, d_dst, d_wgt] = graph_view.decompress_to_edgelist(handle, std::nullopt);

      raft::print_device_vector("d_vertices", d_vertices.data(), d_vertices.size(), std::cout);
      if (d_weights)
        raft::print_device_vector("d_weights", d_weights->data(), d_weights->size(), std::cout);

      if (d_weights)
        std::cout << "d_vertices size = " << d_vertices.size() << ", d_weights size = " << d_weights->size() << std::endl;

      std::vector<vertex_t> h_src(d_src.size());
      std::vector<vertex_t> h_dst(d_dst.size());
      std::vector<weight_t> h_wgt(d_src.size());

      raft::update_host(h_src.data(), d_src.data(), d_src.size(), handle.get_stream());
      raft::update_host(h_dst.data(), d_dst.data(), d_dst.size(), handle.get_stream());
      if (d_wgt)
        raft::update_host(h_wgt.data(), d_wgt->data(), d_wgt->size(), handle.get_stream());
      else
        std::fill(h_wgt.begin(), h_wgt.end(), weight_t{1});

      thrust::for_each(thrust::host,
                       thrust::make_zip_iterator(h_src.begin(), h_dst.begin(), h_wgt.begin()),
                       thrust::make_zip_iterator(h_src.end(), h_dst.end(), h_wgt.end()),
                       [&h_edge_map] __host__(auto tuple) {
                         auto key   = std::make_pair(thrust::get<0>(tuple), thrust::get<1>(tuple));
                         auto value = thrust::get<2>(tuple);
                         h_edge_map[key] = value;
#if 0
                         std::cout << "(" << thrust::get<0>(tuple) << ", " << thrust::get<1>(tuple)
                                   << ") = " << thrust::get<2>(tuple) << std::endl;
#endif
                       });

      std::vector<vertex_t> h_vertices(d_vertices.size());
      std::vector<weight_t> h_weights(num_paths * d_start.size());

#if 1
      raft::update_host(
        h_vertices.data(), d_vertices.data(), d_vertices.size(), handle.get_stream());

      if (d_weights)
        raft::update_host(
          h_weights.data(), d_weights->data(), d_weights->size(), handle.get_stream());
      else
        std::fill(h_weights.begin(), h_weights.end(), weight_t{1});
#endif

      thrust::for_each(
        thrust::host,
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(d_start.size()),
        [&h_edge_map, &h_vertices, &h_weights, num_paths, invalid_vertex_id] __host__(auto i) {
          for (int j = 0; j < num_paths; ++j) {
            vertex_t src = h_vertices[i * (num_paths + 1) + j];
            vertex_t dst = h_vertices[i * (num_paths + 1) + j + 1];

            // FIXME: if src != invalid_vertex_id and dst == invalid_vertex_id
            //    should add a check to verify that degree(src) == 0

            if (dst != invalid_vertex_id) {
              auto pos = h_edge_map.find(std::make_pair(h_vertices[i * (num_paths + 1) + j],
                                                        h_vertices[i * (num_paths + 1) + j + 1]));
              ASSERT_NE(pos, h_edge_map.end())
                << "edge (" << h_vertices[i * (num_paths + 1) + j] << ", "
                << h_vertices[i * (num_paths + 1) + j + 1] << ") not found";
              ASSERT_EQ(pos->second, h_weights[i * num_paths + j])
                << "edge (" << h_vertices[i * (num_paths + 1) + j] << ", "
                << h_vertices[i * (num_paths + 1) + j + 1] << ") weight = " << pos->second
                << ", expected = " << h_weights[i * num_paths + j];
            }
          }
        });
    }
  }
};

using Tests_UniformRandomWalks_File =
  Tests_RandomWalks<std::tuple<UniformRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_UniformRandomWalks_Rmat =
  Tests_RandomWalks<std::tuple<UniformRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;
using Tests_BiasedRandomWalks_File =
  Tests_RandomWalks<std::tuple<BiasedRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_BiasedRandomWalks_Rmat =
  Tests_RandomWalks<std::tuple<BiasedRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;
using Tests_Node2VecRandomWalks_File =
  Tests_RandomWalks<std::tuple<Node2VecRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_Node2VecRandomWalks_Rmat =
  Tests_RandomWalks<std::tuple<Node2VecRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;

TEST_P(Tests_UniformRandomWalks_File, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_UniformRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(UniformRandomWalks_Usecase{false, 0, true},
                      UniformRandomWalks_Usecase{true, 0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_BiasedRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(BiasedRandomWalks_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_Node2VecRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(Node2VecRandomWalks_Usecase{4, 8}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

CUGRAPH_TEST_PROGRAM_MAIN()
