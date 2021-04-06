/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <algorithms.hpp>
#include <experimental/graph.hpp>
#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 1;

// Dijkstra's algorithm
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_reference(edge_t const* offsets,
                    vertex_t const* indices,
                    weight_t const* weights,
                    weight_t* distances,
                    vertex_t* predecessors,
                    vertex_t num_vertices,
                    vertex_t source,
                    weight_t cutoff = std::numeric_limits<weight_t>::max())
{
  using queue_iterm_t = std::tuple<weight_t, vertex_t>;

  std::fill(distances, distances + num_vertices, std::numeric_limits<weight_t>::max());
  std::fill(predecessors, predecessors + num_vertices, cugraph::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = weight_t{0.0};
  std::priority_queue<queue_iterm_t, std::vector<queue_iterm_t>, std::greater<queue_iterm_t>>
    queue{};
  queue.push(std::make_tuple(weight_t{0.0}, source));

  while (queue.size() > 0) {
    weight_t distance{};
    vertex_t row{};
    std::tie(distance, row) = queue.top();
    queue.pop();
    if (distance > *(distances + row)) { continue; }
    auto nbr_offsets     = *(offsets + row);
    auto nbr_offset_last = *(offsets + row + 1);
    for (auto nbr_offset = nbr_offsets; nbr_offset != nbr_offset_last; ++nbr_offset) {
      auto nbr          = *(indices + nbr_offset);
      auto new_distance = distance + *(weights + nbr_offset);
      auto threshold    = std::min(*(distances + nbr), cutoff);
      if (new_distance < threshold) {
        *(distances + nbr)    = new_distance;
        *(predecessors + nbr) = row;
        queue.push(std::make_tuple(new_distance, nbr));
      }
    }
  }

  return;
}

typedef struct SSSP_Usecase_t {
  cugraph::test::input_graph_specifier_t input_graph_specifier{};

  size_t source{0};
  bool check_correctness{false};

  SSSP_Usecase_t(std::string const& graph_file_path, size_t source, bool check_correctness = true)
    : source(source), check_correctness(check_correctness)
  {
    std::string graph_file_full_path{};
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
    input_graph_specifier.tag = cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH;
    input_graph_specifier.graph_file_full_path = graph_file_full_path;
  };

  SSSP_Usecase_t(cugraph::test::rmat_params_t rmat_params,
                 size_t source,
                 bool check_correctness = true)
    : source(source), check_correctness(check_correctness)
  {
    input_graph_specifier.tag         = cugraph::test::input_graph_specifier_t::RMAT_PARAMS;
    input_graph_specifier.rmat_params = rmat_params;
  }
} SSSP_Usecase;

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false>,
           rmm::device_uvector<vertex_t>>
read_graph(raft::handle_t const& handle, SSSP_Usecase const& configuration, bool renumber)
{
  return configuration.input_graph_specifier.tag ==
             cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH
           ? cugraph::test::
               read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
                 handle, configuration.input_graph_specifier.graph_file_full_path, true, renumber)
           : cugraph::test::
               generate_graph_from_rmat_params<vertex_t, edge_t, weight_t, false, false>(
                 handle,
                 configuration.input_graph_specifier.rmat_params.scale,
                 configuration.input_graph_specifier.rmat_params.edge_factor,
                 configuration.input_graph_specifier.rmat_params.a,
                 configuration.input_graph_specifier.rmat_params.b,
                 configuration.input_graph_specifier.rmat_params.c,
                 configuration.input_graph_specifier.rmat_params.seed,
                 configuration.input_graph_specifier.rmat_params.undirected,
                 configuration.input_graph_specifier.rmat_params.scramble_vertex_ids,
                 true,
                 renumber,
                 std::vector<size_t>{0},
                 size_t{1});
}

class Tests_SSSP : public ::testing::TestWithParam<SSSP_Usecase> {
 public:
  Tests_SSSP() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(SSSP_Usecase const& configuration)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());
    std::tie(graph, d_renumber_map_labels) =
      read_graph<vertex_t, edge_t, weight_t>(handle, configuration, renumber);
    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "read_graph took " << elapsed_time * 1e-6 << " s.\n";
    }
    auto graph_view = graph.view();

    ASSERT_TRUE(static_cast<vertex_t>(configuration.source) >= 0 &&
                static_cast<vertex_t>(configuration.source) < graph_view.get_number_of_vertices());

    rmm::device_uvector<weight_t> d_distances(graph_view.get_number_of_vertices(),
                                              handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.get_number_of_vertices(),
                                                 handle.get_stream());

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    cugraph::experimental::sssp(handle,
                                graph_view,
                                d_distances.data(),
                                d_predecessors.data(),
                                static_cast<vertex_t>(configuration.source),
                                std::numeric_limits<weight_t>::max(),
                                false);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "SSSP took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (configuration.check_correctness) {
      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> unrenumbered_graph(
        handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore) =
          read_graph<vertex_t, edge_t, weight_t>(handle, configuration, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets(unrenumbered_graph_view.get_number_of_vertices() + 1);
      std::vector<vertex_t> h_indices(unrenumbered_graph_view.get_number_of_edges());
      std::vector<weight_t> h_weights(unrenumbered_graph_view.get_number_of_edges());
      raft::update_host(h_offsets.data(),
                        unrenumbered_graph_view.offsets(),
                        unrenumbered_graph_view.get_number_of_vertices() + 1,
                        handle.get_stream());
      raft::update_host(h_indices.data(),
                        unrenumbered_graph_view.indices(),
                        unrenumbered_graph_view.get_number_of_edges(),
                        handle.get_stream());
      raft::update_host(h_weights.data(),
                        unrenumbered_graph_view.weights(),
                        unrenumbered_graph_view.get_number_of_edges(),
                        handle.get_stream());

      handle.get_stream_view().synchronize();

      auto unrenumbered_source = static_cast<vertex_t>(configuration.source);
      if (renumber) {
        std::vector<vertex_t> h_renumber_map_labels(d_renumber_map_labels.size());
        raft::update_host(h_renumber_map_labels.data(),
                          d_renumber_map_labels.data(),
                          d_renumber_map_labels.size(),
                          handle.get_stream());

        handle.get_stream_view().synchronize();

        unrenumbered_source = h_renumber_map_labels[configuration.source];
      }

      std::vector<weight_t> h_reference_distances(unrenumbered_graph_view.get_number_of_vertices());
      std::vector<vertex_t> h_reference_predecessors(
        unrenumbered_graph_view.get_number_of_vertices());

      sssp_reference(h_offsets.data(),
                     h_indices.data(),
                     h_weights.data(),
                     h_reference_distances.data(),
                     h_reference_predecessors.data(),
                     unrenumbered_graph_view.get_number_of_vertices(),
                     unrenumbered_source,
                     std::numeric_limits<weight_t>::max());

      std::vector<weight_t> h_cugraph_distances(graph_view.get_number_of_vertices());
      std::vector<vertex_t> h_cugraph_predecessors(graph_view.get_number_of_vertices());
      if (renumber) {
        cugraph::experimental::unrenumber_local_int_vertices(handle,
                                                             d_predecessors.data(),
                                                             d_predecessors.size(),
                                                             d_renumber_map_labels.data(),
                                                             vertex_t{0},
                                                             graph_view.get_number_of_vertices(),
                                                             true);

        auto d_unrenumbered_distances = cugraph::test::sort_by_key(
          handle, d_renumber_map_labels.data(), d_distances.data(), d_renumber_map_labels.size());
        auto d_unrenumbered_predecessors = cugraph::test::sort_by_key(handle,
                                                                      d_renumber_map_labels.data(),
                                                                      d_predecessors.data(),
                                                                      d_renumber_map_labels.size());

        raft::update_host(h_cugraph_distances.data(),
                          d_unrenumbered_distances.data(),
                          d_unrenumbered_distances.size(),
                          handle.get_stream());
        raft::update_host(h_cugraph_predecessors.data(),
                          d_unrenumbered_predecessors.data(),
                          d_unrenumbered_predecessors.size(),
                          handle.get_stream());

        handle.get_stream_view().synchronize();
      } else {
        raft::update_host(
          h_cugraph_distances.data(), d_distances.data(), d_distances.size(), handle.get_stream());
        raft::update_host(h_cugraph_predecessors.data(),
                          d_predecessors.data(),
                          d_predecessors.size(),
                          handle.get_stream());

        handle.get_stream_view().synchronize();
      }

      auto max_weight_element = std::max_element(h_weights.begin(), h_weights.end());
      auto epsilon            = *max_weight_element * weight_t{1e-6};
      auto nearly_equal = [epsilon](auto lhs, auto rhs) { return std::fabs(lhs - rhs) < epsilon; };

      ASSERT_TRUE(std::equal(h_reference_distances.begin(),
                             h_reference_distances.end(),
                             h_cugraph_distances.begin(),
                             nearly_equal))
        << "distances do not match with the reference values.";

      for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
        auto i = std::distance(h_cugraph_predecessors.begin(), it);
        if (*it == cugraph::invalid_vertex_id<vertex_t>::value) {
          ASSERT_TRUE(h_reference_predecessors[i] == *it)
            << "vertex reachability do not match with the reference.";
        } else {
          auto pred_distance = h_reference_distances[*it];
          bool found{false};
          for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
            if (h_indices[j] == i) {
              if (nearly_equal(pred_distance + h_weights[j], h_reference_distances[i])) {
                found = true;
                break;
              }
            }
          }
          ASSERT_TRUE(found)
            << "no edge from the predecessor vertex to this vertex with the matching weight.";
        }
      }
    }
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_SSSP, CheckInt32Int32Float) { run_current_test<int32_t, int32_t, float>(GetParam()); }

INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_SSSP,
  ::testing::Values(
    // enable correctness checks
    SSSP_Usecase("test/datasets/karate.mtx", 0),
    SSSP_Usecase("test/datasets/dblp.mtx", 0),
    SSSP_Usecase("test/datasets/wiki2003.mtx", 1000),
    SSSP_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false}, 0),
    // disable correctness checks for large graphs
    SSSP_Usecase(cugraph::test::rmat_params_t{20, 32, 0.57, 0.19, 0.19, 0, false, false},
                 0,
                 false)));

CUGRAPH_TEST_PROGRAM_MAIN()
