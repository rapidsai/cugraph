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

#include <utilities/base_fixture.hpp>  // cugraph::test::create_memory_resource()
#include <utilities/high_res_timer.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <sampling/random_walks.cuh>

#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda_profiler_api.h>
#include <thrust/random.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

/**
 * @internal
 * @brief Populates the device vector d_start with the starting vertex indices
 * to be used for each RW path specified.
 */
template <typename vertex_t, typename index_t>
void fill_start(raft::handle_t const& handle,
                rmm::device_uvector<vertex_t>& d_start,
                index_t num_vertices)
{
  index_t num_paths = d_start.size();

  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<index_t>(0),
                    thrust::make_counting_iterator<index_t>(num_paths),

                    d_start.begin(),
                    [num_vertices] __device__(auto indx) { return indx % num_vertices; });
}

namespace impl_details = cugraph::detail;

enum class traversal_id_t : int { HORIZONTAL = 0, VERTICAL };

/**
 * @internal
 * @brief Calls the random_walks algorithm with specified traversal strategy and displays the time
 * metrics (total time for all requested paths, average time for each path).
 */
template <typename graph_vt>
void output_random_walks_time(graph_vt const& graph_view,
                              typename graph_vt::edge_type num_paths,
                              traversal_id_t trv_id,
                              int sampling_id)
{
  using vertex_t = typename graph_vt::vertex_type;
  using edge_t   = typename graph_vt::edge_type;
  using weight_t = typename graph_vt::weight_type;
  using real_t   = float;

  raft::handle_t handle{};
  rmm::device_uvector<vertex_t> d_start(num_paths, handle.get_stream());

  vertex_t num_vertices = graph_view.number_of_vertices();
  fill_start(handle, d_start, num_vertices);

  // 0-copy const device view:
  //
  impl_details::device_const_vector_view<vertex_t, edge_t> d_start_view{d_start.data(), num_paths};

  edge_t max_depth{10};

  weight_t p{4};
  weight_t q{8};

  HighResTimer hr_timer;
  std::string label{};

  if (trv_id == traversal_id_t::HORIZONTAL) {
    if (sampling_id == 0) {
      label = std::string("RandomWalks; Horizontal traversal; uniform sampling - ");
      impl_details::uniform_selector_t<graph_vt, real_t> selector{handle, graph_view, real_t{0}};

      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::horizontal_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    } else if (sampling_id == 1) {
      label = std::string("RandomWalks; Horizontal traversal; biased sampling - ");
      impl_details::biased_selector_t<graph_vt, real_t> selector{handle, graph_view, real_t{0}};

      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::horizontal_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    } else if (sampling_id == 2) {
      label =
        std::string("RandomWalks; Horizontal traversal; node2vec sampling with alpha cache - ");
      impl_details::node2vec_selector_t<graph_vt, real_t> selector{
        handle, graph_view, real_t{0}, p, q, num_paths};

      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::horizontal_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    } else {
      label =
        std::string("RandomWalks; Horizontal traversal; node2vec sampling without alpha cache - ");
      impl_details::node2vec_selector_t<graph_vt, real_t> selector{
        handle, graph_view, real_t{0}, p, q};

      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::horizontal_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    }
  } else {
    if (sampling_id == 0) {
      label = std::string("RandomWalks; Vertical traversal; uniform sampling - ");
      impl_details::uniform_selector_t<graph_vt, real_t> selector{handle, graph_view, real_t{0}};
      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::vertical_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    } else if (sampling_id == 1) {
      label = std::string("RandomWalks; Vertical traversal; biased sampling - ");
      impl_details::biased_selector_t<graph_vt, real_t> selector{handle, graph_view, real_t{0}};
      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::vertical_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    } else if (sampling_id == 2) {
      label = std::string("RandomWalks; Vertical traversal; node2vec sampling with alpha cache - ");
      impl_details::node2vec_selector_t<graph_vt, real_t> selector{
        handle, graph_view, real_t{0}, p, q, num_paths};

      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::vertical_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    } else {
      label =
        std::string("RandomWalks; Vertical traversal; node2vec sampling without alpha cache - ");
      impl_details::node2vec_selector_t<graph_vt, real_t> selector{
        handle, graph_view, real_t{0}, p, q};

      hr_timer.start(label);
      cudaProfilerStart();

      auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                       decltype(selector),
                                                       impl_details::vertical_traversal_t>(
        handle,  // prevent clang-format to separate function name from its namespace
        graph_view,
        d_start_view,
        max_depth,
        selector);

      cudaProfilerStop();
      hr_timer.stop();
    }
  }
  try {
    auto runtime = hr_timer.get_average_runtime(label);

    std::cout << "RW for num_paths: " << num_paths
              << ", runtime [ms] / path: " << runtime / num_paths << ":\n";

  } catch (std::exception const& ex) {
    std::cerr << ex.what() << '\n';
    return;

  } catch (...) {
    std::cerr << "ERROR: Unknown exception on timer label search." << '\n';
    return;
  }
  hr_timer.display(std::cout);
}

/**
 * @struct RandomWalks_Usecase
 * @brief Used to specify input to a random_walks benchmark/profile run
 *
 * @var RandomWalks_Usecase::graph_file_full_path  Computed during construction
 * to be an absolute path consisting of the value of the RAPIDS_DATASET_ROOT_DIR
 * env var and the graph_file_path constructor arg. This is initialized to an
 * empty string.
 *
 * @var RandomWalks_Usecase::test_weighted Bool representing if the specified
 * graph is weighted or not. This is initialized to false (unweighted).
 */
struct RandomWalks_Usecase {
  std::string graph_file_full_path{};
  bool test_weighted{false};

  RandomWalks_Usecase(std::string const& graph_file_path, bool test_weighted)
    : test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
};

/**
 * @brief Runs random_walks on a specified input and outputs time metrics
 *
 * Creates a graph_t instance from the configuration specified in the
 * RandomWalks_Usecase instance passed in (currently by reading a dataset to
 * populate the graph_t), then runs random_walks to generate 1, 10, and 100
 * random paths and output statistics for each.
 *
 * @tparam vertex_t          Type of vertex identifiers.
 * @tparam edge_t            Type of edge identifiers.
 * @tparam weight_t          Type of weight identifiers.
 *
 * @param[in] configuration RandomWalks_Usecase instance containing the input
 * file to read for constructing the graph_t.
 * @param[in] trv_id traversal strategy.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void run(RandomWalks_Usecase const& configuration, traversal_id_t trv_id, int sampling_id)
{
  raft::handle_t handle{};

  cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
  std::tie(graph, std::ignore) =
    cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
      handle, configuration.graph_file_full_path, configuration.test_weighted, false);

  auto graph_view = graph.view();

  // FIXME: the num_paths vector might be better specified via the
  // configuration input instead of hardcoding here.
  std::vector<edge_t> v_np{1, 10, 100};
  for (auto&& num_paths : v_np) {
    output_random_walks_time(graph_view, num_paths, trv_id, sampling_id);
  }
}

/**
 * @brief Performs the random_walks benchmark/profiling run
 *
 * main function for performing the random_walks benchmark/profiling run. The
 * resulting executable takes the following options: "rmm_mode" which can be one
 * of "binning", "cuda", "pool", or "managed.  "dataset" which is a path
 * relative to the env var RAPIDS_DATASET_ROOT_DIR to a input .mtx file to use
 * to populate the graph_t instance.
 *
 * To use the default values of rmm_mode=pool and
 * dataset=test/datasets/karate.mtx:
 * @code
 *   RANDOM_WALKS_PROFILING
 * @endcode
 *
 * To specify managed memory and the netscience.mtx dataset (relative to a
 * particular RAPIDS_DATASET_ROOT_DIR setting):
 * @code
 *   RANDOM_WALKS_PROFILING --rmm_mode=managed --dataset=test/datasets/netscience.mtx
 * @endcode
 *
 * @return An int representing a successful run. 0 indicates success.
 */
int main(int argc, char** argv)
{
  // Add command-line processing, provide defaults
  cxxopts::Options options(argv[0], " - Random Walks benchmark command line options");
  options.add_options()(
    "rmm_mode", "RMM allocation mode", cxxopts::value<std::string>()->default_value("pool"));
  options.add_options()(
    "dataset", "dataset", cxxopts::value<std::string>()->default_value("test/datasets/karate.mtx"));
  auto const cmd_options = options.parse(argc, argv);
  auto const rmm_mode    = cmd_options["rmm_mode"].as<std::string>();
  auto const dataset     = cmd_options["dataset"].as<std::string>();

  // Configure RMM
  auto resource = cugraph::test::create_memory_resource(rmm_mode);
  rmm::mr::set_current_device_resource(resource.get());

  // Run benchmarks
  std::cout << "Using dataset: " << dataset << std::endl;

  std::cout << "##### Horizontal traversal strategy:\n";

  std::cout << "### Uniform sampling strategy:\n";
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::HORIZONTAL, 0);

  std::cout << "### Biased sampling strategy:\n";
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::HORIZONTAL, 1);

  std::cout << "### Node2Vec sampling strategy:\n";
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::HORIZONTAL, 2);
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::HORIZONTAL, 3);

  std::cout << "##### Vertical traversal strategy:\n";

  std::cout << "### Uniform sampling strategy:\n";
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::VERTICAL, 0);

  std::cout << "### Biased sampling strategy:\n";
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::VERTICAL, 1);

  std::cout << "### Node2Vec sampling strategy:\n";
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::VERTICAL, 2);
  run<int32_t, int32_t, float>(RandomWalks_Usecase(dataset, true), traversal_id_t::VERTICAL, 3);

  // FIXME: consider returning non-zero for situations that warrant it (eg. if
  // the algo ran but the results are invalid, if a benchmark threshold is
  // exceeded, etc.)
  return 0;
}
