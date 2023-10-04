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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/mtmg/edgelist.hpp>
#include <cugraph/mtmg/graph.hpp>
#include <cugraph/mtmg/per_thread_edgelist.hpp>
#include <cugraph/mtmg/renumber_map.hpp>
#include <cugraph/mtmg/resource_manager.hpp>
#include <cugraph/mtmg/vertex_result.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <nccl.h>

#include <filesystem>
#include <fstream>
#include <vector>

#include <thrust/count.h>
#include <thrust/unique.h>

struct Multithreaded_Usecase {
  bool test_weighted{false};
  bool check_correctness{true};
};

std::string g_comms_dir_name{};
int g_node_rank{};
int g_num_nodes{};
int g_execution_id{0};

template <typename input_usecase_t>
class Tests_Multithreaded
  : public ::testing::TestWithParam<std::tuple<Multithreaded_Usecase, input_usecase_t>> {
 public:
  Tests_Multithreaded() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  std::vector<int> get_gpu_list()
  {
    int num_gpus_per_node{1};
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));

    std::vector<int> gpu_list(num_gpus_per_node);
    std::iota(gpu_list.begin(), gpu_list.end(), 0);

    return gpu_list;
  }

  void wait_for_directory(std::string directory_name, int max_tries = 60)
  {
    while (max_tries > 0) {
      if (std::filesystem::is_directory(directory_name)) break;
      sleep(1);
      --max_tries;
    }

    CUGRAPH_EXPECTS(std::filesystem::is_directory(directory_name),
                    "Timed out waiting for directory to be created");
  }

  std::ifstream wait_for_file(std::string file_name, int max_tries = 60)
  {
    while (max_tries > 0) {
      if (std::filesystem::is_regular_file(file_name)) break;
      sleep(1);
      --max_tries;
    }

    CUGRAPH_EXPECTS(std::filesystem::is_regular_file(file_name),
                    "Timed out waiting for file to be created");

    return std::ifstream(file_name, std::ios::binary);
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename result_t,
            bool multi_gpu>
  void run_current_test(
    std::tuple<Multithreaded_Usecase const&, input_usecase_t const&> const& param,
    std::vector<int> gpu_list)
  {
    using edge_type_t = int32_t;

    constexpr bool renumber           = true;
    constexpr bool do_expensive_check = false;

    auto [multithreaded_usecase, input_usecase] = param;

    raft::handle_t handle{};

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    size_t device_buffer_size{64 * 1024 * 1024};
    size_t thread_buffer_size{4 * 1024 * 1024};

    int num_gpus    = gpu_list.size();
    int num_threads = num_gpus * 4;

    //
    //  This is intended to mimic a multi-node host application (non-MPI) integrating
    //  with MTMG library.  This is a simple implementation using a shared file system
    //  to pass configuration messages.
    //
    ncclUniqueId instance_manager_id;
    int execution_id = g_execution_id++;

    std::ostringstream comms_dir_name;
    comms_dir_name << g_comms_dir_name << "_" << execution_id;

    if (g_node_rank == 0) {
      ncclGetUniqueId(&instance_manager_id);

      // Create directory for configuration files
      std::filesystem::create_directory(comms_dir_name.str());
      std::ofstream instance_manager_file(comms_dir_name.str() + "/instance_manager",
                                          std::ios::binary);
      instance_manager_file.write(reinterpret_cast<char const*>(&instance_manager_id),
                                  sizeof(instance_manager_id));
      instance_manager_file.close();
    } else {
      // Wait for node rank 0 to create directory
      wait_for_directory(comms_dir_name.str());

      auto instance_manager_file = wait_for_file(comms_dir_name.str() + "/instance_manager");
      instance_manager_file.read(reinterpret_cast<char*>(&instance_manager_id),
                                 sizeof(instance_manager_id));
      instance_manager_file.close();
    }

    // Create a file for this process (rank) to identify how many GPUs
    std::ostringstream filename_creator;

    filename_creator << comms_dir_name.str() << "/gpu_count_" << g_node_rank;
    {
      std::ofstream num_gpus_file(filename_creator.str(), std::ios::binary);
      int num_gpus_on_this_node = static_cast<int>(gpu_list.size());
      num_gpus_file.write(reinterpret_cast<char const*>(&num_gpus_on_this_node), sizeof(int));
      num_gpus_file.close();
    }

    cugraph::mtmg::resource_manager_t resource_manager;
    int node_rank{0};

    for (int i = 0; i < g_num_nodes; ++i) {
      if (i != g_node_rank) {
        filename_creator.str("");
        filename_creator << comms_dir_name.str() << "/gpu_count_" << i;
        auto num_gpus_file = wait_for_file(filename_creator.str());
        int num_gpus_this_node{0};
        num_gpus_file.read(reinterpret_cast<char*>(&num_gpus_this_node), sizeof(int));
        num_gpus_file.close();

        for (int j = 0; j < num_gpus_this_node; ++j) {
          resource_manager.register_remote_gpu(node_rank++, i);
        }
      } else {
        std::for_each(
          gpu_list.begin(), gpu_list.end(), [&resource_manager, &node_rank](int gpu_id) {
            resource_manager.register_local_gpu(node_rank++, rmm::cuda_device_id{gpu_id});
          });
      }
    }

    std::cout << "creating instance manager" << std::endl;
    raft::print_host_vector("  registered ranks",
                            resource_manager.registered_ranks().data(),
                            resource_manager.registered_ranks().size(),
                            std::cout);

    auto instance_manager = resource_manager.create_instance_manager(
      resource_manager.registered_ranks(), instance_manager_id);

    std::cout << "done creating instance manager" << std::endl;

    cugraph::mtmg::edgelist_t<vertex_t, weight_t, edge_t, edge_type_t> edgelist;
    cugraph::mtmg::graph_t<vertex_t, edge_t, true, multi_gpu> graph;
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu> graph_view;
    cugraph::mtmg::vertex_result_t<result_t> pageranks;
    std::optional<cugraph::mtmg::renumber_map_t<vertex_t>> renumber_map =
      std::make_optional<cugraph::mtmg::renumber_map_t<vertex_t>>();

    auto edge_weights = multithreaded_usecase.test_weighted
                          ? std::make_optional<cugraph::mtmg::edge_property_t<
                              cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                              weight_t>>()
                          : std::nullopt;

    std::cout << "create edge list" << std::endl;

    //
    // Simulate graph creation by spawning threads to walk through the
    // local COO and add edges
    //
    std::vector<std::thread> running_threads;

    //  Initialize shared edgelist object, one per GPU
    for (int i = 0; i < num_gpus; ++i) {
      running_threads.emplace_back([&instance_manager,
                                    &edgelist,
                                    device_buffer_size,
                                    use_weight    = true,
                                    use_edge_id   = false,
                                    use_edge_type = false]() {
        auto thread_handle = instance_manager->get_handle();

        edgelist.set(thread_handle, device_buffer_size, use_weight, use_edge_id, use_edge_type);
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    std::cout << "load edges" << std::endl;

    // Load SG edge list
    auto [d_src_v, d_dst_v, d_weights_v, d_vertices_v, is_symmetric] =
      input_usecase.template construct_edgelist<vertex_t, weight_t>(
        handle, multithreaded_usecase.test_weighted, false, false);

    auto h_src_v         = cugraph::test::to_host(handle, d_src_v);
    auto h_dst_v         = cugraph::test::to_host(handle, d_dst_v);
    auto h_weights_v     = cugraph::test::to_host(handle, d_weights_v);
    auto unique_vertices = cugraph::test::to_host(handle, d_vertices_v);

    // Load edgelist from different threads.  We'll use more threads than GPUs here
    for (int i = 0; i < num_threads; ++i) {
      running_threads.emplace_back([&instance_manager,
                                    thread_buffer_size,
                                    &edgelist,
                                    &h_src_v,
                                    &h_dst_v,
                                    &h_weights_v,
                                    i,
                                    num_threads]() {
        auto thread_handle = instance_manager->get_handle();
        cugraph::mtmg::per_thread_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t>
          per_thread_edgelist(edgelist.get(thread_handle), thread_buffer_size);

        for (size_t j = i; j < h_src_v.size(); j += num_threads) {
#if 0
          if (h_weights_v) {
            thread_edgelist.append(
              thread_handle, h_src_v[j], h_dst_v[j], (*h_weights_v)[j], std::nullopt, std::nullopt);
          } else {
            thread_edgelist.append(
              thread_handle, h_src_v[j], h_dst_v[j], std::nullopt, std::nullopt, std::nullopt);
          }
#endif
          per_thread_edgelist.append(
            thread_handle,
            h_src_v[j],
            h_dst_v[j],
            h_weights_v ? std::make_optional((*h_weights_v)[j]) : std::nullopt,
            std::nullopt,
            std::nullopt);
        }

        per_thread_edgelist.flush(thread_handle);
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    std::cout << "create graph" << std::endl;

    for (int i = 0; i < num_gpus; ++i) {
      running_threads.emplace_back([&instance_manager,
                                    &graph,
                                    &edge_weights,
                                    &edgelist,
                                    &renumber_map,
                                    &pageranks,
                                    is_symmetric = is_symmetric,
                                    renumber,
                                    do_expensive_check]() {
        auto thread_handle = instance_manager->get_handle();

        if (thread_handle.get_thread_rank() > 0) return;

        std::optional<cugraph::mtmg::edge_property_t<
          cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
          edge_t>>
          edge_ids{std::nullopt};
        std::optional<cugraph::mtmg::edge_property_t<
          cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
          int32_t>>
          edge_types{std::nullopt};

        std::cout << "calling finalize_buffer, rank = " << thread_handle.get_rank() << std::endl;

        edgelist.finalize_buffer(thread_handle);

        std::cout << "calling consolidate_and_shuffle, rank = " << thread_handle.get_rank()
                  << std::endl;
        edgelist.consolidate_and_shuffle(thread_handle, true);

        std::cout << "calling create_graph_from_edgelist, rank = " << thread_handle.get_rank()
                  << std::endl;

        cugraph::mtmg::
          create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, true, multi_gpu>(
            thread_handle,
            edgelist,
            cugraph::graph_properties_t{is_symmetric, true},
            renumber,
            graph,
            edge_weights,
            edge_ids,
            edge_types,
            renumber_map,
            do_expensive_check);
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    std::cout << "call pagerank" << std::endl;

    graph_view = graph.view();

    for (int i = 0; i < num_threads; ++i) {
      running_threads.emplace_back(
        [&instance_manager, &graph_view, &edge_weights, &pageranks, alpha, epsilon]() {
          auto thread_handle = instance_manager->get_handle();

          if (thread_handle.get_thread_rank() > 0) return;

          auto [local_pageranks, metadata] =
            cugraph::pagerank<vertex_t, edge_t, weight_t, weight_t, true>(
              thread_handle.raft_handle(),
              graph_view.get(thread_handle),
              edge_weights ? std::make_optional(edge_weights->get(thread_handle).view())
                           : std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              alpha,
              epsilon,
              500,
              true);

          pageranks.set(thread_handle, std::move(local_pageranks));
        });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    std::vector<std::tuple<std::vector<vertex_t>, std::vector<result_t>>> computed_pageranks_v;
    std::mutex computed_pageranks_lock{};

    auto pageranks_view    = pageranks.view();
    auto renumber_map_view = renumber_map ? std::make_optional(renumber_map->view()) : std::nullopt;

    // Load computed_pageranks from different threads.
    for (int i = 0; i < num_gpus; ++i) {
      running_threads.emplace_back([&instance_manager,
                                    &graph_view,
                                    &renumber_map_view,
                                    &pageranks_view,
                                    &computed_pageranks_lock,
                                    &computed_pageranks_v,
                                    &h_src_v,
                                    &h_dst_v,
                                    &h_weights_v,
                                    &unique_vertices,
                                    i,
                                    num_threads]() {
        auto thread_handle = instance_manager->get_handle();

        auto number_of_vertices = unique_vertices->size();

        std::vector<vertex_t> my_vertex_list;
        my_vertex_list.reserve((number_of_vertices + num_threads - 1) / num_threads);

        for (size_t j = i; j < number_of_vertices; j += num_threads) {
          my_vertex_list.push_back((*unique_vertices)[j]);
        }

        rmm::device_uvector<vertex_t> d_my_vertex_list(my_vertex_list.size(),
                                                       thread_handle.raft_handle().get_stream());
        raft::update_device(d_my_vertex_list.data(),
                            my_vertex_list.data(),
                            my_vertex_list.size(),
                            thread_handle.raft_handle().get_stream());

        auto d_my_pageranks = pageranks_view.gather(
          thread_handle,
          raft::device_span<vertex_t const>{d_my_vertex_list.data(), d_my_vertex_list.size()},
          graph_view,
          renumber_map_view);

        std::vector<result_t> my_pageranks(d_my_pageranks.size());
        raft::update_host(my_pageranks.data(),
                          d_my_pageranks.data(),
                          d_my_pageranks.size(),
                          thread_handle.raft_handle().get_stream());

        {
          std::lock_guard<std::mutex> lock(computed_pageranks_lock);
          computed_pageranks_v.push_back(
            std::make_tuple(std::move(my_vertex_list), std::move(my_pageranks)));
        }
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    if (multithreaded_usecase.check_correctness) {
      // Want to compare the results in computed_pageranks_v with SG results
      cugraph::graph_t<vertex_t, edge_t, true, false> sg_graph(handle);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::optional<rmm::device_uvector<vertex_t>> sg_renumber_map{std::nullopt};

      std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, sg_renumber_map) = cugraph::
        create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, true, false>(
          handle,
          std::nullopt,
          std::move(d_src_v),
          std::move(d_dst_v),
          std::move(d_weights_v),
          std::nullopt,
          std::nullopt,
          cugraph::graph_properties_t{is_symmetric, true},
          true);

      auto [sg_pageranks, meta] = cugraph::pagerank<vertex_t, edge_t, weight_t, weight_t, false>(
        handle,
        sg_graph.view(),
        sg_edge_weights ? std::make_optional(sg_edge_weights->view()) : std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        alpha,
        epsilon);

      auto h_sg_pageranks    = cugraph::test::to_host(handle, sg_pageranks);
      auto h_sg_renumber_map = cugraph::test::to_host(handle, sg_renumber_map);
      auto compare_functor   = cugraph::test::nearly_equal<weight_t>{
        weight_t{1e-3},
        weight_t{(weight_t{1} / static_cast<weight_t>(h_sg_pageranks.size())) * weight_t{1e-3}}};

      std::for_each(computed_pageranks_v.begin(),
                    computed_pageranks_v.end(),
                    [h_sg_pageranks, compare_functor, h_sg_renumber_map](auto t1) {
                      std::for_each(
                        thrust::make_zip_iterator(std::get<0>(t1).begin(), std::get<1>(t1).begin()),
                        thrust::make_zip_iterator(std::get<0>(t1).end(), std::get<1>(t1).end()),
                        [h_sg_pageranks, compare_functor, h_sg_renumber_map](auto t2) {
                          vertex_t v  = thrust::get<0>(t2);
                          weight_t pr = thrust::get<1>(t2);

                          auto pos =
                            std::find(h_sg_renumber_map->begin(), h_sg_renumber_map->end(), v);
                          auto offset = std::distance(h_sg_renumber_map->begin(), pos);

                          if (pos == h_sg_renumber_map->end()) {
                            ASSERT_TRUE(compare_functor(pr, weight_t{0}))
                              << "vertex " << v << ", SG result = " << h_sg_pageranks[offset]
                              << ", mtmg result = " << pr << ", not in renumber map";
                          } else {
                            ASSERT_TRUE(compare_functor(pr, h_sg_pageranks[offset]))
                              << "vertex " << v << ", SG result = " << h_sg_pageranks[offset]
                              << ", mtmg result = " << pr
                              << ", renumber map = " << (*h_sg_renumber_map)[offset];
                          }
                        });
                    });
    }
  }
};

using Tests_Multithreaded_File = Tests_Multithreaded<cugraph::test::File_Usecase>;
using Tests_Multithreaded_Rmat = Tests_Multithreaded<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_Multithreaded_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), std::vector<int>{{0, 1}});
}

TEST_P(Tests_Multithreaded_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), std::vector<int>{{0, 1}});
}

INSTANTIATE_TEST_SUITE_P(file_test,
                         Tests_Multithreaded_File,
                         ::testing::Combine(
                           // enable correctness checks
                           ::testing::Values(Multithreaded_Usecase{false, true},
                                             Multithreaded_Usecase{true, true}),
                           ::testing::Values(cugraph::test::File_Usecase("karate.csv"),
                                             cugraph::test::File_Usecase("dolphins.csv"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Multithreaded_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Multithreaded_Usecase{false, true}, Multithreaded_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Multithreaded_File,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(Multithreaded_Usecase{false, false}, Multithreaded_Usecase{true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Multithreaded_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Multithreaded_Usecase{false, false}, Multithreaded_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

inline auto local_parse_test_options(int argc, char** argv)
{
  try {
    cxxopts::Options options(argv[0], " - cuGraph tests command line options");
    options.allow_unrecognised_options().add_options()(
      "rmm_mode", "RMM allocation mode", cxxopts::value<std::string>()->default_value("pool"))(
      "perf", "enalbe performance measurements", cxxopts::value<bool>()->default_value("false"))(
      "rmat_scale", "override the hardcoded R-mat scale", cxxopts::value<size_t>())(
      "rmat_edge_factor", "override the hardcoded R-mat edge factor", cxxopts::value<size_t>())(
      "node_rank", "rank of this process on multi-node configuration", cxxopts::value<int>())(
      "num_nodes", "number of nodes in this multi-node configuration", cxxopts::value<int>())(
      "comms_dir_name",
      "directory where comms data is stored (shared)",
      cxxopts::value<std::string>())(
      "test_file_name", "override the hardcoded test filename", cxxopts::value<std::string>());

    return options.parse(argc, argv);
  } catch (const cxxopts::OptionException& e) {
    CUGRAPH_FAIL("Error parsing command line options");
  }
}

//
// Need to customize the test configuration to support multi-node comms not using MPI
//
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto const cmd_opts = local_parse_test_options(argc, argv);
  auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();
  auto resource       = cugraph::test::create_memory_resource(rmm_mode);
  rmm::mr::set_current_device_resource(resource.get());
  cugraph::test::g_perf       = cmd_opts["perf"].as<bool>();
  cugraph::test::g_rmat_scale = (cmd_opts.count("rmat_scale") > 0)
                                  ? std::make_optional<size_t>(cmd_opts["rmat_scale"].as<size_t>())
                                  : std::nullopt;
  cugraph::test::g_rmat_edge_factor =
    (cmd_opts.count("rmat_edge_factor") > 0)
      ? std::make_optional<size_t>(cmd_opts["rmat_edge_factor"].as<size_t>())
      : std::nullopt;
  cugraph::test::g_test_file_name =
    (cmd_opts.count("test_file_name") > 0)
      ? std::make_optional<std::string>(cmd_opts["test_file_name"].as<std::string>())
      : std::nullopt;

  g_comms_dir_name = (cmd_opts.count("comms_dir_name") > 0)
                       ? cmd_opts["comms_dir_name"].as<std::string>()
                       : "COMMS_DIR";

  CUGRAPH_EXPECTS(cmd_opts.count("node_rank") > 0, "node_rank not specified");
  CUGRAPH_EXPECTS(cmd_opts.count("num_nodes") > 0, "num_nodes not specified");
  g_node_rank = cmd_opts["node_rank"].as<int>();
  g_num_nodes = cmd_opts["num_nodes"].as<int>();

  return RUN_ALL_TESTS();
}
