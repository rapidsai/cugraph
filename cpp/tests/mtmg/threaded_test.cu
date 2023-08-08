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
#include <cugraph/mtmg/renumber_map.hpp>
#include <cugraph/mtmg/resource_manager.hpp>
#include <cugraph/mtmg/thread_edgelist.hpp>
#include <cugraph/mtmg/vertex_result.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <nccl.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <cugraph/utilities/device_functors.cuh>
#include <detail/graph_partition_utils.cuh>
#include <thrust/count.h>

struct Multithreaded_Usecase {
  bool test_weighted{false};
  bool check_correctness{true};
};

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
    HighResTimer hr_timer{};

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    size_t device_buffer_size{64 * 1024 * 1024};
    size_t thread_buffer_size{4 * 1024 * 1024};

    int num_gpus    = gpu_list.size();
    int num_threads = num_gpus * 4;

    cugraph::mtmg::resource_manager_t resource_manager;

    std::for_each(gpu_list.begin(), gpu_list.end(), [&resource_manager](int gpu_id) {
      resource_manager.register_local_gpu(gpu_id, rmm::cuda_device_id{gpu_id});
    });

    ncclUniqueId instance_manager_id;
    ncclGetUniqueId(&instance_manager_id);

    std::cout << "create instance_manager" << std::endl;

    auto instance_manager = resource_manager.create_instance_manager(
      resource_manager.registered_ranks(), instance_manager_id);

    cugraph::mtmg::edgelist_t<vertex_t, weight_t, edge_t, edge_type_t> edgelist;
    cugraph::mtmg::graph_t<vertex_t, edge_t, true, multi_gpu> graph;
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu> graph_view;
    cugraph::mtmg::vertex_result_t<result_t> pageranks;
    std::optional<cugraph::mtmg::renumber_map_t<vertex_t>> renumber_map =
      std::make_optional<cugraph::mtmg::renumber_map_t<vertex_t>>();

    std::cout << "prepare to create edges" << std::endl;

    //
    // Simulate graph creation by spawning threads to walk through the
    // local COO and add edges
    //
    std::vector<std::thread> running_threads;

    //  Initialize shared edgelist object, one per GPU
    for (int i = 0; i < num_gpus; ++i) {
      running_threads.emplace_back([&instance_manager,
                                    &edgelist,
                                    &renumber_map,
                                    i,
                                    num_gpus,
                                    device_buffer_size,
                                    use_weight    = true,
                                    use_edge_id   = false,
                                    use_edge_type = false]() {
        auto thread_handle = instance_manager->get_handle();

        edgelist.initialize_pointer(
          thread_handle, thread_handle, device_buffer_size, use_weight, use_edge_id, use_edge_type);
        if (renumber_map)
          renumber_map->initialize_pointer(
            thread_handle, 0, thread_handle.raft_handle().get_stream());
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    std::cout << "load sg edge list" << std::endl;

    // Load SG edge list
    auto [d_src_v, d_dst_v, d_weights_v, d_vertices_v, is_symmetric] =
      input_usecase.template construct_edgelist<vertex_t, weight_t>(
        handle, multithreaded_usecase.test_weighted, false, false);

    auto h_src_v     = cugraph::test::to_host(handle, d_src_v);
    auto h_dst_v     = cugraph::test::to_host(handle, d_dst_v);
    auto h_weights_v = cugraph::test::to_host(handle, d_weights_v);

    std::cout << "load edgelist" << std::endl;

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
        sleep(i);
        auto thread_handle = instance_manager->get_handle();
        cugraph::mtmg::thread_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t> thread_edgelist(
          *edgelist.get_pointer(thread_handle), thread_buffer_size);

        for (int j = i; j < h_src_v.size(); j += num_threads) {
          if (h_weights_v) {
            thread_edgelist.append(
              thread_handle, h_src_v[j], h_dst_v[j], (*h_weights_v)[j], std::nullopt, std::nullopt);
          } else {
            thread_edgelist.append(
              thread_handle, h_src_v[j], h_dst_v[j], std::nullopt, std::nullopt, std::nullopt);
          }
        }

        thread_edgelist.flush(thread_handle);
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    std::cout << "now create Graph, num_gpus = " << num_gpus << std::endl;

    // TODO: At this point, the edgelist should be complete on the GPU.  We should be able to create
    // the graph
    //    Should test case where this loop runs more than num_gpus times and less.  Ideally, more
    //    should work fine, less should fail.
    for (int i = 0; i < num_gpus; ++i) {
      running_threads.emplace_back([&instance_manager,
                                    &graph,
                                    &graph_view,
                                    &edgelist,
                                    &renumber_map,
                                    &pageranks,
                                    is_symmetric = is_symmetric,
                                    renumber,
                                    do_expensive_check]() {
        auto thread_handle = instance_manager->get_handle();

        std::cout << "in threads, rank = " << thread_handle.get_rank()
                  << ", thread_rank = " << thread_handle.get_thread_rank() << std::endl;

        if (thread_handle.get_thread_rank() > 0) return;

        std::optional<cugraph::mtmg::edge_property_t<
          cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
          weight_t>>
          edge_weights{std::nullopt};
        std::optional<cugraph::mtmg::edge_property_t<
          cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
          edge_t>>
          edge_ids{std::nullopt};
        std::optional<cugraph::mtmg::edge_property_t<
          cugraph::mtmg::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
          int32_t>>
          edge_types{std::nullopt};

        graph.initialize_pointer(thread_handle, thread_handle.raft_handle());

        edgelist.get_pointer(thread_handle)->finalize_buffer(thread_handle);
        edgelist.get_pointer(thread_handle)->consolidate_and_shuffle(thread_handle, true);

        raft::print_device_vector(" edgelist_majors",
                                  edgelist.get_pointer(thread_handle)->get_dst()[0].data(),
                                  edgelist.get_pointer(thread_handle)->get_dst()[0].size(),
                                  std::cout);
        raft::print_device_vector(" edgelist_minors",
                                  edgelist.get_pointer(thread_handle)->get_src()[0].data(),
                                  edgelist.get_pointer(thread_handle)->get_src()[0].size(),
                                  std::cout);

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

        graph.set_view(thread_handle, graph_view);
        pageranks.initialize_pointer(
          thread_handle,
          graph_view.get_pointer(thread_handle)->local_vertex_partition_range_size(),
          thread_handle.raft_handle().get_stream());
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    //   TODO: Try a facade for mtmg::pagerank
    //
    //    Should test case where this loop runs more than num_gpus times and less.  Ideally, more
    //    should work fine, less should fail.
    for (int i = 0; i < num_gpus; ++i) {
      running_threads.emplace_back([&instance_manager, &graph_view, &pageranks, alpha, epsilon]() {
        auto thread_handle = instance_manager->get_handle();

        if (thread_handle.get_thread_rank() > 0) return;

        // initialize to 0 for now
        auto& p = *pageranks.get_pointer(thread_handle);
        cugraph::detail::scalar_fill(thread_handle.raft_handle(), p.data(), p.size(), weight_t{0});
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();

    std::cout << "compute pageranks..." << std::endl;

    std::vector<std::tuple<std::vector<vertex_t>, std::vector<result_t>>> computed_pageranks_v;
    std::mutex computed_pageranks_lock{};

    // Load computed_pageranks from different threads.  We'll use more threads than GPUs here
    // for (int i = 0; i < num_threads; ++i) {
    for (int i = 0; i < num_gpus; ++i) {
      running_threads.emplace_back([&instance_manager,
                                    &graph_view,
                                    &pageranks,
                                    &computed_pageranks_lock,
                                    &computed_pageranks_v,
                                    &h_src_v,
                                    &h_dst_v,
                                    &h_weights_v,
                                    i,
                                    num_threads]() {
        auto thread_handle = instance_manager->get_handle();

        auto number_of_vertices = graph_view.get_pointer(thread_handle)->number_of_vertices();

        std::cout << "number_of_vertices on rank " << thread_handle.get_rank() << " = "
                  << number_of_vertices << std::endl;

        std::vector<vertex_t> my_vertex_list;
        my_vertex_list.reserve((number_of_vertices + num_threads - 1) / num_threads);

        for (int j = i; j < number_of_vertices; j += num_threads) {
          my_vertex_list.push_back(j);
        }

        rmm::device_uvector<vertex_t> d_my_vertex_list(my_vertex_list.size(),
                                                       thread_handle.raft_handle().get_stream());
        raft::update_device(d_my_vertex_list.data(),
                            my_vertex_list.data(),
                            my_vertex_list.size(),
                            thread_handle.raft_handle().get_stream());

        sleep(thread_handle.get_rank());
        std::cout << "calling pageranks.gather, rank = " << thread_handle.get_rank() << std::endl;
        raft::print_device_vector("  pageranks",
                                  pageranks.get_pointer(thread_handle)->data(),
                                  pageranks.get_pointer(thread_handle)->size(),
                                  std::cout);
        auto d_my_pageranks = pageranks.gather(
          thread_handle,
          raft::device_span<vertex_t const>{d_my_vertex_list.data(), d_my_vertex_list.size()},
          graph_view);

        std::vector<result_t> my_pageranks(d_my_pageranks.size());
        raft::update_host(my_pageranks.data(),
                          d_my_pageranks.data(),
                          d_my_pageranks.size(),
                          thread_handle.raft_handle().get_stream());

        {
          std::lock_guard<std::mutex> lock(computed_pageranks_lock);
          raft::print_host_vector(
            "  my_vertex_list", my_vertex_list.data(), my_vertex_list.size(), std::cout);
          raft::print_host_vector(
            "  my_pageranks", my_pageranks.data(), my_pageranks.size(), std::cout);
          computed_pageranks_v.push_back(
            std::make_tuple(std::move(my_vertex_list), std::move(my_pageranks)));
        }
      });
    }

    // Wait for CPU threads to complete
    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
    running_threads.resize(0);
    instance_manager->reset_threads();
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
    ::testing::Values(Multithreaded_Usecase{true, false}, Multithreaded_Usecase{true, true}),
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

CUGRAPH_TEST_PROGRAM_MAIN()
