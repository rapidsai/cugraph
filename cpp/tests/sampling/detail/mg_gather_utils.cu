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

#include "nbr_sampling_utils.cuh"

#include <sampling/detail/graph_functions.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>

#include <gtest/gtest.h>

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MG_GatherEdges
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_GatherEdges() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using namespace cugraph::test;
    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) {
      --row_comm_size;
    }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    constexpr bool sort_adjacency_list = true;

    auto [mg_graph, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        handle, input_usecase, true, true, false, sort_adjacency_list);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view                        = mg_graph.view();
    constexpr edge_t indices_per_source       = 2;
    constexpr vertex_t repetitions_per_vertex = 5;
    constexpr vertex_t source_sample_count    = 3;

    // 3. Gather mnmg call
    // Generate random vertex ids in the range of current gpu

    auto [global_degree_offsets, global_out_degrees] =
      cugraph::detail::get_global_degree_information(handle, mg_graph_view);
    auto global_adjacency_list_offsets = cugraph::detail::get_global_adjacency_offset(
      handle, mg_graph_view, global_degree_offsets, global_out_degrees);

    // Generate random sources to gather on
    auto random_sources =
      random_vertex_ids(handle,
                        mg_graph_view.local_vertex_partition_range_first(),
                        mg_graph_view.local_vertex_partition_range_last(),
                        std::min(mg_graph_view.local_vertex_partition_range_size() *
                                   (repetitions_per_vertex + vertex_t{1}),
                                 source_sample_count),
                        repetitions_per_vertex);

    auto active_sources =
      cugraph::detail::allgather_active_majors(handle, std::move(random_sources));

    // get source global out degrees to generate indices
    auto active_source_degrees = cugraph::detail::get_active_major_global_degrees(
      handle, mg_graph_view, active_sources, global_out_degrees);

    auto random_destination_offsets =
      generate_random_destination_indices(handle,
                                          active_source_degrees,
                                          mg_graph_view.number_of_vertices(),
                                          mg_graph_view.number_of_edges(),
                                          indices_per_source);

    std::vector<edge_t> input_destination_offsets(random_destination_offsets.size());
    raft::update_host(input_destination_offsets.data(),
                      random_destination_offsets.data(),
                      random_destination_offsets.size(),
                      handle.get_stream());

    sleep(handle.get_comms().get_rank());
    std::cout << "rank = " << handle.get_comms().get_rank() << std::endl;
    raft::print_device_vector(
      "active_sources", active_sources.data(), active_sources.size(), std::cout);
    raft::print_device_vector("random_destination_offsets",
                              random_destination_offsets.data(),
                              random_destination_offsets.size(),
                              std::cout);

    auto [src, dst, dst_map] =
      cugraph::detail::gather_local_edges(handle,
                                          mg_graph_view,
                                          active_sources,
                                          std::move(random_destination_offsets),
                                          indices_per_source,
                                          global_degree_offsets,
                                          global_adjacency_list_offsets);

    if (prims_usecase.check_correctness) {
      // I want to filter the active_sources/input_destination_offsets to identify versions that are
      // relevant to this rank
      std::vector<vertex_t> input_sources(input_destination_offsets.size());
      std::vector<vertex_t> h_active_sources(active_sources.size());
      std::vector<edge_t> h_global_adjacency_list_offsets(global_adjacency_list_offsets.size());

      raft::update_host(
        h_active_sources.data(), active_sources.data(), active_sources.size(), handle.get_stream());

      raft::update_host(h_global_adjacency_list_offsets.data(),
                        global_adjacency_list_offsets.data(),
                        global_adjacency_list_offsets.size(),
                        handle.get_stream());

      raft::print_device_vector("global_adjacency_list_offsets",
                                global_adjacency_list_offsets.data(),
                                global_adjacency_list_offsets.size(),
                                std::cout);

      thrust::tabulate(thrust::seq,
                       input_sources.begin(),
                       input_sources.end(),
                       [&h_active_sources, indices_per_source] __host__(auto i) {
                         return h_active_sources[i / indices_per_source];
                       });

      auto begin_iter =
        thrust::make_zip_iterator(input_sources.begin(), input_destination_offsets.begin());

      auto end_iter = thrust::remove_if(
        thrust::host,
        begin_iter,
        begin_iter + input_sources.size(),
        [&mg_graph_view, &h_global_adjacency_list_offsets] __host__(auto p) {
          auto source = thrust::get<0>(p);
          auto offset = thrust::get<1>(p);

          // Check if source is a src on this GPU
          bool remove{offset == mg_graph_view.number_of_edges()};
          for (size_t i = 0; remove && (i < mg_graph_view.number_of_local_edge_partitions()); ++i) {
            remove = ((source >= mg_graph_view.local_edge_partition_src_range_first(i)) &&
                      (source < mg_graph_view.local_edge_partition_src_range_last(i)));
          }

          if (!remove) {
            //remove = true;

            // h_global_adjacency_list_offsets

            // check offset is in range on this GPU
          }

          return remove;
        });

      input_sources.resize(thrust::distance(begin_iter, end_iter));
      input_destination_offsets.resize(thrust::distance(begin_iter, end_iter));

      sleep(handle.get_comms().get_rank());
      std::cout << "rank = " << handle.get_comms().get_rank() << std::endl;
      raft::print_host_vector(
        "input_sources", input_sources.data(), input_sources.size(), std::cout);
      raft::print_host_vector("input_destination_offsets",
                              input_destination_offsets.data(),
                              input_destination_offsets.size(),
                              std::cout);

#if 0
      // Gather relevant edges from graph
      auto& col_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_rank = col_comm.get_rank();
      auto all_active_sources = cugraph::test::device_allgatherv(
        handle,
        raft::device_span<vertex_t const>{active_sources.data(),
                                          col_rank == 0 ? active_sources.size() : 0});

      thrust::sort(
        handle.get_thrust_policy(), all_active_sources.begin(), all_active_sources.end());

      // 

      // extract COO on each GPU
      rmm::device_uvector<vertex_t> mg_src(0, handle.get_stream());
      rmm::device_uvector<vertex_t> mg_dst(0, handle.get_stream());
      std::tie(mg_src, mg_dst, std::ignore) =
        mg_graph_view.decompress_to_edgelist(handle, std::nullopt);

      auto begin_iter = thrust::make_zip_iterator(mg_src.begin(), mg_dst.begin());
      auto new_end    = thrust::remove_if(
        handle.get_thrust_policy(),
        begin_iter,
        begin_iter + mg_src.size(),
        [sources = all_active_sources.data(), size = all_active_sources.size()] __device__(auto t) {
          auto src = thrust::get<0>(t);
          return !thrust::binary_search(thrust::seq, sources, sources + size, src);
        });

      mg_src.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
      mg_dst.resize(thrust::distance(begin_iter, new_end), handle.get_stream());

      // ASSUMPTION: Sorting the edges incident on a vertex results in a
      // consistent order to how they are organized in the graph data structure
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(mg_src.begin(), mg_dst.begin()),
                   thrust::make_zip_iterator(mg_src.end(), mg_dst.end()));

      // What to do from here???
      //   compare
      std::vector<vertex_t> local_src(mg_src.size());
      std::vector<vertex_t> local_dst(mg_dst.size());

      // copy values to local_src/local_dst
      // copy

      // Gather outputs
      auto mg_out_srcs = cugraph::test::device_gatherv(handle, src.data(), src.size());
      auto mg_out_dsts = cugraph::test::device_gatherv(handle, dst.data(), dst.size());

      // Gather inputs
      auto& col_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_rank = col_comm.get_rank();
      auto sg_random_srcs = cugraph::test::device_gatherv(
        handle, active_sources.data(), col_rank == 0 ? active_sources.size() : 0);
      auto sg_random_dst_indices =
        cugraph::test::device_gatherv(handle,
                                      random_destination_indices.data(),
                                      col_rank == 0 ? random_destination_indices.size() : 0);

      auto aggregated_sg_src = cugraph::test::device_gatherv(handle, sg_src.begin(), sg_src.size());
      auto aggregated_sg_dst = cugraph::test::device_gatherv(handle, sg_dst.begin(), sg_dst.size());

      sort_coo(handle, mg_out_srcs, mg_out_dsts);

      if (handle.get_comms().get_rank() == int{0}) {
        cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> sg_graph(handle);
        auto aggregated_edge_iter = thrust::make_zip_iterator(
          thrust::make_tuple(aggregated_sg_src.begin(), aggregated_sg_dst.begin()));
        thrust::sort(handle.get_thrust_policy(),
                     aggregated_edge_iter,
                     aggregated_edge_iter + aggregated_sg_src.size());
        auto sg_graph_properties =
          cugraph::graph_properties_t{mg_graph_view.is_symmetric(), mg_graph_view.is_multigraph()};

        std::tie(sg_graph, std::ignore) =
          cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, false, false>(
            handle,
            std::nullopt,
            std::move(aggregated_sg_src),
            std::move(aggregated_sg_dst),
            std::nullopt,
            sg_graph_properties,
            false);
        auto sg_graph_view = sg_graph.view();
        // Call single gpu gather
        auto [sg_out_srcs, sg_out_dsts] = sg_gather_edges(handle,
                                                          sg_graph_view,
                                                          sg_random_srcs.begin(),
                                                          sg_random_srcs.end(),
                                                          sg_random_dst_indices.begin(),
                                                          sg_graph_view.number_of_vertices(),
                                                          indices_per_source);
        sort_coo(handle, sg_out_srcs, sg_out_dsts);

        auto passed = thrust::equal(
          handle.get_thrust_policy(), sg_out_srcs.begin(), sg_out_srcs.end(), mg_out_srcs.begin());
        passed &= thrust::equal(
          handle.get_thrust_policy(), sg_out_dsts.begin(), sg_out_dsts.end(), mg_out_dsts.begin());
        ASSERT_TRUE(passed);
      }
#endif
    }
  }
};

using Tests_MG_GatherEdges_File = Tests_MG_GatherEdges<cugraph::test::File_Usecase>;

using Tests_MG_GatherEdges_Rmat = Tests_MG_GatherEdges<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_GatherEdges_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_File, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_File, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_GatherEdges_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_GatherEdges_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MG_GatherEdges_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
