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

#include <prims/property_generator.cuh>

#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/kv_store.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/collect_comm.cuh>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

template <typename weight_t>
struct random_op_t{
  thrust::minstd_rand rng{};
  thrust::uniform_real_distribution<weight_t> dist{};

  __device__ weight_t operator()(auto x)
  {
    rng.discard(x);
    weight_t random_number = dist(rng);
    printf("%f\n", random_number);
    return static_cast<weight_t>(random_number);
  }
};

// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.

unsigned TausStep(unsigned& z, int S1, int S2, int S3, unsigned M)
{
  unsigned b = (((z << S1) ^ z) >> S2);
  return z   = (((z & M) << S3) ^ b);
}

// A and C are constants

unsigned LCGStep(unsigned& z, unsigned A, unsigned C) { return z = (A * z + C); }

unsigned z1, z2, z3, z4;
float HybridTaus()
{
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return 2.3283064365387e-10 * (
                                 // Periods
                                 TausStep(z1, 13, 19, 12, 4294967294UL) ^
                                 // p1=2^31-1
                                 TausStep(z2, 2, 25, 4, 4294967288UL) ^
                                 // p2=2^30-1
                                 TausStep(z3, 3, 11, 17, 4294967280UL) ^
                                 // p3=2^28-1
                                 LCGStep(z4, 1664525, 1013904223UL)
                                 // p4=2^32
                               );
}

struct MaximalIndependentSet_Usecase {
  size_t select_count{std::numeric_limits<size_t>::max()};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGMaximalIndependentSet
  : public ::testing::TestWithParam<std::tuple<MaximalIndependentSet_Usecase, input_usecase_t>> {
 public:
  Tests_MGMaximalIndependentSet() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<MaximalIndependentSet_Usecase, input_usecase_t> const& param)
  {
    auto [mis_usecase, input_usecase] = param;

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    thrust::minstd_rand rng(seed);
    thrust::uniform_real_distribution<float> dist(0, 1);

    thrust::for_each(handle_->get_thrust_policy(),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(10),
                     random_op_t<weight_t>{rng, dist});

#if 0
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    constexpr bool multi_gpu = true;
    rmm::device_uvector<vertex_t> leiden_assignment = rmm::device_uvector<vertex_t>(
      graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    cugraph::detail::sequence_fill(handle_->get_stream(),
                                   leiden_assignment.begin(),
                                   leiden_assignment.size(),
                                   graph_view.local_vertex_partition_range_first());

    std::vector<weight_t> h_louvain_assignment(leiden_assignment.size());

    auto nV = graph_view.number_of_vertices();
    std::srand(comm_rank);

    for (size_t k = 0; k < leiden_assignment.size(); k++) {
      h_louvain_assignment[k] = std::rand() % nV;
    }

    auto louvain_assignment_of_vertices = cugraph::test::to_device(*handle_, h_louvain_assignment);

    cugraph::kv_store_t<vertex_t, vertex_t, false> leiden_to_louvain_map(
      leiden_assignment.begin(),
      leiden_assignment.end(),
      louvain_assignment_of_vertices.begin(),
      cugraph::invalid_vertex_id<vertex_t>::value,
      cugraph::invalid_vertex_id<vertex_t>::value,
      handle_->get_stream());

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int i = 0; i < comm_size; ++i) {
        handle_->get_comms().barrier();
        if (comm_rank == i) {
          std::cout << "Rank: " << i << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector(
            "map keys", leiden_assignment.data(), leiden_assignment.size(), std::cout);
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("map values",
                                    louvain_assignment_of_vertices.data(),
                                    louvain_assignment_of_vertices.size(),
                                    std::cout);
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
        }
        handle_->get_comms().barrier();
      }
    }

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto partitions_range_lasts =
        cugraph::test::to_device(*handle_, graph_view.vertex_partition_range_lasts());
      rmm::device_uvector<vertex_t> gpu_ids(leiden_assignment.size(), handle_->get_stream());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector("partitions_range_lasts:",
                                partitions_range_lasts.data(),
                                partitions_range_lasts.size(),
                                std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      raft::print_device_vector(
        "leiden_assignment:", leiden_assignment.data(), leiden_assignment.size(), std::cout);

      thrust::transform(handle_->get_thrust_policy(),
                        leiden_assignment.begin(),
                        leiden_assignment.end(),
                        gpu_ids.begin(),
                        [

                          key_func =
                            cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                              raft::device_span<vertex_t const>(partitions_range_lasts.data(),
                                                                partitions_range_lasts.size()),
                              major_comm_size,
                              minor_comm_size},
                          comm_rank,
                          comm_size] __device__(auto key) {
                          if (key_func(key) != comm_rank) {
                            printf("--- key %d must not be in GPU-%d\n", key, comm_rank);
                          } else {
                            printf("+++ key %d is on right GPU \n", key);
                          }
                          return key_func(key);
                          // return key % comm_size;
                        });

      handle_->sync_stream();

      auto h_gpu_ids = cugraph::test::to_host(*handle_, gpu_ids);

      std::cout << "h_gpu_ids (size" << h_gpu_ids.size() << "):" << std::endl;
      std::copy(h_gpu_ids.begin(), h_gpu_ids.end(), std::ostream_iterator<int>(std::cout, " "));

      std::cout << std::endl;
    }

    //------
    rmm::device_uvector<vertex_t> leiden_keys_to_read_louvain(leiden_assignment.size(),
                                                              handle_->get_stream());

    thrust::copy(handle_->get_thrust_policy(),
                 leiden_assignment.begin(),
                 leiden_assignment.end(),
                 leiden_keys_to_read_louvain.begin());

    thrust::sort(handle_->get_thrust_policy(),
                 leiden_keys_to_read_louvain.begin(),
                 leiden_keys_to_read_louvain.end());

    auto nr_unique_leiden_clusters =
      static_cast<size_t>(thrust::distance(leiden_keys_to_read_louvain.begin(),
                                           thrust::unique(handle_->get_thrust_policy(),
                                                          leiden_keys_to_read_louvain.begin(),
                                                          leiden_keys_to_read_louvain.end())));

    leiden_keys_to_read_louvain.resize(nr_unique_leiden_clusters, handle_->get_stream());

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int i = 0; i < comm_size; ++i) {
        handle_->get_comms().barrier();
        if (comm_rank == i) {
          std::cout << "Rank: " << i << std::endl;

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          std::cout << "nr_unique_leiden_clusters(before shuffle): " << nr_unique_leiden_clusters
                    << std::endl;

          raft::print_device_vector("leiden_keys_to_read_louvain(before shuffle) :",
                                    leiden_keys_to_read_louvain.data(),
                                    leiden_keys_to_read_louvain.size(),
                                    std::cout);
        }
        handle_->get_comms().barrier();
      }
    }

    if constexpr (multi_gpu) {
      // leiden_keys_to_read_louvain =
      //   cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
      //     *handle_, std::move(leiden_keys_to_read_louvain));

      leiden_keys_to_read_louvain =
        cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
          *handle_,
          std::move(leiden_keys_to_read_louvain),
          graph_view.vertex_partition_range_lasts());

      thrust::sort(handle_->get_thrust_policy(),
                   leiden_keys_to_read_louvain.begin(),
                   leiden_keys_to_read_louvain.end());

      nr_unique_leiden_clusters =
        static_cast<size_t>(thrust::distance(leiden_keys_to_read_louvain.begin(),
                                             thrust::unique(handle_->get_thrust_policy(),
                                                            leiden_keys_to_read_louvain.begin(),
                                                            leiden_keys_to_read_louvain.end())));
      leiden_keys_to_read_louvain.resize(nr_unique_leiden_clusters, handle_->get_stream());
    }

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int i = 0; i < comm_size; ++i) {
        handle_->get_comms().barrier();
        if (comm_rank == i) {
          std::cout << "Rank: " << i << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          std::cout << "nr_unique_leiden_clusters(after shuffle): " << nr_unique_leiden_clusters
                    << std::endl;
          raft::print_device_vector("leiden_keys_to_read_louvain(after shuffle) :",
                                    leiden_keys_to_read_louvain.data(),
                                    leiden_keys_to_read_louvain.size(),
                                    std::cout);
        }
        handle_->get_comms().barrier();
      }
    }

    //---

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int i = 0; i < comm_size; ++i) {
        handle_->get_comms().barrier();
        if (comm_rank == i) {
          std::cout << "Rank: " << i << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector(
            "map keys*", leiden_assignment.data(), leiden_assignment.size(), std::cout);
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("map values*",
                                    louvain_assignment_of_vertices.data(),
                                    louvain_assignment_of_vertices.size(),
                                    std::cout);
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
        }
        handle_->get_comms().barrier();
      }
    }

    rmm::device_uvector<vertex_t> lovain_of_leiden_cluster_keys(0, handle_->get_stream());
    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto partitions_range_lasts =
        cugraph::test::to_device(*handle_, graph_view.vertex_partition_range_lasts());

      cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> vertex_to_gpu_id_op{
        raft::device_span<vertex_t const>(partitions_range_lasts.data(),
                                          partitions_range_lasts.size()),
        major_comm_size,
        minor_comm_size};

      // cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      //   comm_size, major_comm_size, minor_comm_size};

      lovain_of_leiden_cluster_keys =
        cugraph::collect_values_for_keys(*handle_,
                                         leiden_to_louvain_map.view(),
                                         leiden_keys_to_read_louvain.begin(),
                                         leiden_keys_to_read_louvain.end(),
                                         vertex_to_gpu_id_op);
    }

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int i = 0; i < comm_size; ++i) {
        handle_->get_comms().barrier();
        if (comm_rank == i) {
          std::cout << "Rank: " << i << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("leiden_keys_to_read_louvain(MG)",
                                    leiden_keys_to_read_louvain.data(),
                                    leiden_keys_to_read_louvain.size(),
                                    std::cout);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("lovain_of_leiden_cluster_keys(MG)",
                                    lovain_of_leiden_cluster_keys.data(),
                                    lovain_of_leiden_cluster_keys.size(),
                                    std::cout);
        }
        handle_->get_comms().barrier();
      }
    }
#endif

    //--
    /*
    //
    // Create decision graph from edgelist
    //
    constexpr bool store_transposed = false;
    using DecisionGraphViewType     = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> decision_graph(*handle_);

    std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
    std::optional<cugraph::edge_property_t<DecisionGraphViewType, weight_t>> coarse_edge_weights{
      std::nullopt};

    vertex_t nr_valid_tuples = 0;
    if (comm_rank == comm_size - 1) nr_valid_tuples = 1;

    std::vector<vertex_t> h_srcs(nr_valid_tuples);
    std::vector<vertex_t> h_dsts(nr_valid_tuples);
    std::vector<weight_t> h_weights(nr_valid_tuples);

    // rmm::device_uvector<vertex_t> d_srcs(nr_valid_tuples, handle_->get_stream());
    // rmm::device_uvector<vertex_t> d_dsts(nr_valid_tuples, handle_->get_stream());
    // std::optional<rmm::device_uvector<weight_t>> d_weights =
    //   std::make_optional(rmm::device_uvector<weight_t>(nr_valid_tuples, handle_->get_stream()));

    auto& comm       = handle_->get_comms();
    auto& major_comm = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    auto gpu_id_key_func = cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size};
    std::srand(comm_rank);

    for (vertex_t i = 0; i < nr_valid_tuples; i++) {
      h_srcs[i]    = std::rand() % (1 << 10);
      h_dsts[i]    = h_srcs[i] + 1;
      h_weights[i] = (std::rand() % (1 << 10)) * 0.1;
      std::cout << "(" << h_srcs[i] << "," << h_dsts[i] << ") => "
                << gpu_id_key_func(h_srcs[i], h_dsts[i]) << std::endl;
    }

    auto d_srcs    = cugraph::test::to_device(*handle_, h_srcs);
    auto d_dsts    = cugraph::test::to_device(*handle_, h_dsts);
    auto d_weights = std::make_optional(cugraph::test::to_device(*handle_, h_weights));

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (size_t k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          std::cout << "Rank :" << comm_rank << std::endl;

          std::cout << " d_srcs.size(): " << d_srcs.size() << " d_dsts.size(): " << d_dsts.size()
                    << " (*d_weights).size(): " << (*d_weights).size() << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("d_srcs: ", d_srcs.data(), d_srcs.size(), std::cout);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("d_dsts: ", d_dsts.data(), d_dsts.size(), std::cout);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector(
            "(*d_weights): ", (*d_weights).data(), (*d_weights).size(), std::cout);

          std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }
    }

    std::tie(store_transposed ? d_dsts : d_srcs,
             store_transposed ? d_srcs : d_dsts,
             d_weights,
             std::ignore,
             std::ignore) =
      cugraph::detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
        vertex_t,
        vertex_t,
        weight_t,
        int32_t>(*handle_,
                 store_transposed ? std::move(d_dsts) : std::move(d_srcs),
                 store_transposed ? std::move(d_srcs) : std::move(d_dsts),
                 std::move(d_weights),
                 std::nullopt,
                 std::nullopt);

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (size_t k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          std::cout << "Rank :" << comm_rank << std::endl;

          std::cout << " d_srcs.size(): " << d_srcs.size() << " d_dsts.size(): " << d_dsts.size()
                    << " (*d_weights).size(): " << (*d_weights).size() << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("d_srcs: ", d_srcs.data(), d_srcs.size(), std::cout);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector("d_dsts: ", d_dsts.data(), d_dsts.size(), std::cout);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          raft::print_device_vector(
            "(*d_weights): ", (*d_weights).data(), (*d_weights).size(), std::cout);

          std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }
    }

    std::cout << "Before create_graph_from_edgelist ... " << std::endl;
    std::tie(decision_graph, coarse_edge_weights, std::ignore, std::ignore, renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t,
                                          edge_t,
                                          weight_t,
                                          edge_t,
                                          int32_t,
                                          store_transposed,
                                          multi_gpu>(*handle_,
                                                     std::nullopt,
                                                     std::move(d_srcs),
                                                     std::move(d_dsts),
                                                     std::nullopt,  // std::move(d_weights),
                                                     std::nullopt,
                                                     std::nullopt,
                                                     cugraph::graph_properties_t{false, false},
                                                     true,
                                                     true);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Returned from create_graph_from_edgelist" << std::endl;
    auto decision_graph_view = decision_graph.view();*/
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGMaximalIndependentSet<input_usecase_t>::handle_ = nullptr;

using Tests_MGMaximalIndependentSet_File =
  Tests_MGMaximalIndependentSet<cugraph::test::File_Usecase>;
using Tests_MGMaximalIndependentSet_Rmat =
  Tests_MGMaximalIndependentSet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

// TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt64Int64FloatFloat)
// {
//   run_current_test<int64_t, int64_t, float, int>(
//     override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
// }

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_MGMaximalIndependentSet_File,
  ::testing::Combine(::testing::Values(
                       // MaximalIndependentSet_Usecase{20, false},
                       MaximalIndependentSet_Usecase{20, false}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

// INSTANTIATE_TEST_SUITE_P(
//   rmat_small_test,
//   Tests_MGMaximalIndependentSet_Rmat,
//   ::testing::Combine(
//     ::testing::Values(MaximalIndependentSet_Usecase{50, true}),
//     ::testing::Values(cugraph::test::Rmat_Usecase(3, 4, 0.57, 0.19, 0.19, 0, true, false))));

// INSTANTIATE_TEST_SUITE_P(
//   rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
//                           --gtest_filter to select only the rmat_benchmark_test with a specific
//                           vertex & edge type combination) by command line arguments and do not
//                           include more than one Rmat_Usecase that differ only in scale or edge
//                           factor (to avoid running same benchmarks more than once) */
//   Tests_MGMaximalIndependentSet_Rmat,
//   ::testing::Combine(
//     ::testing::Values(MaximalIndependentSet_Usecase{500, false},
//                       MaximalIndependentSet_Usecase{500, false}),
//     ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
