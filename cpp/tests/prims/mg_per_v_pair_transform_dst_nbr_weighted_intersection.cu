/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <prims/per_v_pair_transform_dst_nbr_intersection.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <utilities/thrust_wrapper.hpp>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename edge_t, typename weight_t>
struct intersection_op_t {
  __device__ thrust::tuple<edge_t, edge_t> operator()(
    vertex_t v0,
    vertex_t v1,
    edge_t v0_prop /* out degree */,
    edge_t v1_prop /* out degree */,
    raft::device_span<vertex_t const> intersection,
    raft::device_span<weight_t const> intersection_p0,
    raft::device_span<weight_t const> intersection_p1) const
  {
    // printf("\n%d %d %d %d %d\n",
    //        static_cast<int>(v0),
    //        static_cast<int>(v1),
    //        static_cast<int>(v0_prop),
    //        static_cast<int>(v1_prop),
    //        static_cast<int>(intersection.size()));
    return thrust::make_tuple(v0_prop + v1_prop, static_cast<edge_t>(intersection.size()));
  }
};

struct Prims_Usecase {
  size_t num_vertex_pairs{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPerVPairTransformDstNbrIntersection
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGPerVPairTransformDstNbrIntersection() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Verify the results of per_v_pair_transform_dst_nbr_intersection primitive
  template <typename vertex_t, typename edge_t, typename weight_t, typename property_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};

    // std::tie(mg_graph, std::ignore, mg_renumber_map) =
    //   cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
    //     *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    std::string file_path = "/home/nfs/mnaim/csv/similarity.csv";

    constexpr bool store_transposed = false;
    constexpr bool multi_gpu        = true;

    std::optional<
      cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                               weight_t>>
      edge_weights{std::nullopt};

    /*
      ///
      //
      // Create decision graph from edgelist
      //

      // using DecisionGraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

      // cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> decision_graph(*handle_);

      // std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
      // std::optional<cugraph::edge_property_t<DecisionGraphViewType, weight_t>>
      coarse_edge_weights{
      //   std::nullopt};

      vertex_t N               = 4;
      vertex_t nr_valid_tuples = N * N - N;

      std::vector<vertex_t> h_srcs(nr_valid_tuples);
      std::vector<vertex_t> h_dsts(nr_valid_tuples);
      std::vector<weight_t> h_weights(nr_valid_tuples);

      // rmm::device_uvector<vertex_t> d_srcs(nr_valid_tuples, handle_->get_stream());
      // rmm::device_uvector<vertex_t> d_dsts(nr_valid_tuples, handle_->get_stream());
      // std::optional<rmm::device_uvector<weight_t>> d_weights =
      //   std::make_optional(rmm::device_uvector<weight_t>(nr_valid_tuples,
      handle_->get_stream()));

      auto& comm       = handle_->get_comms();
      auto& major_comm = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto gpu_id_key_func = cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
        comm_size, major_comm_size, minor_comm_size};
      std::srand(comm_rank);

      int edge_counter = 0;
      for (vertex_t i = 0; i < N; i++) {
        for (vertex_t j = 0; j < N; j++) {
          if (i != j) {
            h_srcs[edge_counter]    = i;
            h_dsts[edge_counter]    = j;
            h_weights[edge_counter] = std::max(i, j) * 10 + std::min(i, j);
            edge_counter++;
          }
        }
      }

      comm.barrier();
      if (comm_rank == 0)
        for (int i = 0; i < edge_counter; i++) {
          std::cout << "(" << h_srcs[i] << "," << h_dsts[i] << ") => "
                    << gpu_id_key_func(h_srcs[i], h_dsts[i]) << std::endl;
        }
      comm.barrier();

      auto d_srcs    = cugraph::test::to_device(*handle_, h_srcs);
      auto d_dsts    = cugraph::test::to_device(*handle_, h_dsts);
      auto d_weights = std::make_optional(cugraph::test::to_device(*handle_, h_weights));

      if (multi_gpu) {
        auto& comm           = handle_->get_comms();
        auto const comm_rank = comm.get_rank();
        auto const comm_size = comm.get_size();

        for (int k = 0; k < comm_size; k++) {
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

      cugraph::test::sort_and_remove_multi_edges(*handle_, d_srcs, d_dsts, d_weights);

      if (multi_gpu) {
        auto& comm           = handle_->get_comms();
        auto const comm_rank = comm.get_rank();
        auto const comm_size = comm.get_size();

        for (int k = 0; k < comm_size; k++) {
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
      std::tie(mg_graph, edge_weights, std::ignore, std::ignore, mg_renumber_map) =
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
                                                       std::move(d_weights),
                                                       std::nullopt,
                                                       std::nullopt,
                                                       cugraph::graph_properties_t{true, false},
                                                       true,
                                                       true);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Returned from create_graph_from_edgelist" << std::endl;
      // auto decision_graph_view = decision_graph.view();

      ///
    */

    bool test_weighted = true;
    bool renumber      = true;
    std::tie(mg_graph, edge_weights, mg_renumber_map) =
      cugraph::test::read_graph_from_csv_file<vertex_t, edge_t, weight_t, false, true>(
        *handle_, file_path, test_weighted, renumber);

    auto mg_graph_view    = mg_graph.view();
    auto edge_weight_view = (*edge_weights).view();

    using GraphViewType = decltype(mg_graph.view());

    if (GraphViewType::is_multi_gpu) {
      auto vertex_partitions_range_lasts =
        cugraph::test::to_device(*handle_, mg_graph_view.vertex_partition_range_lasts());

      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      auto& major_comm = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto const major_comm_rank = major_comm.get_rank();

      auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          std::cout << "Rank :" << comm_rank << std::endl;

          RAFT_CUDA_TRY(cudaDeviceSynchronize());

          std::cout << "(" << major_comm_size << minor_comm_size << ")" << std::endl;

          std::cout << "(major_rank, minor_rank): " << major_comm_rank << minor_comm.get_rank()
                    << std::endl;

          raft::print_device_vector("vertex_partitions_range_lasts:",
                                    vertex_partitions_range_lasts.data(),
                                    vertex_partitions_range_lasts.size(),
                                    std::cout);

          std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }
    }

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      /*
      std::vector<vertex_t> h_major_range_lasts(mg_graph_view.number_of_local_edge_partitions());
      for (size_t i = 0; i < mg_graph_view.number_of_local_edge_partitions(); ++i) {
        auto edge_partition =
          cugraph::edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
            mg_graph_view.local_edge_partition_view(i));
        h_major_range_lasts[i] = edge_partition.major_range_last();

        for (int k = 0; k < comm_size; k++) {
          comm.barrier();
          if (comm_rank == k) {
            RAFT_CUDA_TRY(cudaDeviceSynchronize());
            std::cout << "(rank = " << comm_rank << ", edge partittion idx = " << i
                      << ") : " << edge_partition.major_range_first() << " -- "
                      << edge_partition.major_range_last() << std::endl;
          }
          comm.barrier();
        }
      }

      rmm::device_uvector<vertex_t> d_major_range_lasts(h_major_range_lasts.size(),
                                                        handle_->get_stream());
      raft::update_device(d_major_range_lasts.data(),
                          h_major_range_lasts.data(),
                          h_major_range_lasts.size(),
                          handle_->get_stream());
      handle_->sync_stream();

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          std::cout << "Rank :" << comm_rank << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());

          raft::print_device_vector("d_major_range_lasts: ",
                                    d_major_range_lasts.data(),
                                    d_major_range_lasts.size(),
                                    std::cout);

          std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }
      */

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          std::cout << "Rank :" << comm_rank << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());

          std::cout << "edge_counts: ";
          std::copy(edge_weight_view.edge_counts().begin(),
                    edge_weight_view.edge_counts().end(),
                    std::ostream_iterator<edge_t>(std::cout, " "));
          std::cout << std::endl;
          edge_t num_edges = std::reduce(edge_weight_view.edge_counts().begin(),
                                         edge_weight_view.edge_counts().end());

          std::cout << std::endl << "num_edges: " << num_edges << std::endl;

          for (size_t i = 0; i < mg_graph_view.number_of_local_edge_partitions(); ++i) {
            std::cout << "partition " << i << " weights";
            raft::print_device_vector(":",
                                      edge_weight_view.value_firsts()[i],
                                      edge_weight_view.edge_counts()[i],
                                      std::cout);
          }

          std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          std::cout << "Rank :" << comm_rank << std::endl;
          RAFT_CUDA_TRY(cudaDeviceSynchronize());

          raft::print_device_vector("(*mg_renumber_map): ",
                                    (*mg_renumber_map).data(),
                                    (*mg_renumber_map).size(),
                                    std::cout);

          std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }

      for (size_t i = 0; i < mg_graph_view.number_of_local_edge_partitions(); ++i) {
        auto edge_partition =
          cugraph::edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
            mg_graph_view.local_edge_partition_view(i));

        auto edge_partition_weight_view =
          cugraph::detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>(
            edge_weight_view, i);

        auto edge_partition_weight_value_ptr = edge_partition_weight_view.value_first();

        for (int k = 0; k < comm_size; k++) {
          comm.barrier();
          if (comm_rank == k) {
            std::cout << "Rank :" << comm_rank << ", edge partittion idx = " << i << std::endl;
            RAFT_CUDA_TRY(cudaDeviceSynchronize());

            raft::print_device_vector("edge_weight_view: ",
                                      edge_weight_view.value_firsts()[i],
                                      edge_weight_view.edge_counts()[i],
                                      std::cout);

            std::cout << "------------------" << std::endl;
          }
          comm.barrier();
        }

        for (int k = 0; k < comm_size; k++) {
          comm.barrier();
          if (comm_rank == k) {
            std::cout << "Rank :" << comm_rank << std::endl;
            RAFT_CUDA_TRY(cudaDeviceSynchronize());

            std::cout << "rank = " << comm_rank << ", edge partittion idx = " << i << " : "
                      << edge_partition.major_range_first() << "--"
                      << edge_partition.major_range_last() << std::endl;

            thrust::for_each(
              handle_->get_thrust_policy(),
              thrust::make_counting_iterator(edge_partition.major_range_first()),
              thrust::make_counting_iterator(edge_partition.major_range_last()),
              [edge_partition, edge_partition_weight_value_ptr] __device__(vertex_t major) {
                printf("major -> %d\n", major);

                vertex_t major_idx{};
                auto major_hypersparse_first = edge_partition.major_hypersparse_first();
                if (major_hypersparse_first) {
                  printf("*major_hypersparse_first = %d\n",
                         static_cast<int>(*major_hypersparse_first));

                  if (major < *major_hypersparse_first) {
                    major_idx = edge_partition.major_offset_from_major_nocheck(major);
                  } else {
                    auto major_hypersparse_idx =
                      edge_partition.major_hypersparse_idx_from_major_nocheck(major);
                    if (!major_hypersparse_idx) {
                      printf("No major_hypersparse_idx\n");
                      return true;
                    }
                    major_idx =
                      edge_partition.major_offset_from_major_nocheck(*major_hypersparse_first) +
                      *major_hypersparse_idx;
                  }
                } else {
                  printf("No major_hypersparse_first\n");

                  major_idx = edge_partition.major_offset_from_major_nocheck(major);
                }

                printf("==> major_idx = %d\n", major_idx);

                vertex_t const* indices{nullptr};
                edge_t edge_offset{};
                edge_t local_degree{};
                thrust::tie(indices, edge_offset, local_degree) =
                  edge_partition.local_edges(major_idx, true);

                // std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view;

                auto number_of_edges = edge_partition.number_of_edges();

                printf(
                  "major = %d edge_offset = %d  local_degree= %d nr_edges_of_edge_partition=%d\n",
                  major,
                  edge_offset,
                  local_degree,
                  number_of_edges);
                for (edge_t nbr_idx = 0; nbr_idx < local_degree; nbr_idx++) {
                  // printf("%d ", indices[nbr_idx]);
                  printf("%d %d %.2f \n",
                         major,
                         indices[nbr_idx],
                         *(edge_partition_weight_value_ptr + edge_offset + nbr_idx));
                }
                printf("\n");
              });

            std::cout << "------------------" << std::endl;
          }
          comm.barrier();
        }

      }  // end of loop over edge partitions
    }

    /*

    if (multi_gpu) {
      auto& comm           = handle_->get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          std::cout << "Rank :" << comm_rank << std::endl;


          std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }
    }

    */

    // #if 0
    // 2. run MG per_v_pair_transform_dst_nbr_intersection primitive

    ASSERT_TRUE(
      mg_graph_view.number_of_vertices() >
      vertex_t{0});  // the code below to generate vertex pairs is invalid for an empty graph.

    auto mg_vertex_pair_buffer =
      cugraph::allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prims_usecase.num_vertex_pairs / comm_size +
          (static_cast<size_t>(comm_rank) < prims_usecase.num_vertex_pairs % comm_size ? 1 : 0),
        handle_->get_stream());

    std::cout << "Rank: " << comm_rank
              << " prims_usecase.num_vertex_pairs:" << prims_usecase.num_vertex_pairs << std::endl;

    std::cout << "Rank: " << comm_rank << " cugraph::size_dataframe_buffer(mg_vertex_pair_buffer): "
              << cugraph::size_dataframe_buffer(mg_vertex_pair_buffer) << std::endl;

    thrust::tabulate(
      handle_->get_thrust_policy(),
      cugraph::get_dataframe_buffer_begin(mg_vertex_pair_buffer),
      cugraph::get_dataframe_buffer_end(mg_vertex_pair_buffer),
      [comm_rank, num_vertices = mg_graph_view.number_of_vertices()] __device__(size_t i) {
        cuco::detail::MurmurHash3_32<size_t>
          hash_func{};  // use hash_func to generate arbitrary vertex pairs
        auto v0 = 2;    // static_cast<vertex_t>(hash_func(i + comm_rank) % num_vertices);
        auto v1 =
          3;  // static_cast<vertex_t>(hash_func(i + num_vertices + comm_rank) % num_vertices);
        printf("comm_rank=%d v0= %d, v1=%d\n",
               static_cast<int>(comm_rank),
               static_cast<int>(v0),
               static_cast<int>(v1));
        return thrust::make_tuple(v0, v1);
      });

    auto h_vertex_partition_range_lasts = mg_graph_view.vertex_partition_range_lasts();
    std::tie(std::get<0>(mg_vertex_pair_buffer),
             std::get<1>(mg_vertex_pair_buffer),
             std::ignore,
             std::ignore,
             std::ignore) =
      cugraph::detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
        vertex_t,
        edge_t,
        weight_t,
        int32_t>(*handle_,
                 std::move(std::get<0>(mg_vertex_pair_buffer)),
                 std::move(std::get<1>(mg_vertex_pair_buffer)),
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 h_vertex_partition_range_lasts);

    for (int k = 0; k < comm_size; k++) {
      auto& comm = handle_->get_comms();

      comm.barrier();
      if (comm_rank == k) {
        std::cout << "Rank :" << comm_rank << std::endl;
        RAFT_CUDA_TRY(cudaDeviceSynchronize());

        raft::print_device_vector("std::get<0>(mg_vertex_pair_buffer)",
                                  std::get<0>(mg_vertex_pair_buffer).data(),
                                  std::get<0>(mg_vertex_pair_buffer).size(),
                                  std::cout);

        raft::print_device_vector("std::get<1>(mg_vertex_pair_buffer)",
                                  std::get<1>(mg_vertex_pair_buffer).data(),
                                  std::get<1>(mg_vertex_pair_buffer).size(),
                                  std::cout);

        std::cout << "------------------" << std::endl;
      }
      comm.barrier();
    }

    auto mg_result_buffer = cugraph::allocate_dataframe_buffer<thrust::tuple<edge_t, edge_t>>(
      cugraph::size_dataframe_buffer(mg_vertex_pair_buffer), handle_->get_stream());
    auto mg_out_degrees = mg_graph_view.compute_out_degrees(*handle_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG per_v_pair_transform_dst_nbr_intersection");
    }

    cugraph::per_v_pair_transform_dst_nbr_intersection(
      *handle_,
      mg_graph_view,
      edge_weight_view,
      cugraph::get_dataframe_buffer_begin(mg_vertex_pair_buffer),
      cugraph::get_dataframe_buffer_end(mg_vertex_pair_buffer),
      mg_out_degrees.begin(),
      intersection_op_t<vertex_t, edge_t, weight_t>{},
      cugraph::get_dataframe_buffer_begin(mg_result_buffer));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    for (int k = 0; k < comm_size; k++) {
      auto& comm = handle_->get_comms();

      comm.barrier();
      if (comm_rank == k) {
        std::cout << "Rank :" << comm_rank << std::endl;
        RAFT_CUDA_TRY(cudaDeviceSynchronize());

        raft::print_device_vector("std::get<0>(mg_result_buffer)",
                                  std::get<0>(mg_result_buffer).data(),
                                  std::get<0>(mg_result_buffer).size(),
                                  std::cout);

        raft::print_device_vector("std::get<1>(mg_result_buffer)",
                                  std::get<1>(mg_result_buffer).data(),
                                  std::get<1>(mg_result_buffer).size(),
                                  std::cout);

        std::cout << "------------------" << std::endl;
      }
      comm.barrier();
    }

    // 3. validate MG results

    if (prims_usecase.check_correctness) {
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        std::get<0>(mg_vertex_pair_buffer).data(),
        cugraph::size_dataframe_buffer(mg_vertex_pair_buffer),
        (*mg_renumber_map).data(),
        h_vertex_partition_range_lasts);
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        std::get<1>(mg_vertex_pair_buffer).data(),
        cugraph::size_dataframe_buffer(mg_vertex_pair_buffer),
        (*mg_renumber_map).data(),
        h_vertex_partition_range_lasts);

      auto mg_aggregate_vertex_pair_buffer =
        cugraph::allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          0, handle_->get_stream());
      std::get<0>(mg_aggregate_vertex_pair_buffer) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<0>(mg_vertex_pair_buffer).data(),
                                      std::get<0>(mg_vertex_pair_buffer).size());
      std::get<1>(mg_aggregate_vertex_pair_buffer) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<1>(mg_vertex_pair_buffer).data(),
                                      std::get<1>(mg_vertex_pair_buffer).size());

      auto mg_aggregate_result_buffer =
        cugraph::allocate_dataframe_buffer<thrust::tuple<edge_t, edge_t>>(0, handle_->get_stream());
      std::get<0>(mg_aggregate_result_buffer) = cugraph::test::device_gatherv(
        *handle_, std::get<0>(mg_result_buffer).data(), std::get<0>(mg_result_buffer).size());
      std::get<1>(mg_aggregate_result_buffer) = cugraph::test::device_gatherv(
        *handle_, std::get<1>(mg_result_buffer).data(), std::get<1>(mg_result_buffer).size());

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                              (*mg_renumber_map).size()),
        false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();

        auto sg_result_buffer = cugraph::allocate_dataframe_buffer<thrust::tuple<edge_t, edge_t>>(
          cugraph::size_dataframe_buffer(mg_aggregate_vertex_pair_buffer), handle_->get_stream());
        auto sg_out_degrees = sg_graph_view.compute_out_degrees(*handle_);

        // cugraph::per_v_pair_transform_dst_nbr_intersection(
        //   *handle_,
        //   sg_graph_view,
        //   edge_weight_view,
        //   cugraph::get_dataframe_buffer_begin(
        //     mg_aggregate_vertex_pair_buffer /* now unrenumbered */),
        //   cugraph::get_dataframe_buffer_end(mg_aggregate_vertex_pair_buffer /* now unrenumbered
        //   */), sg_out_degrees.begin(),  intersection_op_t<vertex_t, edge_t, weight_t>{},
        //   cugraph::get_dataframe_buffer_begin(sg_result_buffer));

        // bool valid = thrust::equal(handle_->get_thrust_policy(),
        //                            cugraph::get_dataframe_buffer_begin(mg_aggregate_result_buffer),
        //                            cugraph::get_dataframe_buffer_end(mg_aggregate_result_buffer),
        //                            cugraph::get_dataframe_buffer_begin(sg_result_buffer));

        // ASSERT_TRUE(valid);
      }
    }
    // #endif
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGPerVPairTransformDstNbrIntersection<input_usecase_t>::handle_ = nullptr;

using Tests_MGPerVPairTransformDstNbrIntersection_File =
  Tests_MGPerVPairTransformDstNbrIntersection<cugraph::test::File_Usecase>;
using Tests_MGPerVPairTransformDstNbrIntersection_Rmat =
  Tests_MGPerVPairTransformDstNbrIntersection<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_File, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(std::get<0>(param),
                                                                       std::get<1>(param));
}
/*
TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int64FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt64Int64FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}
*/

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVPairTransformDstNbrIntersection_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{size_t{1}, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

// INSTANTIATE_TEST_SUITE_P(rmat_small_test,
//                          Tests_MGPerVPairTransformDstNbrIntersection_Rmat,
//                          ::testing::Combine(::testing::Values(Prims_Usecase{size_t{1024},
//                          true}),
//                                             ::testing::Values(cugraph::test::Rmat_Usecase(
//                                               10, 16, 0.57, 0.19, 0.19, 0, false, false))));

// INSTANTIATE_TEST_SUITE_P(
//   rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
//                           --gtest_filter to select only the rmat_benchmark_test with a specific
//                           vertex & edge type combination) by command line arguments and do not
//                           include more than one Rmat_Usecase that differ only in scale or edge
//                           factor (to avoid running same benchmarks more than once) */
//   Tests_MGPerVPairTransformDstNbrIntersection_Rmat,
//   ::testing::Combine(
//     ::testing::Values(Prims_Usecase{size_t{1024 * 1024}, false}),
//     ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false,
//     false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
