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

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/mtmg/resource_manager.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <nccl.h>

#include <vector>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <detail/graph_partition_utils.cuh>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

class Tests_MTMG : public ::testing::Test {};

template <typename vertex_t>
struct check_edge_t {
  vertex_t const* sorted_valid_major_range_first{nullptr};
  vertex_t const* sorted_valid_major_range_last{nullptr};
  vertex_t const* sorted_valid_minor_range_first{nullptr};
  vertex_t const* sorted_valid_minor_range_last{nullptr};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> const& e) const
  {
    return !thrust::binary_search(thrust::seq,
                                  sorted_valid_major_range_first,
                                  sorted_valid_major_range_last,
                                  thrust::get<0>(e)) ||
           !thrust::binary_search(thrust::seq,
                                  sorted_valid_minor_range_first,
                                  sorted_valid_minor_range_last,
                                  thrust::get<1>(e));
  }
};

template <typename vertex_t, bool multi_gpu>
void expensive_check_edgelist(raft::handle_t const& handle,
                              std::optional<rmm::device_uvector<vertex_t>> const& vertices,
                              rmm::device_uvector<vertex_t> const& edgelist_majors,
                              rmm::device_uvector<vertex_t> const& edgelist_minors,
                              bool renumber)
{
  if (vertices) {
    rmm::device_uvector<vertex_t> sorted_vertices((*vertices).size(), handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), (*vertices).begin(), (*vertices).end(), sorted_vertices.begin());
    thrust::sort(handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end());
    CUGRAPH_EXPECTS(static_cast<size_t>(thrust::distance(sorted_vertices.begin(),
                                                         thrust::unique(handle.get_thrust_policy(),
                                                                        sorted_vertices.begin(),
                                                                        sorted_vertices.end()))) ==
                      sorted_vertices.size(),
                    "Invalid input argument: vertices should not have duplicates.");
    if (!renumber) {
      CUGRAPH_EXPECTS(static_cast<size_t>(thrust::count_if(
                        handle.get_thrust_policy(),
                        sorted_vertices.begin(),
                        sorted_vertices.end(),
                        cugraph::detail::check_out_of_range_t<vertex_t>{
                          vertex_t{0}, std::numeric_limits<vertex_t>::max()})) == size_t{0},
                      "Invalid input argument: vertex IDs should be in [0, "
                      "std::numeric_limits<vertex_t>::max()) if renumber is false.");
      assert(!multi_gpu);  // renumbering is required in multi-GPU
      rmm::device_uvector<vertex_t> sequences(sorted_vertices.size(), handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(), sequences.begin(), sequences.end(), vertex_t{0});
      CUGRAPH_EXPECTS(thrust::equal(handle.get_thrust_policy(),
                                    sorted_vertices.begin(),
                                    sorted_vertices.end(),
                                    sequences.begin()),
                      "Invalid input argument: vertex IDs should be consecutive integers starting "
                      "from 0 if renumber is false.");
    }
  } else if (!renumber) {
    CUGRAPH_EXPECTS(static_cast<size_t>(thrust::count_if(
                      handle.get_thrust_policy(),
                      edgelist_majors.begin(),
                      edgelist_majors.end(),
                      cugraph::detail::check_out_of_range_t<vertex_t>{
                        vertex_t{0}, std::numeric_limits<vertex_t>::max()})) == size_t{0},
                    "Invalid input argument: vertex IDs should be in [0, "
                    "std::numeric_limits<vertex_t>::max()) if renumber is false.");
    CUGRAPH_EXPECTS(static_cast<size_t>(thrust::count_if(
                      handle.get_thrust_policy(),
                      edgelist_minors.begin(),
                      edgelist_minors.end(),
                      cugraph::detail::check_out_of_range_t<vertex_t>{
                        vertex_t{0}, std::numeric_limits<vertex_t>::max()})) == size_t{0},
                    "Invalid input argument: vertex IDs should be in [0, "
                    "std::numeric_limits<vertex_t>::max()) if renumber is false.");
  }

  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    if (vertices) {
      auto num_unique_vertices = cugraph::host_scalar_allreduce(
        comm, (*vertices).size(), raft::comms::op_t::SUM, handle.get_stream());
      CUGRAPH_EXPECTS(
        num_unique_vertices < static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
        "Invalid input arguments: # unique vertex IDs should be smaller than "
        "std::numeric_limits<vertex_t>::Max().");

      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         (*vertices).begin(),
                         (*vertices).end(),
                         [comm_rank,
                          key_func =
                            cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                              comm_size, major_comm_size, minor_comm_size}] __device__(auto val) {
                           return key_func(val) != comm_rank;
                         }) == 0,
        "Invalid input argument: vertices should be pre-shuffled.");
    }

    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
    CUGRAPH_EXPECTS(
      thrust::count_if(handle.get_thrust_policy(),
                       edge_first,
                       edge_first + edgelist_majors.size(),
                       [comm_rank,
                        gpu_id_key_func =
                          cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
                            comm_size, major_comm_size, minor_comm_size}] __device__(auto e) {
                         return (gpu_id_key_func(e) != comm_rank);
                       }) == 0,
      "Invalid input argument: edgelist_majors & edgelist_minors should be pre-shuffled.");

    if (vertices) {
      rmm::device_uvector<vertex_t> sorted_majors(0, handle.get_stream());
      {
        auto recvcounts =
          cugraph::host_scalar_allgather(minor_comm, (*vertices).size(), handle.get_stream());
        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
        sorted_majors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
        cugraph::device_allgatherv(minor_comm,
                                   (*vertices).data(),
                                   sorted_majors.data(),
                                   recvcounts,
                                   displacements,
                                   handle.get_stream());
        thrust::sort(handle.get_thrust_policy(), sorted_majors.begin(), sorted_majors.end());
      }

      rmm::device_uvector<vertex_t> sorted_minors(0, handle.get_stream());
      {
        auto recvcounts =
          cugraph::host_scalar_allgather(major_comm, (*vertices).size(), handle.get_stream());
        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
        sorted_minors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
        cugraph::device_allgatherv(major_comm,
                                   (*vertices).data(),
                                   sorted_minors.data(),
                                   recvcounts,
                                   displacements,
                                   handle.get_stream());
        thrust::sort(handle.get_thrust_policy(), sorted_minors.begin(), sorted_minors.end());
      }

      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         edge_first,
                         edge_first + edgelist_majors.size(),
                         check_edge_t<vertex_t>{sorted_majors.data(),
                                                sorted_majors.data() + sorted_majors.size(),
                                                sorted_minors.data(),
                                                sorted_minors.data() + sorted_minors.size()}) == 0,
        "Invalid input argument: edgelist_majors and/or edgelist_minors have invalid vertex "
        "ID(s).");
    }
  } else {
    if (vertices) {
      rmm::device_uvector<vertex_t> sorted_vertices((*vertices).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*vertices).begin(),
                   (*vertices).end(),
                   sorted_vertices.begin());
      thrust::sort(handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end());
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         edge_first,
                         edge_first + edgelist_majors.size(),
                         check_edge_t<vertex_t>{sorted_vertices.data(),
                                                sorted_vertices.data() + sorted_vertices.size(),
                                                sorted_vertices.data(),
                                                sorted_vertices.data() + sorted_vertices.size()}) ==
          0,
        "Invalid input argument: edgelist_majors and/or edgelist_minors have invalid vertex "
        "ID(s).");
    }
  }
}

template <typename vertex_t, bool store_transposed, bool multi_gpu>
bool check_symmetric(raft::handle_t const& handle,
                     raft::device_span<vertex_t const> edgelist_srcs,
                     raft::device_span<vertex_t const> edgelist_dsts)
{
  rmm::device_uvector<vertex_t> org_srcs(edgelist_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> org_dsts(edgelist_dsts.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), edgelist_srcs.begin(), edgelist_srcs.end(), org_srcs.begin());
  thrust::copy(
    handle.get_thrust_policy(), edgelist_dsts.begin(), edgelist_dsts.end(), org_dsts.begin());

  rmm::device_uvector<vertex_t> symmetrized_srcs(org_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> symmetrized_dsts(org_dsts.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), org_srcs.begin(), org_srcs.end(), symmetrized_srcs.begin());
  thrust::copy(
    handle.get_thrust_policy(), org_dsts.begin(), org_dsts.end(), symmetrized_dsts.begin());
  std::tie(symmetrized_srcs, symmetrized_dsts, std::ignore) =
    cugraph::symmetrize_edgelist<vertex_t, float /* dummy */, store_transposed, multi_gpu>(
      handle, std::move(symmetrized_srcs), std::move(symmetrized_dsts), std::nullopt, true);

  if (org_srcs.size() != symmetrized_srcs.size()) { return false; }

  auto org_edge_first =
    thrust::make_zip_iterator(thrust::make_tuple(org_srcs.begin(), org_dsts.begin()));
  thrust::sort(handle.get_thrust_policy(), org_edge_first, org_edge_first + org_srcs.size());
  auto symmetrized_edge_first = thrust::make_zip_iterator(
    thrust::make_tuple(symmetrized_srcs.begin(), symmetrized_dsts.begin()));
  thrust::sort(handle.get_thrust_policy(),
               symmetrized_edge_first,
               symmetrized_edge_first + symmetrized_srcs.size());

  return thrust::equal(handle.get_thrust_policy(),
                       org_edge_first,
                       org_edge_first + org_srcs.size(),
                       symmetrized_edge_first);
}

void run_current_test()
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  int num_gpus{2};

  cugraph::mtmg::resource_manager_t resource_manager;

  resource_manager.register_local_gpu(0, 0);
  resource_manager.register_local_gpu(1, 1);

  ncclUniqueId instance_manager_id;
  ncclGetUniqueId(&instance_manager_id);

  std::cout << "create instance_manager" << std::endl;

  auto instance_manager = resource_manager.create_instance_manager(
    resource_manager.registered_ranks(), instance_manager_id);

  std::cout << "prepare to create edges" << std::endl;

  std::vector<std::thread> running_threads;

  for (int i = 0; i < num_gpus; ++i) {
    running_threads.emplace_back([&instance_manager]() {
      auto thread_handle = instance_manager->get_handle();

      if (thread_handle.get_thread_rank() > 0) return;

      rmm::device_uvector<vertex_t> d_edgelist_majors(0, thread_handle.get_stream());
      rmm::device_uvector<vertex_t> d_edgelist_minors(0, thread_handle.get_stream());

      if (thread_handle.get_rank() == 0) {
        std::vector<vertex_t> edgelist_majors{
          {0,  6,  15, 3,  12, 7,  29, 33, 0,  8,  18, 26, 5,  17, 17, 0,  23, 7,
           21, 21, 10, 25, 0,  1,  14, 24, 1,  10, 2,  16, 32, 33, 31, 33, 0,  3,
           8,  23, 30, 4,  13, 13, 33, 4,  23, 31, 6,  19, 19, 6,  33, 0,  5,  20,
           8,  31, 30, 16, 32, 27, 32, 1,  2,  5,  22, 29, 2,  11, 3,  33, 31, 32}};
        std::vector<vertex_t> edgelist_minors{
          {1,  16, 32, 0,  0,  1,  26, 30, 12, 32, 32, 29, 0,  0,  1,  5,  32, 0,
           0,  1,  5,  24, 21, 21, 32, 25, 0,  0,  1,  6,  15, 22, 24, 29, 11, 12,
           30, 25, 32, 0,  0,  1,  26, 6,  29, 32, 0,  0,  1,  5,  32, 6,  6,  32,
           0,  0,  1,  5,  22, 24, 29, 30, 32, 16, 32, 32, 0,  0,  1,  15, 25, 30}};

        std::cout << "allocate space, majors size = " << edgelist_majors.size() << std::endl;
        d_edgelist_majors.resize(edgelist_majors.size(), thread_handle.get_stream());
        d_edgelist_minors.resize(edgelist_minors.size(), thread_handle.get_stream());

        raft::update_device(d_edgelist_majors.data(),
                            edgelist_majors.data(),
                            edgelist_majors.size(),
                            thread_handle.get_stream());

        raft::update_device(d_edgelist_minors.data(),
                            edgelist_minors.data(),
                            edgelist_minors.size(),
                            thread_handle.get_stream());
      } else {
        std::vector<vertex_t> edgelist_majors = {
          {7,  3,  28, 20, 5,  2,  0,  33, 33, 32, 28, 31, 27, 19, 9,  4,  2,  1,  0,  33, 33,
           32, 33, 33, 10, 13, 30, 23, 3,  2,  1,  0,  27, 32, 32, 13, 8,  29, 24, 22, 3,  2,
           1,  0,  25, 30, 12, 7,  24, 14, 0,  0,  33, 32, 32, 28, 23, 13, 2,  1,  0,  31, 33,
           32, 33, 27, 26, 18, 8,  2,  1,  0,  0,  32, 29, 33, 33, 6,  9,  25, 15, 2,  1,  0}};
        std::vector<vertex_t> edgelist_minors = {
          {3,  2,  33, 33, 10, 28, 7,  28, 20, 14, 2,  33, 33, 33, 33, 10, 13, 17, 17, 31, 27,
           23, 19, 9,  4,  2,  33, 27, 13, 8,  7,  3,  23, 18, 8,  3,  2,  33, 31, 33, 7,  3,
           2,  10, 23, 8,  3,  2,  27, 33, 31, 8,  14, 2,  33, 31, 33, 33, 27, 19, 19, 28, 23,
           20, 13, 2,  33, 33, 33, 9,  13, 13, 4,  31, 23, 18, 8,  4,  2,  31, 33, 7,  3,  2}};

        std::cout << "allocate space, majors size = " << edgelist_majors.size() << std::endl;
        d_edgelist_majors.resize(edgelist_majors.size(), thread_handle.get_stream());
        d_edgelist_minors.resize(edgelist_minors.size(), thread_handle.get_stream());

        raft::update_device(d_edgelist_majors.data(),
                            edgelist_majors.data(),
                            edgelist_majors.size(),
                            thread_handle.get_stream());

        raft::update_device(d_edgelist_minors.data(),
                            edgelist_minors.data(),
                            edgelist_minors.size(),
                            thread_handle.get_stream());
      }

      thread_handle.raft_handle().sync_stream();

      raft::print_device_vector(
        " edgelist_majors", d_edgelist_majors.data(), d_edgelist_majors.size(), std::cout);
      raft::print_device_vector(
        " edgelist_minors", d_edgelist_minors.data(), d_edgelist_minors.size(), std::cout);

      auto& comm           = thread_handle.raft_handle().get_comms();
      auto const comm_size = comm.get_size();
      auto const comm_rank = comm.get_rank();
      auto& major_comm =
        thread_handle.raft_handle().get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm =
        thread_handle.raft_handle().get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      std::cout << "in expensive_check_edgelist, multi_gpu block, rank = " << comm_rank
                << ", size = " << comm_size << std::endl;

      int xxx;
      RAFT_CUDA_TRY(cudaGetDevice(&xxx));
      std::cout << "   device = " << xxx << std::endl;
      std::cout << "   stream = " << thread_handle.raft_handle().get_stream() << std::endl;

      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(d_edgelist_majors.begin(), d_edgelist_minors.begin()));
      CUGRAPH_EXPECTS(
        thrust::count_if(thread_handle.raft_handle().get_thrust_policy(),
                         edge_first,
                         edge_first + d_edgelist_majors.size(),
                         [comm_rank,
                          gpu_id_key_func =
                            cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
                              comm_size, major_comm_size, minor_comm_size}] __device__(auto e) {
                           printf("(%d,%d) on rank %d, expected %d\n",
                                  (int)thrust::get<0>(e),
                                  (int)thrust::get<1>(e),
                                  comm_rank,
                                  gpu_id_key_func(e));

                           return (gpu_id_key_func(e) != comm_rank);
                         }) == 0,
        "Invalid input argument: edgelist_majors & edgelist_minors should be pre-shuffled.");

      sleep(10);

      std::cout << "calling cugraph::create_graph_from_edgelist, rank = "
                << thread_handle.get_rank() << std::endl;
#if 1
      auto [local_graph, local_edge_weights, local_edge_ids, local_edge_types, local_renumber_map] =
        cugraph::
          create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, true, true>(
            thread_handle.raft_handle(),
            std::nullopt,
            std::move(d_edgelist_minors),
            std::move(d_edgelist_majors),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cugraph::graph_properties_t{false, true},
            true,
            true);
#else
      CUGRAPH_EXPECTS(d_edgelist_majors.size() == d_edgelist_minors.size(),
                      "Invalid input arguments: edgelist_srcs.size() != edgelist_dsts.size().");

      expensive_check_edgelist<vertex_t, true>(
        thread_handle.raft_handle(), std::nullopt, d_edgelist_majors, d_edgelist_minors, true);

      // 1. groupby edges to their target local adjacency matrix partition (and further groupby
      // within the local partition by applying the compute_gpu_id_from_vertex_t to minor vertex
      // IDs).

      std::optional<rmm::device_uvector<weight_t>> aaa{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> bbb{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> ccc{std::nullopt};

      auto d_edge_counts = cugraph::detail::groupby_and_count_edgelist_by_local_partition_id(
        thread_handle.raft_handle(), d_edgelist_majors, d_edgelist_minors, aaa, bbb, ccc, true);

      std::vector<size_t> h_edge_counts(d_edge_counts.size());
      raft::update_host(h_edge_counts.data(),
                        d_edge_counts.data(),
                        d_edge_counts.size(),
                        thread_handle.get_stream());
      thread_handle.raft_handle().sync_stream();

      std::vector<edge_t> edgelist_edge_counts(minor_comm_size, edge_t{0});
      auto edgelist_intra_partition_segment_offsets =
        std::make_optional<std::vector<std::vector<edge_t>>>(
          minor_comm_size, std::vector<edge_t>(major_comm_size + 1, edge_t{0}));

      for (int i = 0; i < minor_comm_size; ++i) {
        edgelist_edge_counts[i] = std::accumulate(h_edge_counts.begin() + major_comm_size * i,
                                                  h_edge_counts.begin() + major_comm_size * (i + 1),
                                                  edge_t{0});
        std::partial_sum(h_edge_counts.begin() + major_comm_size * i,
                         h_edge_counts.begin() + major_comm_size * (i + 1),
                         (*edgelist_intra_partition_segment_offsets)[i].begin() + 1);
      }
      std::vector<edge_t> edgelist_displacements(minor_comm_size, edge_t{0});
      std::partial_sum(edgelist_edge_counts.begin(),
                       edgelist_edge_counts.end() - 1,
                       edgelist_displacements.begin() + 1);

      std::cout
        << "in threaded_test... after groupby_and_count_edgelist_by_local_partition_id, rank = "
        << thread_handle.get_rank() << std::endl;
      raft::print_host_vector("  edgelist_displacements",
                              edgelist_displacements.data(),
                              edgelist_displacements.size(),
                              std::cout);
#endif

      std::cout << "after cugraph::create_graph_from_edgelist call, rank = "
                << thread_handle.get_rank() << std::endl;
    });
  }

  // Wait for CPU threads to complete
  std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
  running_threads.resize(0);
  instance_manager->reset_threads();
}

TEST_F(Tests_MTMG, CheckInt32Int32FloatFloat) { run_current_test(); }

CUGRAPH_TEST_PROGRAM_MAIN()
