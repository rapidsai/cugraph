/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/count_if_e.cuh"
#include "prims/extract_transform_e.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/kv_store.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/collect_comm.cuh"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/unique.h>

#include <gtest/gtest.h>

#include <random>

struct Graph500_BFS_Usecase {
  bool unrenumber_predecessors{true};
  bool validate{true};
};

template <typename input_usecase_t>
class Tests_GRAPH500_MGBFS
  : public ::testing::TestWithParam<std::tuple<Graph500_BFS_Usecase, input_usecase_t>> {
 public:
  Tests_GRAPH500_MGBFS() {}

  static void SetUpTestCase()
  {
#if 1
    auto ret = setenv("NCCL_DEBUG", "WARN", 1);
    if (ret != 0) std::cout << "setenv(\"NCCL_DEBUG\", \"TRACE\", 1) returned " << ret << std::endl;
#endif
#if 0  // workstation
       // nothing
#else
#if 0  // for CW
    ret = setenv("NCCL_NET", "IB", 1);
    if (ret != 0) std::cout << "setenv(\"NCCL_NET\", \"IB\", 1) returned " << ret << std::endl;
    ret = setenv("NCCL_SOCKET_IFNAME", "enp90s0f0np0", 1);
    if (ret != 0)
      std::cout << "setenv(\"NCCL_SOCKET_IFNAME\", \"enp90s0f0np0\", 1) returned " << ret
                << std::endl;
#else  // for EOS
    ret = setenv("NCCL_COLLNET_ENABLE", "0", 1);
    if (ret != 0)
      std::cout << "setenv(\"NCCL_COLLNET_ENABLE\", \"0\", 1) returned " << ret << std::endl;
    ret = setenv("NCCL_SHARP_DISABLE", "1", 1);
    if (ret != 0)
      std::cout << "setenv(\"NCCL_SHARP_DISABLE\", \"1\", 1) returned " << ret << std::endl;
    ret = setenv("NCCL_SHARP_GROUP_SIZE_THRESH", "8", 1);
    if (ret != 0)
      std::cout << "setenv(\"NCCL_SHARP_GROUP_SIZE_THRESH\", \"8\", 1) returned " << ret
                << std::endl;
#endif
#endif
    size_t pool_size =
      16;  // note that CUDA_DEVICE_MAX_CONNECTIONS (default: 8) should be set to a value larger
           // than pool_size to avoid false dependency among different streams
    handle_ = cugraph::test::initialize_mg_handle(pool_size);
  }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(Graph500_BFS_Usecase const& bfs_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t    = float;
    using edge_type_t = int32_t;  // dummy

    bool constexpr store_transposed = false;
    bool constexpr multi_gpu        = true;
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = false;
    bool constexpr shuffle = false;  // Graph 500 requirement (edges can't be pre-shuffled, edges
                                     // should be shuffled in Kernel 1)
    size_t constexpr num_warmup_starting_vertices =
      1;  // to enforce all CUDA & NCCL initializations
    size_t constexpr num_timed_starting_vertices = 64;  // Graph 500 requirement

    HighResTimer hr_timer{};

    auto& comm           = handle_->get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();
    auto& major_comm     = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    constexpr auto invalid_distance = std::numeric_limits<vertex_t>::max();
    constexpr auto invalid_vertex   = cugraph::invalid_vertex_id<vertex_t>::value;

    // 1. force NCCL P2P initialization

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.start("NCCL P2P buffer initialization");
    }

    cugraph::test::enforce_p2p_initialization(comm, handle_->get_stream());
    cugraph::test::enforce_p2p_initialization(major_comm, handle_->get_stream());
    cugraph::test::enforce_p2p_initialization(minor_comm, handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 2. create an edge list

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.start("MG Construct edge list");
    }

    std::vector<rmm::device_uvector<vertex_t>> src_chunks{};
    std::vector<rmm::device_uvector<vertex_t>> dst_chunks{};
    std::tie(src_chunks, dst_chunks, std::ignore, std::ignore, std::ignore) =
      input_usecase.template construct_edgelist<vertex_t, weight_t>(
        *handle_, test_weighted, store_transposed, multi_gpu, shuffle);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. create an MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.start("MG Construct graph (Kernel 1)");
    }

    for (size_t i = 0; i < src_chunks.size(); ++i) {  // shuffle edges
      std::tie(src_chunks[i], dst_chunks[i], std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::shuffle_external_edges<vertex_t, edge_t, weight_t, edge_type_t>(
          *handle_,
          std::move(src_chunks[i]),
          std::move(dst_chunks[i]),
          std::nullopt,
          std::nullopt,
          std::nullopt);
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, std::ignore, std::ignore, mg_renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t,
                                          edge_t,
                                          weight_t,
                                          edge_t,
                                          int32_t,
                                          store_transposed,
                                          multi_gpu>(
        *handle_,
        std::nullopt,
        std::move(src_chunks),
        std::move(dst_chunks),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{input_usecase.undirected() /* symmetric */,
                                    true /* multi-graph */},
        renumber);

    auto mg_graph_view = mg_graph.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 4. randomly select starting vertices

    rmm::device_uvector<vertex_t> d_starting_vertices(0, handle_->get_stream());
    {
      raft::random::RngState rng_state(comm_size + comm_rank /* seed */);
      auto tot_vertices = num_warmup_starting_vertices + num_timed_starting_vertices;
      auto out_degrees  = mg_graph_view.compute_out_degrees(*handle_);

      size_t num_generated{0};
      while (num_generated < tot_vertices) {
        auto candidates =
          cugraph::select_random_vertices<vertex_t, edge_t, store_transposed, multi_gpu>(
            *handle_,
            mg_graph_view,
            std::nullopt,
            rng_state,
            tot_vertices - num_generated,
            true /* with_replacement */,
            false /* sort_vertices */);
        candidates.resize(
          thrust::distance(
            candidates.begin(),
            thrust::remove_if(handle_->get_thrust_policy(),
                              candidates.begin(),
                              candidates.end(),
                              [v_first     = mg_graph_view.local_vertex_partition_range_first(),
                               out_degrees = raft::device_span<edge_t const>(
                                 out_degrees.data(), out_degrees.size())] __device__(auto v) {
                                auto out_degree = out_degrees[v - v_first];
                                return out_degree == 0;  // remove isolated vertices
                              })),
          handle_->get_stream());
        auto num_valids = cugraph::host_scalar_allreduce(
          comm, candidates.size(), raft::comms::op_t::SUM, handle_->get_stream());
        num_generated += num_valids;
        auto old_size = d_starting_vertices.size();
        d_starting_vertices.resize(old_size + candidates.size(), handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     candidates.begin(),
                     candidates.end(),
                     d_starting_vertices.begin() + old_size);
      }
    }
    auto starting_vertex_counts =
      cugraph::host_scalar_allgather(comm, d_starting_vertices.size(), handle_->get_stream());
    auto starting_vertex_offsets = std::vector<size_t>(starting_vertex_counts.size() + 1);
    starting_vertex_offsets[0]   = 0;
    std::inclusive_scan(starting_vertex_counts.begin(),
                        starting_vertex_counts.end(),
                        starting_vertex_offsets.begin() + 1);

    // 5. run MG BFS

    // FIXME: Graph500 doesn't require computing distances.
    rmm::device_uvector<vertex_t> d_mg_distances(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle_->get_stream());
    rmm::device_uvector<vertex_t> d_mg_predecessors(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    double total_elapsed{0.0};
    for (size_t i = 0; i < (num_warmup_starting_vertices + num_timed_starting_vertices); ++i) {
      auto starting_vertex_comm_rank = static_cast<int>(std::distance(
        starting_vertex_offsets.begin() + 1,
        std::upper_bound(starting_vertex_offsets.begin() + 1, starting_vertex_offsets.end(), i)));
      raft::device_span<vertex_t const> d_starting_vertex(static_cast<vertex_t const*>(nullptr),
                                                          size_t{0});
      if (comm_rank == starting_vertex_comm_rank) {
        d_starting_vertex = raft::device_span<vertex_t const>(
          d_starting_vertices.data() + (i - starting_vertex_offsets[comm_rank]), 1);
      }
      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.start("MG BFS (Kernel 2)");
      }

      cugraph::bfs(*handle_,
                   mg_graph_view,
                   d_mg_distances.data(),
                   d_mg_predecessors.data(),
                   d_starting_vertex.data(),
                   d_starting_vertex.size(),
                   mg_graph_view.is_symmetric() ? true : false,
                   std::numeric_limits<vertex_t>::max());

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        auto elapsed = hr_timer.stop();
        if (i >= num_warmup_starting_vertices) { total_elapsed += elapsed; }
        hr_timer.display_and_clear(std::cout);
      }

      /* compute the number of visisted edges */

      {
        rmm::device_uvector<bool> flags(mg_graph_view.local_vertex_partition_range_size(),
                                        handle_->get_stream());
        thrust::transform(handle_->get_thrust_policy(),
                          d_mg_distances.begin(),
                          d_mg_distances.end(),
                          flags.begin(),
                          cuda::proclaim_return_type<bool>([invalid_distance] __device__(auto d) {
                            return d != invalid_distance;
                          }));
        cugraph::edge_src_property_t<decltype(mg_graph_view), bool> edge_src_flags(*handle_,
                                                                                   mg_graph_view);
        cugraph::update_edge_src_property(
          *handle_, mg_graph_view, flags.begin(), edge_src_flags.mutable_view());
        auto m = cugraph::count_if_e(
                   *handle_,
                   mg_graph_view,
                   edge_src_flags.view(),
                   cugraph::edge_dst_dummy_property_t{}.view(),
                   cugraph::edge_dummy_property_t{}.view(),
                   [] __device__(auto, auto, auto src_flag, auto, auto) { return src_flag; }) /
                 edge_t{2};
        std::cout << "# visited undirected edges=" << m << std::endl;
      }

      if (bfs_usecase.validate) {
        /* check starting vertex's predecessor */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (starting vertex's predecessor)");
        }

        {
          size_t num_invalids{0};
          if (d_starting_vertex.size() > 0) {
            assert(d_starting_vertex.size() == 1);
            num_invalids = thrust::count_if(
              handle_->get_thrust_policy(),
              d_starting_vertex.begin(),
              d_starting_vertex.end(),
              [v_first      = mg_graph_view.local_vertex_partition_range_first(),
               predecessors = raft::device_span<vertex_t const>(
                 d_mg_predecessors.data(), d_mg_predecessors.size())] __device__(auto v) {
                return predecessors[v - v_first] != invalid_vertex;
              });
          }
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
          ASSERT_EQ(num_invalids, 0)
            << "predecessor of a starting vertex should be invalid_vertex";  // Graph 500 requires
                                                                             // the predecessor of a
                                                                             // starting vertex to
                                                                             // be itself (cuGraph
                                                                             // API specifies that
                                                                             // the predecessor of a
                                                                             // starting vertex is
                                                                             // an invalid vertex,
                                                                             // but this really
                                                                             // doesn't impact
                                                                             // perforamnce)
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check for cycles (update predecessor to predecessor's predecessor till reaching the
         * starting vertex, if there exists a cycle, this won't finish) */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (cycle)");
        }

        {
          vertex_t h_starting_vertex{};
          if (comm_rank == starting_vertex_comm_rank) {
            raft::update_host(
              &h_starting_vertex, d_starting_vertex.data(), 1, handle_->get_stream());
            handle_->sync_stream();
          }
          h_starting_vertex = cugraph::host_scalar_bcast(
            comm, h_starting_vertex, starting_vertex_comm_rank, handle_->get_stream());

          rmm::device_uvector<vertex_t> ancestors(d_mg_predecessors.size(), handle_->get_stream());
          ancestors.resize(
            thrust::distance(
              ancestors.begin(),
              thrust::copy_if(handle_->get_thrust_policy(),
                              d_mg_predecessors.begin(),
                              d_mg_predecessors.end(),
                              ancestors.begin(),
                              cugraph::detail::is_not_equal_t<vertex_t>{invalid_vertex})),
            handle_->get_stream());

          cugraph::kv_store_t<vertex_t, vertex_t, true /* use_binary_search */> kv_store(
            thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_first()),
            thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_last()),
            d_mg_predecessors.begin(),
            invalid_vertex,
            true /* key_sorted */,
            handle_->get_stream());
          auto kv_store_view                  = kv_store.view();
          auto h_vertex_partition_range_lasts = mg_graph_view.vertex_partition_range_lasts();
          auto d_vertex_partition_range_lasts =
            cugraph::test::to_device(*handle_, h_vertex_partition_range_lasts);
          size_t level{0};
          auto aggregate_size = cugraph::host_scalar_allreduce(
            comm, ancestors.size(), raft::comms::op_t::SUM, handle_->get_stream());
          while (aggregate_size > 0) {
            ASSERT_TRUE(level < mg_graph_view.number_of_vertices() - 1)
              << "BFS predecessor tree has a cycle.";
            ancestors.resize(
              thrust::distance(
                ancestors.begin(),
                thrust::remove_if(handle_->get_thrust_policy(),
                                  ancestors.begin(),
                                  ancestors.end(),
                                  cugraph::detail::is_equal_t<vertex_t>{h_starting_vertex})),
              handle_->get_stream());
            ancestors = cugraph::collect_values_for_keys(
              comm,
              kv_store_view,
              ancestors.begin(),
              ancestors.end(),
              cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                  d_vertex_partition_range_lasts.size()),
                major_comm_size,
                minor_comm_size},
              handle_->get_stream());
            aggregate_size = cugraph::host_scalar_allreduce(
              comm, ancestors.size(), raft::comms::op_t::SUM, handle_->get_stream());
            ++level;
          }
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check that distance(src) = distance(predecssor(v)) + 1 */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (predecessor tree distances)");
        }

        {
          rmm::device_uvector<vertex_t> tree_srcs(mg_graph_view.local_vertex_partition_range_size(),
                                                  handle_->get_stream());
          tree_srcs.resize(
            thrust::distance(
              tree_srcs.begin(),
              thrust::copy_if(handle_->get_thrust_policy(),
                              d_mg_predecessors.begin(),
                              d_mg_predecessors.end(),
                              tree_srcs.begin(),
                              cugraph::detail::is_not_equal_t<vertex_t>{invalid_vertex})),
            handle_->get_stream());

          auto tree_src_dists = cugraph::collect_values_for_int_vertices(
            comm,
            tree_srcs.begin(),
            tree_srcs.end(),
            d_mg_distances.begin(),
            mg_graph_view.vertex_partition_range_lasts(),
            mg_graph_view.local_vertex_partition_range_first(),
            handle_->get_stream());

          rmm::device_uvector<vertex_t> tree_dst_dists(tree_src_dists.size(),
                                                       handle_->get_stream());
          thrust::copy_if(handle_->get_thrust_policy(),
                          d_mg_distances.begin(),
                          d_mg_distances.end(),
                          d_mg_predecessors.begin(),
                          tree_dst_dists.begin(),
                          cugraph::detail::is_not_equal_t<vertex_t>{invalid_vertex});

          auto input_pair_first =
            thrust::make_zip_iterator(tree_src_dists.begin(), tree_dst_dists.begin());
          auto num_invalids = thrust::count_if(handle_->get_thrust_policy(),
                                               input_pair_first,
                                               input_pair_first + tree_src_dists.size(),
                                               [] __device__(auto pair) {
                                                 auto src_dist = thrust::get<0>(pair);
                                                 auto dst_dist = thrust::get<1>(pair);
                                                 return (src_dist + 1) != dst_dist;
                                               });
          num_invalids      = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());

          ASSERT_EQ(num_invalids, 0)
            << " source and destination vertices in the BFS predecessor tree are not one hop away.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (graph distances & connected components)");
        }

        /* check distances and connect component coverage in the input graph */

        {
          constexpr size_t num_rounds = 24;  // to cut peak memory usage

          rmm::device_uvector<uint8_t> d_mg_typecasted_distances(d_mg_distances.size(),
                                                                 handle_->get_stream());
          auto max_distance = thrust::transform_reduce(
            handle_->get_thrust_policy(),
            d_mg_distances.begin(),
            d_mg_distances.end(),
            cuda::proclaim_return_type<vertex_t>([invalid_distance] __device__(auto d) {
              return d == invalid_distance ? vertex_t{0} : d;
            }),
            vertex_t{0},
            thrust::maximum<vertex_t>{});
          max_distance = cugraph::host_scalar_allreduce(
            comm, max_distance, raft::comms::op_t::MAX, handle_->get_stream());
          ASSERT_TRUE(max_distance <= std::numeric_limits<uint8_t>::max())
            << "the input graph diameter exceeds std::numeric_limits<uint8_t>::max(), so we "
               "can't use uint8_t to store distances in validation.";
          thrust::transform(handle_->get_thrust_policy(),
                            d_mg_distances.begin(),
                            d_mg_distances.end(),
                            d_mg_typecasted_distances.begin(),
                            cugraph::detail::typecast_t<vertex_t, uint8_t>{});
          cugraph::edge_src_property_t<decltype(mg_graph_view), uint8_t> edge_src_dist(
            *handle_, mg_graph_view);
          cugraph::update_edge_src_property(*handle_,
                                            mg_graph_view,
                                            d_mg_typecasted_distances.begin(),
                                            edge_src_dist.mutable_view());

          size_t num_invalids{0};
          for (size_t r = 0; r < num_rounds; ++r) {
            auto dst_first      = mg_graph_view.local_edge_partition_dst_range_first();
            auto dst_range_size = mg_graph_view.local_edge_partition_dst_range_size();
            auto num_this_round_dsts =
              dst_range_size / num_rounds +
              (r < (dst_range_size % num_rounds) ? vertex_t{1} : vertex_t{0});
            rmm::device_uvector<vertex_t> this_round_dsts(num_this_round_dsts,
                                                          handle_->get_stream());
            thrust::tabulate(handle_->get_thrust_policy(),
                             this_round_dsts.begin(),
                             this_round_dsts.end(),
                             [dst_first, r, num_rounds] __device__(size_t i) {
                               return dst_first + static_cast<vertex_t>(r + i * num_rounds);
                             });

            auto this_round_dst_dists = cugraph::collect_values_for_sorted_unique_int_vertices(
              comm,
              raft::device_span<vertex_t const>(this_round_dsts.data(), this_round_dsts.size()),
              d_mg_typecasted_distances.begin(),
              mg_graph_view.vertex_partition_range_lasts(),
              mg_graph_view.local_vertex_partition_range_first(),
              handle_->get_stream());

            num_invalids += cugraph::count_if_e(
              *handle_,
              mg_graph_view,
              edge_src_dist.view(),
              cugraph::edge_dst_dummy_property_t{}.view(),
              cugraph::edge_dummy_property_t{}.view(),
              [invalid_distance,
               num_rounds,
               r,
               dst_first,
               this_round_dst_dists = raft::device_span<uint8_t const>(
                 this_round_dst_dists.data(),
                 this_round_dst_dists
                   .size())] __device__(auto src, auto dst, auto src_dist, auto, auto) {
                auto dst_offset = dst - dst_first;
                if ((dst_offset % num_rounds) == r) {
                  auto dst_dist = this_round_dst_dists[dst_offset / num_rounds];
                  if (src_dist != invalid_distance) {
                    return (dst_dist == invalid_distance) ||
                           (((src_dist >= dst_dist) ? (src_dist - dst_dist)
                                                    : (dst_dist - src_dist)) > 1);
                  } else {
                    return (dst_dist != invalid_distance);
                  }
                } else {
                  return false;
                }
              });
          }

          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());

          ASSERT_EQ(num_invalids, 0)
            << "only one of the two connected vertices is reachable from the starting vertex or "
               "the distances from the starting vertex differ by more than one.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check that predecessor->v edges exist in the input graph */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (predecessor->v edge existence)");
        }

        {
          rmm::device_uvector<vertex_t> query_srcs(d_mg_predecessors.size(), handle_->get_stream());
          rmm::device_uvector<vertex_t> query_dsts(query_srcs.size(), handle_->get_stream());
          auto input_edge_first = thrust::make_zip_iterator(
            d_mg_predecessors.begin(),
            thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_first()));
          auto output_edge_first =
            thrust::make_zip_iterator(query_srcs.begin(), query_dsts.begin());
          query_srcs.resize(
            thrust::distance(
              output_edge_first,
              thrust::copy_if(handle_->get_thrust_policy(),
                              input_edge_first,
                              input_edge_first + d_mg_predecessors.size(),
                              d_mg_predecessors.begin(),
                              output_edge_first,
                              cugraph::detail::is_not_equal_t<vertex_t>{invalid_vertex})),
            handle_->get_stream());
          query_dsts.resize(query_srcs.size(), handle_->get_stream());

          std::tie(query_srcs, query_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
            cugraph::detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
              vertex_t,
              edge_t,
              weight_t,
              edge_type_t>(*handle_,
                           std::move(query_srcs),
                           std::move(query_dsts),
                           std::nullopt,
                           std::nullopt,
                           std::nullopt,
                           mg_graph_view.vertex_partition_range_lasts());

          auto flags = mg_graph_view.has_edge(
            *handle_,
            raft::device_span<vertex_t const>(query_srcs.data(), query_srcs.size()),
            raft::device_span<vertex_t const>(query_dsts.data(), query_dsts.size()),
            true /* FIXME: remove */);
          auto num_invalids =
            thrust::count(handle_->get_thrust_policy(), flags.begin(), flags.end(), false);
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
          ASSERT_EQ(num_invalids, 0) << "predecessor->v missing in the input graph.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }
      }
    }

    std::cout << "average MG BFS (Kernel 2) time: " << (total_elapsed / num_timed_starting_vertices)
              << std::endl;
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_GRAPH500_MGBFS<input_usecase_t>::handle_ = nullptr;

using Tests_GRAPH500_MGBFS_Rmat = Tests_GRAPH500_MGBFS<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_GRAPH500_MGBFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_GRAPH500_MGBFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(Graph500_BFS_Usecase{false, true},
                    cugraph::test::Rmat_Usecase(10,
                                                16,
                                                0.57,
                                                0.19,
                                                0.19,
                                                0 /* base RNG seed */,
                                                true /* undirected */,
                                                true /* scramble vertex ID */))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_GRAPH500_MGBFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(Graph500_BFS_Usecase{false, false},
                    cugraph::test::Rmat_Usecase(20,
                                                16,
                                                0.57,
                                                0.19,
                                                0.19,
                                                0 /* base RNG seed */,
                                                true /* undirected */,
                                                true /* scramble vertex IDs */))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
