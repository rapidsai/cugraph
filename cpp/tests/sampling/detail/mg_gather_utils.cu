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
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/graph_functions.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <cuco/detail/hash_functions.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/equal.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t>
rmm::device_uvector<vertex_t> random_vertex_ids(raft::handle_t const& handle,
                                                vertex_t begin,
                                                vertex_t end,
                                                vertex_t count,
                                                int repetitions_per_vertex = 0)
{
  auto& comm                  = handle.get_comms();
  auto const comm_rank        = comm.get_rank();
  vertex_t number_of_vertices = end - begin;

  rmm::device_uvector<vertex_t> vertices(
    std::max((repetitions_per_vertex + 1) * number_of_vertices, count), handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    vertices.begin(),
    vertices.end(),
    [begin, number_of_vertices] __device__(auto v) { return begin + (v % number_of_vertices); });
  thrust::default_random_engine g;
  g.seed(comm_rank);
  thrust::shuffle(handle.get_thrust_policy(), vertices.begin(), vertices.end(), g);
  vertices.resize(count, handle.get_stream());
  return vertices;
}

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>>
create_segmented_data(raft::handle_t const& handle,
                      vertex_t invalid_vertex_id,
                      rmm::device_uvector<edge_t> const& out_degrees)
{
  rmm::device_uvector<edge_t> offset(out_degrees.size() + 1, handle.get_stream());
  // no need for sync since gather call is on stream
  offset.set_element_to_zero_async(0, handle.get_stream());
  thrust::inclusive_scan(
    handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end(), offset.begin() + 1);
  auto total_edge_count = offset.back_element(handle.get_stream());
  rmm::device_uvector<vertex_t> segmented_sources(total_edge_count, handle.get_stream());
  rmm::device_uvector<edge_t> segmented_sequence(total_edge_count, handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), segmented_sources.begin(), segmented_sources.end(), vertex_t{0});
  thrust::fill(
    handle.get_thrust_policy(), segmented_sequence.begin(), segmented_sequence.end(), edge_t{1});
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(offset.size()),
                   [offset       = offset.data(),
                    source_count = out_degrees.size(),
                    src          = segmented_sources.data(),
                    seq          = segmented_sequence.data()] __device__(auto index) {
                     auto location = offset[index];
                     if (index == 0) {
                       seq[location] = edge_t{0};
                     } else {
                       seq[location] = offset[index - 1] - offset[index] + 1;
                     }
                     if (index < source_count) { src[location] = index; }
                   });
  thrust::inclusive_scan(handle.get_thrust_policy(),
                         segmented_sequence.begin(),
                         segmented_sequence.end(),
                         segmented_sequence.begin());
  thrust::inclusive_scan(handle.get_thrust_policy(),
                         segmented_sources.begin(),
                         segmented_sources.end(),
                         segmented_sources.begin(),
                         thrust::maximum<vertex_t>());
  return std::make_tuple(
    std::move(offset), std::move(segmented_sources), std::move(segmented_sequence));
}

template <typename GraphViewType, typename VertexIterator, typename EdgeIndexIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
sg_gather_edges(raft::handle_t const& handle,
                GraphViewType const& graph_view,
                VertexIterator vertex_input_first,
                VertexIterator vertex_input_last,
                EdgeIndexIterator edge_index_first,
                typename GraphViewType::vertex_type invalid_vertex_id,
                typename GraphViewType::edge_type indices_per_source)
{
  static_assert(GraphViewType::is_adj_matrix_transposed == false);
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using weight_t    = typename GraphViewType::weight_type;
  auto source_count = thrust::distance(vertex_input_first, vertex_input_last);
  auto edge_count   = source_count * indices_per_source;
  rmm::device_uvector<vertex_t> sources(edge_count, handle.get_stream());
  rmm::device_uvector<vertex_t> destinations(edge_count, handle.get_stream());
  auto matrix_partition =
    cugraph::matrix_partition_device_view_t<vertex_t, edge_t, weight_t, false>(
      graph_view.get_matrix_partition_view());
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(edge_count),
                   [vertex_input_first,
                    indices_per_source,
                    edge_index_first,
                    sources      = sources.data(),
                    destinations = destinations.data(),
                    offsets      = matrix_partition.get_offsets(),
                    indices      = matrix_partition.get_indices(),
                    invalid_vertex_id] __device__(auto index) {
                     auto source        = vertex_input_first[index / indices_per_source];
                     sources[index]     = source;
                     auto source_offset = offsets[source];
                     auto degree        = offsets[source + 1] - source_offset;
                     auto e_index       = edge_index_first[index];
                     if (e_index < degree) {
                       destinations[index] = indices[source_offset + e_index];
                     } else {
                       destinations[index] = invalid_vertex_id;
                     }
                   });
  auto input_iter =
    thrust::make_zip_iterator(thrust::make_tuple(sources.begin(), destinations.begin()));
  auto compacted_length = thrust::distance(
    input_iter,
    thrust::remove_if(
      handle.get_thrust_policy(),
      input_iter,
      input_iter + destinations.size(),
      destinations.begin(),
      [invalid_vertex_id] __device__(auto dst) { return (dst == invalid_vertex_id); }));
  sources.resize(compacted_length, handle.get_stream());
  destinations.resize(compacted_length, handle.get_stream());
  return std::make_tuple(std::move(sources), std::move(destinations));
}

template <typename vertex_t>
void sort_coo(raft::handle_t const& handle,
              rmm::device_uvector<vertex_t>& src,
              rmm::device_uvector<vertex_t>& dst)
{
  thrust::sort_by_key(handle.get_thrust_policy(), dst.begin(), dst.end(), src.begin());
  thrust::sort_by_key(handle.get_thrust_policy(), src.begin(), src.end(), dst.begin());
}

template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> generate_random_destination_indices(
  raft::handle_t const& handle,
  const rmm::device_uvector<edge_t>& out_degrees,
  vertex_t invalid_vertex_id,
  edge_t invalid_destination_index,
  edge_t indices_per_source)
{
  auto [random_source_offsets, segmented_source_ids, segmented_sequence] =
    create_segmented_data(handle, invalid_vertex_id, out_degrees);
  // Generate random weights to shuffle sequence of destination indices
  rmm::device_uvector<int> random_weights(segmented_sequence.size(), handle.get_stream());
  auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_rank  = row_comm.get_rank();
  auto& comm           = handle.get_comms();
  auto const comm_rank = comm.get_rank();
  auto force_seed      = 0;
  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(random_weights.size()),
                    random_weights.begin(),
                    [force_seed] __device__(auto index) {
                      thrust::default_random_engine g;
                      g.seed(force_seed);
                      thrust::uniform_int_distribution<int> dist;
                      g.discard(index);
                      return dist(g);
                    });
  thrust::sort_by_key(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(
      thrust::make_tuple(segmented_source_ids.begin(), random_weights.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(segmented_source_ids.end(), random_weights.end())),
    segmented_sequence.begin(),
    [] __device__(auto left, auto right) { return left < right; });

  rmm::device_uvector<edge_t> dst_index(indices_per_source * out_degrees.size(),
                                        handle.get_stream());

  thrust::for_each(handle.get_thrust_policy(),
                   thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(out_degrees.size()),
                   [offset    = random_source_offsets.data(),
                    dst_index = dst_index.data(),
                    seg_seq   = segmented_sequence.data(),
                    k         = indices_per_source,
                    invalid_destination_index] __device__(auto index) {
                     auto length = thrust::minimum<edge_t>()(offset[index + 1] - offset[index], k);
                     auto source_offset = offset[index];
                     // copy first k valid destination indices. If k is larger
                     // than out degree then stop at out degree to avoid
                     // out of bounds access
                     for (edge_t i = 0; i < length; ++i) {
                       dst_index[index * k + i] = seg_seq[source_offset + i];
                     }
                     // If requested number of destination indices is larger than
                     // out degree then write out invalid destination index
                     for (edge_t i = length; i < k; ++i) {
                       dst_index[index * k + i] = invalid_destination_index;
                     }
                   });
  return dst_index;
}

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

    auto [global_degree_offset, global_out_degrees] =
      cugraph::detail::get_global_degree_information(handle, mg_graph_view);

    // Generate random sources to gather on
    auto random_sources = random_vertex_ids(handle,
                                            mg_graph_view.get_local_vertex_first(),
                                            mg_graph_view.get_local_vertex_last(),
                                            source_sample_count,
                                            repetitions_per_vertex);
    rmm::device_uvector<int> random_source_gpu_ids(random_sources.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 random_source_gpu_ids.begin(),
                 random_source_gpu_ids.end(),
                 comm_rank);

    auto [active_sources_in_row, active_source_gpu_ids] =
      cugraph::detail::gather_active_sources_in_row(handle,
                                                    mg_graph_view,
                                                    random_sources.cbegin(),
                                                    random_sources.cend(),
                                                    random_source_gpu_ids.cbegin());

    // get source global out degrees to generate indices
    auto active_source_degrees = cugraph::detail::get_active_major_global_degrees(
      handle, mg_graph_view, active_sources_in_row, global_out_degrees);

    auto random_destination_indices =
      generate_random_destination_indices(handle,
                                          active_source_degrees,
                                          mg_graph_view.get_number_of_vertices(),
                                          mg_graph_view.get_number_of_edges(),
                                          indices_per_source);

    auto [src, dst, gpu_ids] =
      cugraph::detail::gather_local_edges(handle,
                                          mg_graph_view,
                                          active_sources_in_row,
                                          active_source_gpu_ids,
                                          random_destination_indices.cbegin(),
                                          indices_per_source,
                                          global_degree_offset);

    if (prims_usecase.check_correctness) {
      // Gather outputs
      auto mg_out_srcs = cugraph::test::device_gatherv(handle, src.data(), src.size());
      auto mg_out_dsts = cugraph::test::device_gatherv(handle, dst.data(), dst.size());

      // Gather inputs
      auto& col_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_rank = col_comm.get_rank();
      auto sg_random_srcs = cugraph::test::device_gatherv(
        handle, active_sources_in_row.data(), col_rank == 0 ? active_sources_in_row.size() : 0);
      auto sg_random_dst_indices =
        cugraph::test::device_gatherv(handle,
                                      random_destination_indices.data(),
                                      col_rank == 0 ? random_destination_indices.size() : 0);

      // Gather input graph edgelist
      rmm::device_uvector<vertex_t> sg_src(0, handle.get_stream());
      rmm::device_uvector<vertex_t> sg_dst(0, handle.get_stream());
      std::tie(sg_src, sg_dst, std::ignore) =
        mg_graph_view.decompress_to_edgelist(handle, std::nullopt);

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
                                                          sg_graph_view.get_number_of_vertices(),
                                                          indices_per_source);
        sort_coo(handle, sg_out_srcs, sg_out_dsts);

        auto passed = thrust::equal(
          handle.get_thrust_policy(), sg_out_srcs.begin(), sg_out_srcs.end(), mg_out_srcs.begin());
        passed &= thrust::equal(
          handle.get_thrust_policy(), sg_out_dsts.begin(), sg_out_dsts.end(), mg_out_dsts.begin());
        ASSERT_TRUE(passed);
      }
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
