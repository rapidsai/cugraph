/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/partition_manager.hpp>

#include <cuco/detail/hash_functions.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/gather_edges.cuh>

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
  thrust::fill(handle.get_thrust_policy(),
               segmented_sources.begin(),
               segmented_sources.end(),
               vertex_t{invalid_vertex_id});
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
                         thrust::minimum<vertex_t>());
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
                int indices_per_source)
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
                    indices      = matrix_partition.get_indices()] __device__(auto index) {
                     auto source         = vertex_input_first[index / indices_per_source];
                     sources[index]      = source;
                     destinations[index] = indices[offsets[source] + edge_index_first[index]];
                   });
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
  int indices_per_source)
{
  auto [random_source_offsets, segmented_source_ids, segmented_sequence] =
    create_segmented_data(handle, invalid_vertex_id, out_degrees);
  // Generate random weights to shuffle sequence of destination indices
  rmm::device_uvector<int> random_weights(segmented_sequence.size(), handle.get_stream());
  auto& row_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_rank        = row_comm.get_rank();
  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(random_weights.size()),
                    random_weights.begin(),
                    [row_rank] __device__(auto index) {
                      thrust::default_random_engine g;
                      g.seed(row_rank);
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
                    k         = static_cast<edge_t>(indices_per_source),
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

template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
generate_random_destination_indices(raft::handle_t const& handle,
                                    GraphViewType const& graph_view,
                                    typename GraphViewType::vertex_type vertex_count,
                                    int indices_per_source,
                                    int repetitions_per_vertex)
{
  auto random_sources = random_vertex_ids(handle,
                                          graph_view.get_local_vertex_first(),
                                          graph_view.get_local_vertex_last(),
                                          repetitions_per_vertex);
  auto out_degrees    = graph_view.compute_out_degrees(handle);
  auto [random_source_offsets, segmented_sources, segmented_sequence] =
    create_segmented_data(handle, graph_view, random_sources, out_degrees);

  // Generate random weights to shuffle sequence of destination indices
  rmm::device_uvector<int> random_weights(segmented_sequence.size(), handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(random_weights.size()),
                    random_weights.begin(),
                    [] __device__(auto index) {
                      thrust::default_random_engine g;
                      thrust::uniform_int_distribution<int> dist;
                      g.discard(index);
                      return dist(g);
                    });
  thrust::sort_by_key(handle.get_thrust_policy(),
                      segmented_sources.begin(),
                      segmented_sources.end(),
                      thrust::make_zip_iterator(
                        thrust::make_tuple(segmented_sequence.begin(), random_weights.begin())));

  using edge_t = typename GraphViewType::edge_type;
  rmm::device_uvector<edge_t> dst_index(indices_per_source * random_sources.size(),
                                        handle.get_stream());

  // random_source_offsets.size() == random_sources.size() + 1
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(random_sources.size()),
                   [offset            = random_source_offsets.data(),
                    dst_index         = dst_index.data(),
                    seg_seq           = segmented_sequence.data(),
                    k                 = static_cast<edge_t>(indices_per_source),
                    invalid_dst_index = graph_view.get_number_of_edges()] __device__(auto index) {
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
                       dst_index[index * k + i] = invalid_dst_index;
                     }
                   });

  return std::make_tuple(std::move(random_sources), std::move(dst_index));
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

    auto [mg_graph, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        handle, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view                        = mg_graph.view();
    constexpr int indices_per_source          = 2;
    constexpr vertex_t repetitions_per_vertex = 5;
    constexpr vertex_t source_sample_count    = 8;

    // 3. Gather mnmg call
    // Generate random vertex ids in the range of current gpu

    // cugraph::detail::debug::print(handle, mg_graph_view);

    auto [global_degree_offset, global_out_degrees] =
      cugraph::get_global_degree_information(handle, mg_graph_view);

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
      cugraph::gather_active_sources_in_row(handle,
                                            mg_graph_view,
                                            random_sources.begin(),
                                            random_sources.end(),
                                            random_source_gpu_ids.begin());

    // get source global out degrees to generate indices
    auto active_source_degrees = get_active_source_global_degrees(
      handle, mg_graph_view, active_sources_in_row, global_out_degrees);

    auto random_destination_indices =
      generate_random_destination_indices(handle,
                                          active_source_degrees,
                                          mg_graph_view.get_number_of_vertices(),
                                          mg_graph_view.get_number_of_edges(),
                                          indices_per_source);

    auto [src, dst, gpu_ids] = cugraph::gather_local_edges(handle,
                                                           mg_graph_view,
                                                           active_sources_in_row,
                                                           active_source_gpu_ids,
                                                           random_destination_indices.begin(),
                                                           mg_graph_view.get_number_of_edges(),
                                                           indices_per_source,
                                                           global_degree_offset);

    // cugraph::detail::debug::print(handle, src, "src");
    // cugraph::detail::debug::print(handle, dst, "dst");
    //    if (prims_usecase.check_correctness) {
    //      //Gather renumbering labels
    //      auto mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
    //        handle, (*mg_renumber_map_labels).data(), (*mg_renumber_map_labels).size());
    //
    //      //Gather inputs
    //      auto sg_random_srcs = cugraph::test::device_gatherv(
    //        handle, random_sources.data(), random_sources.size());
    //      auto sg_random_dst_indices = cugraph::test::device_gatherv(
    //        handle, random_destination_indices.data(), random_destination_indices.size());
    //
    //      //Gather outputs
    //      auto mg_out_srcs = cugraph::test::device_gatherv(
    //        handle, src.data(), src.size());
    //      auto mg_out_dsts = cugraph::test::device_gatherv(
    //        handle, dst.data(), dst.size());
    //
    //      if (handle.get_comms().get_rank() == int{0}) {
    //        //Create sg graph
    //        cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> sg_graph(
    //          handle);
    //        std::tie(sg_graph, std::ignore) = cugraph::test::
    //          construct_graph<vertex_t, edge_t, weight_t, false, false>(
    //            handle, input_usecase, false, false);
    //        auto sg_graph_view = sg_graph.view();
    //
    //        //Unrenumber input
    //        cugraph::unrenumber_int_vertices<vertex_t, false>(
    //          handle,
    //          sg_random_srcs.begin(),
    //          sg_random_srcs.size(),
    //          mg_aggregate_renumber_map_labels.data(),
    //          std::vector<vertex_t>{mg_graph_view.get_number_of_vertices()});
    //        //cugraph::unrenumber_int_vertices<vertex_t, false>(
    //        //  handle,
    //        //  sg_random_dst_indices.begin(),
    //        //  sg_random_dst_indices.size(),
    //        //  mg_aggregate_renumber_map_labels.data(),
    //        //  std::vector<vertex_t>{mg_graph_view.get_number_of_vertices()});
    //
    //        //Unrenumber output
    //        cugraph::unrenumber_int_vertices<vertex_t, false>(
    //          handle,
    //          mg_out_srcs.begin(),
    //          mg_out_srcs.size(),
    //          mg_aggregate_renumber_map_labels.data(),
    //          std::vector<vertex_t>{mg_graph_view.get_number_of_vertices()});
    //        cugraph::unrenumber_int_vertices<vertex_t, false>(
    //          handle,
    //          mg_out_dsts.begin(),
    //          mg_out_dsts.size(),
    //          mg_aggregate_renumber_map_labels.data(),
    //          std::vector<vertex_t>{mg_graph_view.get_number_of_vertices()});
    //
    //        //Call single gpu gather
    //        auto [sg_out_srcs, sg_out_dsts] =
    //          sg_gather_edges(handle,
    //                          sg_graph_view,
    //                          sg_random_srcs.begin(),
    //                          sg_random_srcs.end(),
    //                          sg_random_dst_indices.begin(),
    //                          indices_per_source);

    // Sort mg output for comparison
    // sort_coo(handle, sg_out_srcs, sg_out_dsts);
    // sort_coo(handle, mg_out_srcs, mg_out_dsts);

#if 0
  auto matrix_partition =
    cugraph::matrix_partition_device_view_t<vertex_t, edge_t, weight_t, false>(
      sg_graph_view.get_matrix_partition_view());
  rmm::device_uvector<vertex_t> ur_indices(sg_graph_view.get_number_of_edges(), handle.get_stream());
  raft::update_device(ur_indices.data(), 
                      matrix_partition.get_indices(),
                      sg_graph_view.get_number_of_edges(), handle.get_stream());
  //cugraph::unrenumber_int_vertices<vertex_t, false>(
  //  handle,
  //  ur_indices.begin(),
  //  ur_indices.size(),
  //  mg_aggregate_renumber_map_labels.data(),
  //  std::vector<vertex_t>{mg_graph_view.get_number_of_vertices()});
  //cugraph::detail::debug::print(handle,
  //                              matrix_partition.get_offsets(),
  //                              matrix_partition.get_offsets() + sg_graph_view.get_number_of_vertices()+1,
  //                              "offsets", "\t");
  //cugraph::detail::debug::print(handle,
  //                              ur_indices.begin(),
  //                              ur_indices.end(),
  //                              "indices", "\t");
#endif
    // cugraph::detail::debug::print(handle, sg_random_srcs, "sg_random_srcs");
    // cugraph::detail::debug::print(handle, sg_random_dst_indices, "sg_random_dst_indices");

    // cugraph::detail::debug::print(handle, sg_out_srcs, "sg_out_srcs");
    // cugraph::detail::debug::print(handle, sg_out_dsts, "sg_out_dsts");

    // cugraph::detail::debug::print(handle, mg_out_srcs, "mg_out_srcs");
    // cugraph::detail::debug::print(handle, mg_out_dsts, "mg_out_dsts");

    //        auto passed = thrust::equal(handle.get_thrust_policy(),
    //                      sg_out_srcs.begin(),
    //                      sg_out_srcs.end(),
    //                      mg_out_srcs.begin());
    //        passed &= thrust::equal(handle.get_thrust_policy(),
    //                      sg_out_dsts.begin(),
    //                      sg_out_dsts.end(),
    //                      mg_out_dsts.begin());
    //        ASSERT_TRUE(passed);
    //      }
    //    }
  }
};

using Tests_MG_GatherEdges_File = Tests_MG_GatherEdges<cugraph::test::File_Usecase>;

TEST_P(Tests_MG_GatherEdges_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_GatherEdges_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
#if 1
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));
#else
                     ::testing::Values(
                       cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                       cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                       cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                       cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));
#endif

CUGRAPH_MG_TEST_PROGRAM_MAIN()
