/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

template <typename T>
std::vector<T>
to_host(raft::handle_t const& handle, rmm::device_uvector<T> &data) {
  std::vector<T> h_data(data.size());
  raft::update_host(h_data.data(), data.data(), data.size(), handle.get_stream());
  handle.get_stream_view().synchronize();
  return h_data;
}

template <typename T, typename L>
std::vector<T>
to_host(raft::handle_t const& handle, T const* data, L size) {
  std::vector<T> h_data(size);
  raft::update_host(h_data.data(), data, size, handle.get_stream());
  handle.get_stream_view().synchronize();
  return h_data;
}

template <typename T>
void
print(std::vector<T> &h_data) {
  for (size_t i = 0; i < h_data.size(); ++i) { std::cerr<<h_data[i]<<"\n"; }
}

template <typename result_t, typename vertex_t, typename edge_t>
std::tuple<std::vector<result_t>, std::vector<result_t>, double, size_t>
hits_reference(edge_t const* h_offsets,
               vertex_t const* h_indices,
               vertex_t num_vertices,
               edge_t num_edges,
               size_t max_iterations,
               std::optional<result_t const*> starting_hub_values,
               bool normalized,
               double tolerance)
{
  CUGRAPH_EXPECTS(num_vertices > 1, "number of vertices expected to be non-zero");
  std::vector<result_t> prev_hubs(num_vertices, result_t{1.0} / num_vertices);
  std::vector<result_t> prev_authorities(num_vertices, result_t{1.0} / num_vertices);
  std::vector<result_t> curr_hubs(num_vertices);
  std::vector<result_t> curr_authorities(num_vertices);
  double hubs_error{std::numeric_limits<double>::max()};
  size_t hubs_iterations{0};

  if (starting_hub_values) {
    std::copy((*starting_hub_values), (*starting_hub_values) + num_vertices, prev_hubs.begin());
    auto prev_hubs_norm =
      std::reduce(prev_hubs.begin(), prev_hubs.end(), result_t{0}, [](auto x, auto y) {
        return std::max(x, y);
      });
    std::transform(prev_hubs.begin(), prev_hubs.end(), prev_hubs.begin(), [prev_hubs_norm](auto x) {
      return std::divides{}(x, prev_hubs_norm);
    });
  }

  for (; hubs_iterations < max_iterations; ++hubs_iterations) {
    std::cerr<<"h"<<hubs_iterations<<"-1 :\tprev_hubs\n";
    print(prev_hubs); std::cerr<<"\n";
    std::fill(curr_hubs.begin(), curr_hubs.end(), result_t{0});
    std::fill(curr_authorities.begin(), curr_authorities.end(), result_t{0});
    for (vertex_t src = 0; src < num_vertices; ++src) {
      for (vertex_t dest_index = h_offsets[src]; dest_index < h_offsets[src + 1]; ++dest_index) {
        curr_authorities[h_indices[dest_index]] += prev_hubs[src];
      }
    }
    for (vertex_t src = 0; src < num_vertices; ++src) {
      for (vertex_t dest_index = h_offsets[src]; dest_index < h_offsets[src + 1]; ++dest_index) {
        curr_hubs[src] += curr_authorities[h_indices[dest_index]];
      }
    }
    std::cerr<<"h"<<hubs_iterations<<"-2 :\tauthorities\n";
    print(curr_authorities); std::cerr<<"\n";
    std::cerr<<"h"<<hubs_iterations<<"-3 :\tcurr_hubs\n";
    print(curr_hubs); std::cerr<<"\n";
    auto curr_hubs_norm =
      std::reduce(curr_hubs.begin(), curr_hubs.end(), result_t{0}, [](auto x, auto y) {
        return std::max(x, y);
      });
    std::transform(curr_hubs.begin(), curr_hubs.end(), curr_hubs.begin(), [curr_hubs_norm](auto x) {
      return std::divides{}(x, curr_hubs_norm);
    });
    auto curr_authorities_norm = std::reduce(
      curr_authorities.begin(), curr_authorities.end(), result_t{0}, [](auto x, auto y) {
        return std::max(x, y);
      });
    std::transform(
      curr_authorities.begin(),
      curr_authorities.end(),
      curr_authorities.begin(),
      [curr_authorities_norm](auto x) { return std::divides{}(x, curr_authorities_norm); });
    hubs_error = std::transform_reduce(curr_hubs.begin(),
                                       curr_hubs.end(),
                                       prev_hubs.begin(),
                                       result_t{0},
                                       std::plus<result_t>{},
                                       [](auto x, auto y) { return std::abs(x - y); });
    if (hubs_error < tolerance) {
      break;
    } else {
      std::copy(curr_authorities.begin(), curr_authorities.end(), prev_authorities.begin());
      std::copy(curr_hubs.begin(), curr_hubs.end(), prev_hubs.begin());
    }
  }
  return std::make_tuple(
    std::move(curr_hubs), std::move(curr_authorities), hubs_error, hubs_iterations);
}

struct Hits_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Hits : public ::testing::TestWithParam<std::tuple<Hits_Usecase, input_usecase_t>> {
 public:
  Tests_Hits() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Hits_Usecase const& hits_usecase, input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = false;

    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    // 2. create SG graph

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 3. run hits

    auto graph_view = graph.view();
    auto maximum_iterations = 2;
    weight_t tolerance      = 1e-5;
    rmm::device_uvector<weight_t> d_hubs(graph_view.get_number_of_local_vertices(),
                                                      handle.get_stream());

    rmm::device_uvector<weight_t> d_authorities(graph_view.get_number_of_local_vertices(),
                                                      handle.get_stream());

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto result = cugraph::hits(handle,
        graph_view,
        d_hubs.data(),
        d_authorities.data(),
        tolerance,
        maximum_iterations,
        false,
        true,
        false);

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "HITS took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (hits_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, weight_t, true, false> unrenumbered_graph(handle);
      if (renumber) {
        std::cout << "renumber = true" << std::endl;
        //std::cerr<<"construct_graph unrenumbered"<<std::endl;
        std::tie(unrenumbered_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
            handle, input_usecase, false, false);
      }
        //std::cerr<<"err 0"<<std::endl;
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;
      auto offsets = 
        to_host(handle,
            unrenumbered_graph_view.get_matrix_partition_view().get_offsets(),
            unrenumbered_graph_view.get_number_of_vertices() + 1);
        //std::cerr<<"err 10"<<std::endl;
      auto indices =
        to_host(handle,
            unrenumbered_graph_view.get_matrix_partition_view().get_indices(),
            unrenumbered_graph_view.get_number_of_edges());
        //std::cerr<<"err 11"<<std::endl;
      auto reference_result =
        hits_reference<weight_t>(
            offsets.data(),
            indices.data(),
            unrenumbered_graph_view.get_number_of_vertices(),
            unrenumbered_graph_view.get_number_of_edges(),
            maximum_iterations,
            std::nullopt,
            true,
            tolerance);
        //std::cerr<<"err 12"<<std::endl;
        //std::cerr<<"err 1"<<std::endl;

      std::vector<weight_t> h_cugraph_hits(d_hubs.size());
      if (renumber) {
        rmm::device_uvector<weight_t> d_unrenumbered_hubs(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_hubs) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_hubs);
        //std::cerr<<"err 2"<<std::endl;
        raft::update_host(h_cugraph_hits.data(),
                          d_unrenumbered_hubs.data(),
                          d_unrenumbered_hubs.size(),
                          handle.get_stream());
      } else {
        //std::cerr<<"err 3"<<std::endl;
        raft::update_host(h_cugraph_hits.data(),
                          d_hubs.data(),
                          d_hubs.size(),
                          handle.get_stream());
      }
      handle.get_stream_view().synchronize();
        //std::cerr<<"err 4"<<std::endl;
      //for (size_t i = 0; i < h_cugraph_hits.size(); ++i) {
      //  std::cout<<h_cugraph_hits[i]<<" "<<std::get<0>(reference_result)[i]<<std::endl;
      //}
      auto threshold_ratio = 1e-3;
      auto threshold_magnitude =
        (1.0 / static_cast<weight_t>(graph_view.get_number_of_vertices())) *
        threshold_ratio;  // skip comparison for low Katz Centrality verties (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        bool passed = std::abs(lhs - rhs) <=
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
        if (passed) { std::cout<<lhs<<" "<<rhs<<"\n"; }
        if (!passed) { std::cout<<lhs<<" "<<rhs<<" x\n"; }
        return passed;
      };

      ASSERT_TRUE(std::equal(std::get<0>(reference_result).begin(),
                             std::get<0>(reference_result).end(),
                             h_cugraph_hits.begin(),
                             nearly_equal))
        << "Hits values do not match with the reference values.";
    }

  //  rmm::device_uvector<weight_t> d_hubs(graph_view.get_number_of_vertices(), handle.get_stream());

  //  rmm::device_uvector<weight_t> d_authorities(graph_view.get_number_of_vertices(),
  //                                              handle.get_stream());

  //  cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> hits_graph(
  //    const_cast<edge_t*>(graph_view.get_matrix_partition_view().get_offsets()),
  //    const_cast<vertex_t*>(graph_view.get_matrix_partition_view().get_indices()),
  //    nullptr,
  //    graph_view.get_number_of_vertices(),
  //    graph_view.get_number_of_edges());

  //  if (cugraph::test::g_perf) {
  //    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
  //    hr_clock.start();
  //  }

  //  weight_t* d_hubs_ptr        = d_hubs.data();
  //  weight_t* d_authorities_ptr = d_authorities.data();
  //  weight_t* dummy_ptr         = nullptr;
  //  cugraph::gunrock::hits(
  //    hits_graph, maximum_iterations, tolerance, dummy_ptr, true, d_hubs_ptr, d_authorities_ptr);

  //  if (cugraph::test::g_perf) {
  //    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
  //    double elapsed_time{0.0};
  //    hr_clock.stop(&elapsed_time);
  //    std::cout << "HITS took " << elapsed_time * 1e-6 << " s.\n";
  //  }

  //  if (hits_usecase.check_correctness) {
  //    rmm::device_uvector<vertex_t> d_ranked_hubs_id(size_t{0}, handle.get_stream());
  //    rmm::device_uvector<vertex_t> d_ranked_authorities_id(size_t{0}, handle.get_stream());
  //    std::tie(std::ignore, d_ranked_hubs_id) =
  //      cugraph::test::sort_by_key(handle, d_hubs, *d_renumber_map_labels);
  //    std::tie(std::ignore, d_ranked_authorities_id) =
  //      cugraph::test::sort_by_key(handle, d_authorities, *d_renumber_map_labels);
  //    rmm::device_uvector<vertex_t> ref_ranked_hubs_id(size_t{0}, handle.get_stream());
  //    rmm::device_uvector<vertex_t> ref_ranked_authorities_id(size_t{0}, handle.get_stream());
  //    std::tie(std::ignore, ref_ranked_hubs_id) =
  //      cugraph::test::sort_by_key(handle, reference_result.hubs, (*d_renumber_map_labels));
  //    std::tie(std::ignore, ref_ranked_authorities_id) =
  //      cugraph::test::sort_by_key(handle, reference_result.authorities, (*d_renumber_map_labels));
  //    // compare top k ids
  //  }
  //  if (hits_usecase.check_correctness) {
  //    rmm::device_uvector<result_t> d_unrenumbered_hubs(size_t{0}, handle.get_stream());
  //    rmm::device_uvector<result_t> d_unrenumbered_authorities(size_t{0}, handle.get_stream());
  //    std::tie(std::ignore, d_unrenumbered_hubs) =
  //      cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_hubs);
  //    std::tie(std::ignore, d_unrenumbered_authorities) =
  //      cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_authorities);
  //    std::vector<result_t> h_cugraph_hubs(graph_view.get_number_of_vertices());
  //    std::vector<result_t> h_cugraph_authorities(graph_view.get_number_of_vertices());
  //    raft::update_host(h_cugraph_hubs.data(),
  //                      d_unrenumbered_hubs.data(),
  //                      d_unrenumbered_hubs.size(),
  //                      handle.get_stream());
  //    raft::update_host(h_cugraph_authorities.data(),
  //                      d_unrenumbered_authorities.data(),
  //                      d_unrenumbered_authorities.size(),
  //                      handle.get_stream());
  //    handle.get_stream_view().synchronize();
  //    auto threshold_ratio = 1e-2;
  //    auto threshold_magnitude =
  //      (1.0 / static_cast<result_t>(graph_view.get_number_of_vertices())) *
  //      threshold_ratio;  // skip comparison for low HITS verties (lowly ranked vertices)
  //    auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
  //      return std::abs(lhs - rhs) <=
  //             std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
  //    };

  //    std::tie(std::ignore, reference_result.hubs) =
  //      cugraph::test::sort_by_key(handle, (*d_renumber_map_labels), reference_result.hubs);
  //    std::tie(std::ignore, reference_result.authorities) =
  //      cugraph::test::sort_by_key(handle, (*d_renumber_map_labels), reference_result.authorities);
  //    std::vector<result_t> h_reference_hubs(reference_result.hubs.size());
  //    std::vector<result_t> h_reference_authorities(reference_result.authorities.size());
  //    raft::update_host(h_reference_hubs.data(),
  //                      reference_result.hubs.data(),
  //                      reference_result.hubs.size(),
  //                      handle.get_stream());
  //    raft::update_host(h_reference_authorities.data(),
  //                      reference_result.authorities.data(),
  //                      reference_result.authorities.size(),
  //                      handle.get_stream());
  //    handle.get_stream_view().synchronize();

  //    bool passed = std::equal(
  //      h_reference_hubs.begin(), h_reference_hubs.end(), h_cugraph_hubs.begin(), nearly_equal);
  //    weight_t first, second;
  //    if (!passed) {
  //      auto iter =
  //        std::mismatch(h_reference_hubs.begin(), h_reference_hubs.end(), h_cugraph_hubs.begin());
  //      first  = *iter.first;
  //      second = *iter.second;
  //    }
  //    ASSERT_TRUE(std::equal(
  //      h_reference_hubs.begin(), h_reference_hubs.end(), h_cugraph_hubs.begin(), nearly_equal))
  //      << "HITS values do not match with the reference values." << reference_result.error << " "
  //      << reference_result.iterations << " " << first << " " << second << " "
  //      << "\n";
  //    ASSERT_TRUE(std::equal(h_reference_authorities.begin(),
  //                           h_reference_authorities.end(),
  //                           h_cugraph_authorities.begin(),
  //                           nearly_equal))
  //      << "HITS values do not match with the reference values.";
  //  }
  }
};

using Tests_Hits_File = Tests_Hits<cugraph::test::File_Usecase>;

TEST_P(Tests_Hits_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Hits_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Hits_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));
    //::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
    //                  // cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
    //                  // cugraph::test::File_Usecase("test/datasets/ibm32.mtx"),
    //                  // cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));
    //                  cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
    //                  cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
    //                  cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));
