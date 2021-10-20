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

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

template <typename result_t>
struct hits_result {
  thrust::host_vector<result_t> hubs;
  thrust::host_vector<result_t> authorities;
  double error;
  size_t iterations;
};

template <typename result_t, typename vertex_t>
std::vector<result_t>
unrenumber(
raft::handle_t const& handle,
const rmm::device_uvector<vertex_t> &map,
const thrust::host_vector<result_t> &result) {
  std::vector<vertex_t> h_map(map.size());
  std::vector<result_t> sorted_result(result.begin(), result.end());
  raft::update_host(h_map.data(),
      map.data(),
      map.size(),
      handle.get_stream());
  thrust::sort_by_key(thrust::host,
      h_map.begin(),
      h_map.end(),
      sorted_result.begin());
  return sorted_result;
}

struct abs_sum{
  template <typename T>
  __host__ T operator()(T x, T y) {
    return std::abs(x) + std::abs(y);
  }
};

template <typename result_t, typename vertex_t, typename graph_t>
hits_result<result_t>
hits_reference(raft::handle_t const& handle,
    graph_t &graph_view,
    size_t max_iterations,
    std::optional<result_t const*> starting_hub_values,
    bool normalized,
    double tolerance)
{
  using edge_t = typename graph_t::edge_type;
  std::vector<edge_t> h_offsets(graph_view.get_number_of_vertices() + 1);
  std::vector<vertex_t> h_indices(graph_view.get_number_of_edges());
  vertex_t num_vertices = graph_view.get_number_of_vertices();
  raft::update_host(h_offsets.data(),
                    graph_view.get_matrix_partition_view().get_offsets(),
                    graph_view.get_number_of_vertices() + 1,
                    handle.get_stream());
  raft::update_host(h_indices.data(),
                    graph_view.get_matrix_partition_view().get_indices(),
                    graph_view.get_number_of_edges(),
                    handle.get_stream());
  handle.get_stream_view().synchronize();
  thrust::host_vector<result_t> prev_hubs(num_vertices, result_t{1.0}/num_vertices);
  thrust::host_vector<result_t> prev_authorities(num_vertices, result_t{1.0}/num_vertices);
  thrust::host_vector<result_t> curr_hubs(num_vertices);
  thrust::host_vector<result_t> curr_authorities(num_vertices);
  double hubs_error{tolerance};
  size_t hubs_iterations{0};

  if (starting_hub_values) {
    std::copy((*starting_hub_values), (*starting_hub_values) + num_vertices,
        prev_hubs.begin());
    auto prev_hubs_norm = thrust::reduce(prev_hubs.begin(), prev_hubs.end(),
        result_t{0},
        abs_sum());
        //[] (auto x, auto y) { return std::abs(x) + std::abs(y);});
    thrust::transform(
        prev_hubs.begin(),
        prev_hubs.end(),
        thrust::make_constant_iterator(prev_hubs_norm),
        prev_hubs.begin(),
        thrust::divides<result_t>());
  }

  for (;hubs_iterations < max_iterations; ++hubs_iterations) {
    thrust::copy(
        thrust::make_constant_iterator(0),
        thrust::make_constant_iterator(0) + curr_hubs.size(),
        curr_hubs.begin());
    thrust::copy(
        thrust::make_constant_iterator(0),
        thrust::make_constant_iterator(0) + curr_authorities.size(),
        curr_authorities.begin());
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
    auto curr_hubs_norm = thrust::reduce(curr_hubs.begin(), curr_hubs.end(),
        result_t{0},
        abs_sum());
        //[] (auto x, auto y) { return std::abs(x) + std::abs(y);});
    thrust::transform(
        curr_hubs.begin(),
        curr_hubs.end(),
        thrust::make_constant_iterator(curr_hubs_norm),
        curr_hubs.begin(),
        thrust::divides<result_t>());
    auto curr_authorities_norm = thrust::reduce(curr_authorities.begin(), curr_authorities.end(),
        result_t{0},
        abs_sum());
        //[] (auto x, auto y) { return std::abs(x) + std::abs(y);});
    thrust::transform(
        curr_authorities.begin(),
        curr_authorities.end(),
        thrust::make_constant_iterator(curr_authorities_norm),
        curr_authorities.begin(),
        thrust::divides<result_t>());
    auto hubs_error_iter = thrust::make_transform_iterator(
        thrust::make_zip_iterator(curr_hubs.begin(), prev_hubs.begin()),
        [] (auto x) { return std::abs(thrust::get<0>(x) - thrust::get<1>(x)); });
    hubs_error = thrust::reduce(hubs_error_iter, hubs_error_iter + curr_hubs.size());
    if (hubs_error < tolerance) {
      break;
    } else {
      std::copy(curr_authorities.begin(), curr_authorities.end(),
          prev_authorities.begin());
      std::copy(curr_hubs.begin(), curr_hubs.end(),
          prev_hubs.begin());
    }
  }
  return hits_result<result_t>{std::move(curr_hubs), std::move(curr_authorities), hubs_error, hubs_iterations};
}

struct Hits_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Hits
  : public ::testing::TestWithParam<std::tuple<Hits_Usecase, input_usecase_t>> {
 public:
  Tests_Hits() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(Hits_Usecase const& hits_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto graph_view = graph.view();

    auto maximum_iterations = 1000;
    weight_t tolerance = 1e-6;

    auto reference_result = hits_reference<weight_t, vertex_t>(handle, graph_view, maximum_iterations, std::nullopt, true, tolerance);

    rmm::device_uvector<weight_t> d_hubs(graph_view.get_number_of_vertices(),
                                                      handle.get_stream());

    rmm::device_uvector<weight_t> d_authorities(graph_view.get_number_of_vertices(),
                                                      handle.get_stream());

    cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t>
      hits_graph(
        const_cast<edge_t *>(graph_view.get_matrix_partition_view().get_offsets()),
        const_cast<vertex_t *>(graph_view.get_matrix_partition_view().get_indices()),
        nullptr,
        graph_view.get_number_of_vertices(),
        graph_view.get_number_of_edges());

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    weight_t * d_hubs_ptr = d_hubs.data();
    weight_t * d_authorities_ptr = d_authorities.data();
    weight_t * dummy_ptr = nullptr;
    cugraph::gunrock::hits(
        hits_graph,
        maximum_iterations,
        tolerance,
        dummy_ptr,
        true,
        d_hubs_ptr,
        d_authorities_ptr);

    if (cugraph::test::g_perf) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "HITS took " << elapsed_time * 1e-6 << " s.\n";
    }

    //if (hits_usecase.check_correctness) {
    //  rmm::device_uvector<vertex_t> d_ranked_hubs_id(size_t{0}, handle.get_stream());
    //  rmm::device_uvector<vertex_t> d_ranked_authorities_id(size_t{0}, handle.get_stream());
    //  std::tie(std::ignore, d_ranked_hubs_id) =
    //    cugraph::test::sort_by_key(handle,
    //        d_hubs,
    //        *d_renumber_map_labels);
    //  std::tie(std::ignore, d_ranked_authorities_id) =
    //    cugraph::test::sort_by_key(handle,
    //        d_authorities,
    //        *d_renumber_map_labels);
    //  std::vector<vertex_t> ref_ranked_hubs_id(d_ranked_hubs_id.size());
    //  std::vector<vertex_t> ref_ranked_authorities_id(d_ranked_authorities_id.size());
    //}
    if (hits_usecase.check_correctness) {

      rmm::device_uvector<result_t> d_unrenumbered_hubs(size_t{0},
                                                                      handle.get_stream());
      rmm::device_uvector<result_t> d_unrenumbered_authorities(size_t{0},
                                                                      handle.get_stream());
      std::tie(std::ignore, d_unrenumbered_hubs) =
        cugraph::test::sort_by_key(handle,
            *d_renumber_map_labels,
            d_hubs);
      std::tie(std::ignore, d_unrenumbered_authorities) =
        cugraph::test::sort_by_key(handle,
            *d_renumber_map_labels,
            d_authorities);
      std::vector<result_t> h_cugraph_hubs(graph_view.get_number_of_vertices());
      std::vector<result_t> h_cugraph_authorities(graph_view.get_number_of_vertices());
      raft::update_host(h_cugraph_hubs.data(),
          d_unrenumbered_hubs.data(),
          d_unrenumbered_hubs.size(),
          handle.get_stream());
      raft::update_host(h_cugraph_authorities.data(),
          d_unrenumbered_authorities.data(),
          d_unrenumbered_authorities.size(),
          handle.get_stream());
      auto threshold_ratio = 1e-2;
      auto threshold_magnitude =
        (1.0 / static_cast<result_t>(graph_view.get_number_of_vertices())) *
        threshold_ratio;  // skip comparison for low HITS verties (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        return std::abs(lhs - rhs) <=
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
      };
      
      auto h_reference_hubs = unrenumber(handle, (*d_renumber_map_labels), reference_result.hubs);
      auto h_reference_authorities = unrenumber(handle, (*d_renumber_map_labels), reference_result.authorities);

      bool passed = std::equal(h_reference_hubs.begin(),
                             h_reference_hubs.end(),
                             h_cugraph_hubs.begin(),
                             nearly_equal);
      weight_t first, second;
      if (!passed) {
        auto iter = std::mismatch(h_reference_hubs.begin(), h_reference_hubs.end(),
            h_cugraph_hubs.begin());
        first = *iter.first;
        second = *iter.second;
      }
      ASSERT_TRUE(std::equal(h_reference_hubs.begin(),
                             h_reference_hubs.end(),
                             h_cugraph_hubs.begin(),
                             nearly_equal))
        << "HITS values do not match with the reference values."
        << reference_result.error << " "
        << reference_result.iterations << " "
        << first << " "
        << second << " "
        << "\n";
      ASSERT_TRUE(std::equal(h_reference_authorities.begin(),
                             h_reference_authorities.end(),
                             h_cugraph_authorities.begin(),
                             nearly_equal))
        << "HITS values do not match with the reference values.";
    }
  }
};

using Tests_Hits_File = Tests_Hits<cugraph::test::File_Usecase>;

TEST_P(Tests_Hits_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Hits_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Hits_Usecase{true}),
    //::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      //cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      //cugraph::test::File_Usecase("test/datasets/ibm32.mtx"),
                      //cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));
