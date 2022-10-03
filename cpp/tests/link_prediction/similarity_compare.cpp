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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <link_prediction/similarity_compare.hpp>
#include <utilities/test_utilities.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// FIXME:  To support weighted variants we need to compute the minimum weight of edge edge going to
// the intersection
//         and the maximum weight of each edge otherwise.  Need to think about how that works.
//         Probably need to implement a custom set intersection loop rather than using the stl
//         version, since then we can operate on each element appropriately.  That would mean
//         adapting this class.
template <typename T>
struct intersection_count_t {
  size_t count{0};
  using value_type = T;

  void push_back(T const&) { ++count; }
};

namespace cugraph {
namespace test {

template <typename vertex_t, typename weight_t, typename test_t>
void similarity_compare(
  vertex_t num_vertices,
  std::tuple<std::vector<vertex_t>&, std::vector<vertex_t>&, std::optional<std::vector<weight_t>>&>
    edge_list,
  std::tuple<std::vector<vertex_t>&, std::vector<vertex_t>&> vertex_pairs,
  std::vector<weight_t>& similarity_score,
  test_t const& test_functor)
{
  auto& [graph_src, graph_dst, graph_wgt] = edge_list;
  auto& [v1, v2]                          = vertex_pairs;

  auto compare_pairs = [](thrust::tuple<vertex_t, vertex_t> lhs,
                          thrust::tuple<vertex_t, vertex_t> rhs) {
    return ((thrust::get<0>(lhs) < thrust::get<0>(rhs)) ||
            ((thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
             (thrust::get<1>(lhs) < thrust::get<1>(rhs))));
  };

  std::sort(thrust::make_zip_iterator(graph_src.begin(), graph_dst.begin()),
            thrust::make_zip_iterator(graph_src.end(), graph_dst.end()),
            compare_pairs);

  // FIXME: This only tests unweighted, weighted implementation needs to be different
  std::vector<size_t> vertex_degrees(static_cast<size_t>(num_vertices), size_t{0});

  std::for_each(
    graph_src.begin(), graph_src.end(), [&vertex_degrees](auto v) { ++vertex_degrees[v]; });

  auto compare_functor = cugraph::test::nearly_equal<weight_t>{
    weight_t{1e-3}, weight_t{(weight_t{1} / static_cast<weight_t>(num_vertices)) * weight_t{1e-3}}};

  std::for_each(
    thrust::make_zip_iterator(v1.begin(), v2.begin(), similarity_score.begin()),
    thrust::make_zip_iterator(v1.end(), v2.end(), similarity_score.end()),
    [compare_functor, test_functor, &vertex_degrees, &graph_src, &graph_dst, &graph_wgt](
      auto tuple) {
      auto v1    = thrust::get<0>(tuple);
      auto v2    = thrust::get<1>(tuple);
      auto score = thrust::get<2>(tuple);

      auto v1_begin =
        std::distance(graph_src.begin(), std::lower_bound(graph_src.begin(), graph_src.end(), v1));
      auto v1_end =
        std::distance(graph_src.begin(), std::upper_bound(graph_src.begin(), graph_src.end(), v1));

      auto v2_begin =
        std::distance(graph_src.begin(), std::lower_bound(graph_src.begin(), graph_src.end(), v2));
      auto v2_end =
        std::distance(graph_src.begin(), std::upper_bound(graph_src.begin(), graph_src.end(), v2));

      std::vector<vertex_t> intersection(std::min((v1_end - v1_begin), (v2_end - v2_begin)));
      auto intersection_end = std::set_intersection(graph_dst.begin() + v1_begin,
                                                    graph_dst.begin() + v1_end,
                                                    graph_dst.begin() + v2_begin,
                                                    graph_dst.begin() + v2_end,
                                                    intersection.begin());

      auto expected_score = test_functor.compute_score(
        vertex_degrees[v1],
        vertex_degrees[v2],
        static_cast<weight_t>(std::distance(intersection.begin(), intersection_end)));

      EXPECT_TRUE(compare_functor(score, expected_score))
        << "score mismatch, got " << score << ", expected " << expected_score;
    });
}

template void similarity_compare(
  int32_t num_vertices,
  std::tuple<std::vector<int32_t>&, std::vector<int32_t>&, std::optional<std::vector<float>>&>
    edge_list,
  std::tuple<std::vector<int32_t>&, std::vector<int32_t>&> vertex_pairs,
  std::vector<float>& result_score,
  test_jaccard_t const& test_functor);

template void similarity_compare(
  int32_t num_vertices,
  std::tuple<std::vector<int32_t>&, std::vector<int32_t>&, std::optional<std::vector<float>>&>
    edge_list,
  std::tuple<std::vector<int32_t>&, std::vector<int32_t>&> vertex_pairs,
  std::vector<float>& result_score,
  test_sorensen_t const& test_functor);

template void similarity_compare(
  int32_t num_vertices,
  std::tuple<std::vector<int32_t>&, std::vector<int32_t>&, std::optional<std::vector<float>>&>
    edge_list,
  std::tuple<std::vector<int32_t>&, std::vector<int32_t>&> vertex_pairs,
  std::vector<float>& result_score,
  test_overlap_t const& test_functor);

template void similarity_compare(
  int64_t num_vertices,
  std::tuple<std::vector<int64_t>&, std::vector<int64_t>&, std::optional<std::vector<float>>&>
    edge_list,
  std::tuple<std::vector<int64_t>&, std::vector<int64_t>&> vertex_pairs,
  std::vector<float>& result_score,
  test_jaccard_t const& test_functor);

template void similarity_compare(
  int64_t num_vertices,
  std::tuple<std::vector<int64_t>&, std::vector<int64_t>&, std::optional<std::vector<float>>&>
    edge_list,
  std::tuple<std::vector<int64_t>&, std::vector<int64_t>&> vertex_pairs,
  std::vector<float>& result_score,
  test_sorensen_t const& test_functor);

template void similarity_compare(
  int64_t num_vertices,
  std::tuple<std::vector<int64_t>&, std::vector<int64_t>&, std::optional<std::vector<float>>&>
    edge_list,
  std::tuple<std::vector<int64_t>&, std::vector<int64_t>&> vertex_pairs,
  std::vector<float>& result_score,
  test_overlap_t const& test_functor);

}  // namespace test
}  // namespace cugraph
