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
  raft::print_host_vector("pair1", std::get<0>(vertex_pairs).data(), std::get<0>(vertex_pairs).size(), std::cout);
  raft::print_host_vector("pair2", std::get<1>(vertex_pairs).data(), std::get<1>(vertex_pairs).size(), std::cout);
  raft::print_host_vector("score", similarity_score.data(), similarity_score.size(), std::cout);
  // TBD
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
