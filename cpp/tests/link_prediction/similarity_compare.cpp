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
void similarity_compare(vertex_t num_vertices,
                        std::vector<vertex_t>&& src,
                        std::vector<vertex_t>&& dst,
                        std::optional<std::vector<weight_t>>&& wgt,
                        std::vector<vertex_t>&& result_src,
                        std::vector<vertex_t>&& result_dst,
                        std::vector<weight_t>&& result_score,
                        test_t const& test_functor)
{
  ASSERT_TRUE(num_vertices > 1) << "number of vertices expected to be non-zero";
  ASSERT_EQ(src.size(), dst.size());
  ASSERT_EQ(result_src.size(), result_dst.size());
  ASSERT_EQ(result_src.size(), result_score.size());

  thrust::sort(thrust::host,
               thrust::make_zip_iterator(src.begin(), dst.begin()),
               thrust::make_zip_iterator(src.end(), dst.end()));
  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(result_src.begin(), result_dst.begin(), result_score.begin()),
    thrust::make_zip_iterator(result_src.end(), result_dst.end(), result_score.end()));

  size_t result_pos = 0;

  // Iterate over all (u,v) pairs
  for (vertex_t u = 0; u < num_vertices; ++u) {
    auto pos     = std::lower_bound(result_src.begin(), result_src.end(), u);
    auto u_start = std::lower_bound(src.begin(), src.end(), u);
    auto u_end   = std::lower_bound(u_start, src.end(), u);

    for (vertex_t v = 0; v < num_vertices; ++v) {
      if (u != v) {
        auto v_start = std::lower_bound(src.begin(), src.end(), v);
        auto v_end   = std::lower_bound(v_start, src.end(), v);

        intersection_count_t<vertex_t> intersection{0};
        std::set_intersection(u_start, u_end, v_start, v_end, std::back_inserter(intersection));

        weight_t u_intersect_v = intersection.count;

        if (u_intersect_v > weight_t{0}) {
          ASSERT_EQ(u, result_src[result_pos]);
          ASSERT_EQ(v, result_dst[result_pos]);

#if 1
          weight_t score = test_functor.compute_score(
            std::distance(u_start, u_end), std::distance(v_start, v_end), u_intersect_v);
#else
          weight_t score = test_functor.compute_score(
            std::distance(u_start, u_end), std::distance(v_start, v_end), u_intersect_v);
#endif
          ASSERT_NEAR(score, result_score[result_pos], 1e-6);

          ++result_pos;
        }
      }
    }
  }

  ASSERT_EQ(result_pos, result_src.size());
}

template void similarity_compare(int32_t num_vertices,
                                 std::vector<int32_t>&& src,
                                 std::vector<int32_t>&& dst,
                                 std::optional<std::vector<float>>&& wgt,
                                 std::vector<int32_t>&& result_src,
                                 std::vector<int32_t>&& result_dst,
                                 std::vector<float>&& result_score,
                                 test_jaccard_t const& test_functor);

template void similarity_compare(int32_t num_vertices,
                                 std::vector<int32_t>&& src,
                                 std::vector<int32_t>&& dst,
                                 std::optional<std::vector<float>>&& wgt,
                                 std::vector<int32_t>&& result_src,
                                 std::vector<int32_t>&& result_dst,
                                 std::vector<float>&& result_score,
                                 test_sorensen_t const& test_functor);

template void similarity_compare(int32_t num_vertices,
                                 std::vector<int32_t>&& src,
                                 std::vector<int32_t>&& dst,
                                 std::optional<std::vector<float>>&& wgt,
                                 std::vector<int32_t>&& result_src,
                                 std::vector<int32_t>&& result_dst,
                                 std::vector<float>&& result_score,
                                 test_overlap_t const& test_functor);

template void similarity_compare(int64_t num_vertices,
                                 std::vector<int64_t>&& src,
                                 std::vector<int64_t>&& dst,
                                 std::optional<std::vector<float>>&& wgt,
                                 std::vector<int64_t>&& result_src,
                                 std::vector<int64_t>&& result_dst,
                                 std::vector<float>&& result_score,
                                 test_jaccard_t const& test_functor);

template void similarity_compare(int64_t num_vertices,
                                 std::vector<int64_t>&& src,
                                 std::vector<int64_t>&& dst,
                                 std::optional<std::vector<float>>&& wgt,
                                 std::vector<int64_t>&& result_src,
                                 std::vector<int64_t>&& result_dst,
                                 std::vector<float>&& result_score,
                                 test_sorensen_t const& test_functor);

template void similarity_compare(int64_t num_vertices,
                                 std::vector<int64_t>&& src,
                                 std::vector<int64_t>&& dst,
                                 std::optional<std::vector<float>>&& wgt,
                                 std::vector<int64_t>&& result_src,
                                 std::vector<int64_t>&& result_dst,
                                 std::vector<float>&& result_score,
                                 test_overlap_t const& test_functor);

}  // namespace test
}  // namespace cugraph
