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

#include <centrality/betweenness_centrality_validate.hpp>
#include <utilities/test_utilities.hpp>

#include <thrust/sort.h>

#include <gtest/gtest.h>

namespace cugraph {
namespace test {

template <typename vertex_t, typename weight_t>
void betweenness_centrality_validate(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>& d_cugraph_vertex_ids,
  rmm::device_uvector<weight_t>& d_cugraph_results,
  std::optional<rmm::device_uvector<vertex_t>>& d_reference_vertex_ids,
  rmm::device_uvector<weight_t>& d_reference_results)
{
  auto compare_functor = cugraph::test::device_nearly_equal<weight_t>{
    weight_t{1e-3},
    weight_t{(weight_t{1} / static_cast<weight_t>(d_cugraph_results.size())) * weight_t{1e-3}}};

  EXPECT_EQ(d_cugraph_results.size(), d_reference_results.size());

  if (d_cugraph_vertex_ids) {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        d_cugraph_vertex_ids->begin(),
                        d_cugraph_vertex_ids->end(),
                        d_cugraph_results.begin());
  }

  if (d_reference_vertex_ids) {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        d_reference_vertex_ids->begin(),
                        d_reference_vertex_ids->end(),
                        d_reference_results.begin());
  }

  EXPECT_TRUE(thrust::equal(handle.get_thrust_policy(),
                            d_cugraph_results.begin(),
                            d_cugraph_results.end(),
                            d_reference_results.begin(),
                            compare_functor))
    << "Mismatch in centrality results";
}

template <typename vertex_t, typename weight_t>
void edge_betweenness_centrality_validate(raft::handle_t const& handle,
                                          rmm::device_uvector<vertex_t>& d_cugraph_src_vertex_ids,
                                          rmm::device_uvector<vertex_t>& d_cugraph_dst_vertex_ids,
                                          rmm::device_uvector<weight_t>& d_cugraph_results,
                                          rmm::device_uvector<vertex_t>& d_reference_src_vertex_ids,
                                          rmm::device_uvector<vertex_t>& d_reference_dst_vertex_ids,
                                          rmm::device_uvector<weight_t>& d_reference_results)
{
  auto compare_functor = cugraph::test::device_nearly_equal<weight_t>{
    weight_t{1e-3},
    weight_t{(weight_t{1} / static_cast<weight_t>(d_reference_results.size())) * weight_t{1e-3}}};

  EXPECT_EQ(d_cugraph_results.size(), d_reference_results.size());

  thrust::sort_by_key(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(d_cugraph_src_vertex_ids.begin(), d_cugraph_dst_vertex_ids.begin()),
    thrust::make_zip_iterator(d_cugraph_src_vertex_ids.end(), d_cugraph_dst_vertex_ids.begin()),
    d_cugraph_results.begin());

  thrust::sort_by_key(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(d_reference_src_vertex_ids.begin(),
                              d_reference_dst_vertex_ids.begin()),
    thrust::make_zip_iterator(d_reference_src_vertex_ids.end(), d_reference_dst_vertex_ids.begin()),
    d_reference_results.begin());

  EXPECT_TRUE(thrust::equal(handle.get_thrust_policy(),
                            d_cugraph_results.begin(),
                            d_cugraph_results.end(),
                            d_reference_results.begin(),
                            compare_functor))
    << "Mismatch in centrality results";
}

template void betweenness_centrality_validate(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>& d_cugraph_vertex_ids,
  rmm::device_uvector<float>& d_cugraph_results,
  std::optional<rmm::device_uvector<int32_t>>& d_reference_vertex_ids,
  rmm::device_uvector<float>& d_reference_results);

template void betweenness_centrality_validate(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>& d_cugraph_vertex_ids,
  rmm::device_uvector<float>& d_cugraph_results,
  std::optional<rmm::device_uvector<int64_t>>& d_reference_vertex_ids,
  rmm::device_uvector<float>& d_reference_results);

template void edge_betweenness_centrality_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_cugraph_src_vertex_ids,
  rmm::device_uvector<int32_t>& d_cugraph_dst_vertex_ids,
  rmm::device_uvector<float>& d_cugraph_results,
  rmm::device_uvector<int32_t>& d_reference_src_vertex_ids,
  rmm::device_uvector<int32_t>& d_reference_dst_vertex_ids,
  rmm::device_uvector<float>& d_reference_results);
template void edge_betweenness_centrality_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_cugraph_src_vertex_ids,
  rmm::device_uvector<int64_t>& d_cugraph_dst_vertex_ids,
  rmm::device_uvector<float>& d_cugraph_results,
  rmm::device_uvector<int64_t>& d_reference_src_vertex_ids,
  rmm::device_uvector<int64_t>& d_reference_dst_vertex_ids,
  rmm::device_uvector<float>& d_reference_results);

}  // namespace test
}  // namespace cugraph
