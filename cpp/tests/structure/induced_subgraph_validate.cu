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
#include <structure/induced_subgraph_validate.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <thrust/binary_search.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

template <typename vertex_t, typename weight_t>
void induced_subgraph_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_cugraph_subgraph_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_cugraph_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_cugraph_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_cugraph_subgraph_edge_offsets,
  rmm::device_uvector<vertex_t>& d_reference_subgraph_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_reference_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_reference_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_reference_subgraph_edge_offsets)
{
  ASSERT_EQ(d_reference_subgraph_edge_offsets.size(), d_cugraph_subgraph_edge_offsets.size())
    << "Returned subgraph edge offset vector has an invalid size.";

  ASSERT_TRUE(thrust::equal(handle.get_thrust_policy(),
                            d_reference_subgraph_edge_offsets.begin(),
                            d_reference_subgraph_edge_offsets.end(),
                            d_cugraph_subgraph_edge_offsets.begin()))
    << "Returned subgraph edge offset values do not match with the reference values.";
  ASSERT_EQ(d_cugraph_subgraph_edgelist_weights.has_value(),
            d_reference_subgraph_edgelist_weights.has_value());

  // FIXME: This might be more efficient if we could do a segmented sort on the subgraphs.
  rmm::device_uvector<vertex_t> d_subgraph_index(d_cugraph_subgraph_edgelist_majors.size(),
                                                 handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    d_subgraph_index.begin(),
    d_subgraph_index.end(),
    [subgraph_edge_offsets_begin = d_cugraph_subgraph_edge_offsets.begin(),
     subgraph_edge_offsets_end = d_cugraph_subgraph_edge_offsets.begin()] __device__(vertex_t idx) {
      auto offset = thrust::upper_bound(
        thrust::device, subgraph_edge_offsets_begin, subgraph_edge_offsets_end, idx);
      return static_cast<vertex_t>(thrust::distance(subgraph_edge_offsets_begin, offset));
    });

  if (d_reference_subgraph_edgelist_weights) {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        thrust::make_zip_iterator(d_subgraph_index.begin(),
                                                  d_reference_subgraph_edgelist_majors.begin(),
                                                  d_reference_subgraph_edgelist_minors.begin()),
                        thrust::make_zip_iterator(d_subgraph_index.end(),
                                                  d_reference_subgraph_edgelist_majors.end(),
                                                  d_reference_subgraph_edgelist_minors.end()),
                        d_reference_subgraph_edgelist_weights->begin());
    thrust::sort_by_key(handle.get_thrust_policy(),
                        thrust::make_zip_iterator(d_subgraph_index.begin(),
                                                  d_cugraph_subgraph_edgelist_majors.begin(),
                                                  d_cugraph_subgraph_edgelist_minors.begin()),
                        thrust::make_zip_iterator(d_subgraph_index.end(),
                                                  d_cugraph_subgraph_edgelist_majors.end(),
                                                  d_cugraph_subgraph_edgelist_minors.end()),
                        d_cugraph_subgraph_edgelist_weights->begin());

    ASSERT_TRUE(
      thrust::equal(handle.get_thrust_policy(),
                    thrust::make_zip_iterator(d_reference_subgraph_edgelist_majors.begin(),
                                              d_reference_subgraph_edgelist_minors.begin(),
                                              d_reference_subgraph_edgelist_weights->begin()),
                    thrust::make_zip_iterator(d_reference_subgraph_edgelist_majors.end(),
                                              d_reference_subgraph_edgelist_minors.end(),
                                              d_reference_subgraph_edgelist_weights->end()),
                    thrust::make_zip_iterator(d_cugraph_subgraph_edgelist_majors.begin(),
                                              d_cugraph_subgraph_edgelist_minors.begin(),
                                              d_cugraph_subgraph_edgelist_weights->begin()),
                    [] __device__(auto left, auto right) {
                      auto l0 = thrust::get<0>(left);
                      auto l1 = thrust::get<1>(left);
                      auto l2 = thrust::get<2>(left);
                      auto r0 = thrust::get<0>(right);
                      auto r1 = thrust::get<1>(right);
                      auto r2 = thrust::get<2>(right);
                      return (l0 == r0) && (l1 == r1) && (l2 == r2);
                    }))
      << "Extracted subgraph edges do not match with the edges extracted by the reference "
         "implementation.";
  } else {
    thrust::sort(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(d_subgraph_index.begin(),
                                           d_reference_subgraph_edgelist_majors.begin(),
                                           d_reference_subgraph_edgelist_minors.begin()),
                 thrust::make_zip_iterator(d_subgraph_index.end(),
                                           d_reference_subgraph_edgelist_majors.end(),
                                           d_reference_subgraph_edgelist_minors.end()));
    thrust::sort(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(d_subgraph_index.begin(),
                                           d_cugraph_subgraph_edgelist_majors.begin(),
                                           d_cugraph_subgraph_edgelist_minors.begin()),
                 thrust::make_zip_iterator(d_subgraph_index.end(),
                                           d_cugraph_subgraph_edgelist_majors.end(),
                                           d_cugraph_subgraph_edgelist_minors.end()));

    ASSERT_TRUE(
      thrust::equal(handle.get_thrust_policy(),
                    thrust::make_zip_iterator(d_reference_subgraph_edgelist_majors.begin(),
                                              d_reference_subgraph_edgelist_minors.begin()),
                    thrust::make_zip_iterator(d_reference_subgraph_edgelist_majors.end(),
                                              d_reference_subgraph_edgelist_minors.end()),
                    thrust::make_zip_iterator(d_cugraph_subgraph_edgelist_majors.begin(),
                                              d_cugraph_subgraph_edgelist_minors.begin()),
                    [] __device__(auto left, auto right) {
                      auto l0 = thrust::get<0>(left);
                      auto l1 = thrust::get<1>(left);
                      auto r0 = thrust::get<0>(right);
                      auto r1 = thrust::get<1>(right);
                      return (l0 == r0) && (l1 == r1);
                    }))
      << "Extracted subgraph edges do not match with the edges extracted by the reference "
         "implementation.";
  }
}

template void induced_subgraph_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_cugraph_subgraph_edgelist_majors,
  rmm::device_uvector<int32_t>& d_cugraph_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_cugraph_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_cugraph_subgraph_edge_offsets,
  rmm::device_uvector<int32_t>& d_reference_subgraph_edgelist_majors,
  rmm::device_uvector<int32_t>& d_reference_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_reference_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_reference_subgraph_edge_offsets);

template void induced_subgraph_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_cugraph_subgraph_edgelist_majors,
  rmm::device_uvector<int64_t>& d_cugraph_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_cugraph_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_cugraph_subgraph_edge_offsets,
  rmm::device_uvector<int64_t>& d_reference_subgraph_edgelist_majors,
  rmm::device_uvector<int64_t>& d_reference_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_reference_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_reference_subgraph_edge_offsets);
