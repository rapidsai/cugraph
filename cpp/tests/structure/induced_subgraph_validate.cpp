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

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <raft/span.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<vertex_t>,
           std::vector<vertex_t>,
           std::optional<std::vector<weight_t>>,
           std::vector<size_t>>
extract_induced_subgraph_reference(edge_t const* offsets,
                                   vertex_t const* indices,
                                   std::optional<weight_t const*> weights,
                                   size_t const* subgraph_offsets,
                                   vertex_t const* subgraph_vertices,
                                   size_t num_vertices,
                                   size_t num_subgraphs)
{
  std::vector<vertex_t> edgelist_majors{};
  std::vector<vertex_t> edgelist_minors{};
  auto edgelist_weights = weights ? std::make_optional<std::vector<weight_t>>(0) : std::nullopt;
  std::vector<size_t> subgraph_edge_offsets{0};

  for (size_t i = 0; i < num_subgraphs; ++i) {
    std::for_each(subgraph_vertices + subgraph_offsets[i],
                  subgraph_vertices + subgraph_offsets[i + 1],
                  [offsets,
                   indices,
                   weights,
                   subgraph_vertices,
                   subgraph_offsets,
                   &edgelist_majors,
                   &edgelist_minors,
                   &edgelist_weights,
                   i](auto v) {
                    auto first = offsets[v];
                    auto last  = offsets[v + 1];
                    for (auto j = first; j < last; ++j) {
                      if (std::binary_search(subgraph_vertices + subgraph_offsets[i],
                                             subgraph_vertices + subgraph_offsets[i + 1],
                                             indices[j])) {
                        edgelist_majors.push_back(v);
                        edgelist_minors.push_back(indices[j]);
                        if (weights) { (*edgelist_weights).push_back((*weights)[j]); }
                      }
                    }
                  });
    subgraph_edge_offsets.push_back(edgelist_majors.size());
  }

  return std::make_tuple(edgelist_majors, edgelist_minors, edgelist_weights, subgraph_edge_offsets);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void induced_subgraph_validate(
  std::vector<edge_t> const& h_offsets,
  std::vector<vertex_t> const& h_indices,
  std::optional<std::vector<weight_t>> const& h_weights,
  std::vector<size_t> const& h_subgraph_offsets,
  std::vector<vertex_t> const& h_subgraph_vertices,
  std::vector<vertex_t> const& h_cugraph_subgraph_edgelist_majors,
  std::vector<vertex_t> const& h_cugraph_subgraph_edgelist_minors,
  std::optional<std::vector<weight_t>> const& h_cugraph_subgraph_edgelist_weights,
  std::vector<size_t> const& h_cugraph_subgraph_edge_offsets)
{
  auto [h_reference_subgraph_edgelist_majors,
        h_reference_subgraph_edgelist_minors,
        h_reference_subgraph_edgelist_weights,
        h_reference_subgraph_edge_offsets] =
    extract_induced_subgraph_reference(
      h_offsets.data(),
      h_indices.data(),
      h_weights ? std::optional<weight_t const*>{(*h_weights).data()} : std::nullopt,
      h_subgraph_offsets.data(),
      h_subgraph_vertices.data(),
      h_offsets.size() - 1,
      h_subgraph_offsets.size() - 1);

  ASSERT_TRUE(h_reference_subgraph_edge_offsets.size() == h_cugraph_subgraph_edge_offsets.size())
    << "Returned subgraph edge offset vector has an invalid size.";
  ASSERT_TRUE(std::equal(h_reference_subgraph_edge_offsets.begin(),
                         h_reference_subgraph_edge_offsets.end(),
                         h_cugraph_subgraph_edge_offsets.begin()))
    << "Returned subgraph edge offset values do not match with the reference values.";
  ASSERT_TRUE(h_reference_subgraph_edgelist_weights.has_value() == h_weights.has_value());
  ASSERT_TRUE(h_cugraph_subgraph_edgelist_weights.has_value() ==
              h_reference_subgraph_edgelist_weights.has_value());

  for (size_t i = 0; i < (h_reference_subgraph_edge_offsets.size() - 1); ++i) {
    auto start = h_reference_subgraph_edge_offsets[i];
    auto last  = h_reference_subgraph_edge_offsets[i + 1];
    if (h_weights) {
      std::vector<std::tuple<vertex_t, vertex_t, weight_t>> reference_tuples(last - start);
      std::vector<std::tuple<vertex_t, vertex_t, weight_t>> cugraph_tuples(last - start);
      for (auto j = start; j < last; ++j) {
        reference_tuples[j - start] = std::make_tuple(h_reference_subgraph_edgelist_majors[j],
                                                      h_reference_subgraph_edgelist_minors[j],
                                                      (*h_reference_subgraph_edgelist_weights)[j]);
        cugraph_tuples[j - start]   = std::make_tuple(h_cugraph_subgraph_edgelist_majors[j],
                                                    h_cugraph_subgraph_edgelist_minors[j],
                                                    (*h_cugraph_subgraph_edgelist_weights)[j]);
      }
      ASSERT_TRUE(
        std::equal(reference_tuples.begin(), reference_tuples.end(), cugraph_tuples.begin()))
        << "Extracted subgraph edges do not match with the edges extracted by the reference "
           "implementation.";
    } else {
      std::vector<std::tuple<vertex_t, vertex_t>> reference_tuples(last - start);
      std::vector<std::tuple<vertex_t, vertex_t>> cugraph_tuples(last - start);
      for (auto j = start; j < last; ++j) {
        reference_tuples[j - start] = std::make_tuple(h_reference_subgraph_edgelist_majors[j],
                                                      h_reference_subgraph_edgelist_minors[j]);
        cugraph_tuples[j - start]   = std::make_tuple(h_cugraph_subgraph_edgelist_majors[j],
                                                    h_cugraph_subgraph_edgelist_minors[j]);
      }
      ASSERT_TRUE(
        std::equal(reference_tuples.begin(), reference_tuples.end(), cugraph_tuples.begin()))
        << "Extracted subgraph edges do not match with the edges extracted by the reference "
           "implementation.";
    }
  }
}

template void induced_subgraph_validate(
  std::vector<int32_t> const& h_offsets,
  std::vector<int32_t> const& h_indices,
  std::optional<std::vector<float>> const& h_weights,
  std::vector<size_t> const& h_subgraph_offsets,
  std::vector<int32_t> const& h_subgraph_vertices,
  std::vector<int32_t> const& h_cugraph_subgraph_edgelist_majors,
  std::vector<int32_t> const& h_cugraph_subgraph_edgelist_minors,
  std::optional<std::vector<float>> const& h_cugraph_subgraph_edgelist_weights,
  std::vector<size_t> const& h_cugraph_subgraph_edge_offsets);

template void induced_subgraph_validate(
  std::vector<int64_t> const& h_offsets,
  std::vector<int32_t> const& h_indices,
  std::optional<std::vector<float>> const& h_weights,
  std::vector<size_t> const& h_subgraph_offsets,
  std::vector<int32_t> const& h_subgraph_vertices,
  std::vector<int32_t> const& h_cugraph_subgraph_edgelist_majors,
  std::vector<int32_t> const& h_cugraph_subgraph_edgelist_minors,
  std::optional<std::vector<float>> const& h_cugraph_subgraph_edgelist_weights,
  std::vector<size_t> const& h_cugraph_subgraph_edge_offsets);

template void induced_subgraph_validate(
  std::vector<int64_t> const& h_offsets,
  std::vector<int64_t> const& h_indices,
  std::optional<std::vector<float>> const& h_weights,
  std::vector<size_t> const& h_subgraph_offsets,
  std::vector<int64_t> const& h_subgraph_vertices,
  std::vector<int64_t> const& h_cugraph_subgraph_edgelist_majors,
  std::vector<int64_t> const& h_cugraph_subgraph_edgelist_minors,
  std::optional<std::vector<float>> const& h_cugraph_subgraph_edgelist_weights,
  std::vector<size_t> const& h_cugraph_subgraph_edge_offsets);
