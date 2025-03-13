/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

typedef struct RemoveMultiEdges_Usecase_t {
  int num_chunks{1};
  bool keep_min_value_edge{false};
  bool check_correctness{true};
} RemoveMultiEdges_Usecase;

template <typename value_t>
rmm::device_uvector<value_t> aggregate_buffers(raft::handle_t const handle,
                                               std::vector<rmm::device_uvector<value_t>>&& chunks)
{
  size_t tot_edge_count{0};
  for (size_t i = 0; i < chunks.size(); ++i) {
    tot_edge_count += chunks[i].size();
  }

  rmm::device_uvector<value_t> values(tot_edge_count, handle.get_stream());
  size_t offset{0};
  for (size_t i = 0; i < chunks.size(); ++i) {
    raft::copy_async(
      values.data() + offset, chunks[i].data(), chunks[i].size(), handle.get_stream());
    offset += chunks[i].size();
  }
  return values;
}

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<edge_t>,
           rmm::device_uvector<edge_type_t>>
aggregate_buffers(raft::handle_t const handle,
                  std::vector<rmm::device_uvector<vertex_t>>&& src_chunks,
                  std::vector<rmm::device_uvector<vertex_t>>&& dst_chunks,
                  std::vector<rmm::device_uvector<weight_t>>&& weight_chunks,
                  std::vector<rmm::device_uvector<edge_t>>&& edge_id_chunks,
                  std::vector<rmm::device_uvector<edge_type_t>>&& edge_type_chunks)
{
  size_t tot_edge_count{0};
  for (size_t i = 0; i < src_chunks.size(); ++i) {
    tot_edge_count += src_chunks[i].size();
  }
  rmm::device_uvector<vertex_t> srcs(tot_edge_count, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(tot_edge_count, handle.get_stream());
  rmm::device_uvector<weight_t> weights(tot_edge_count, handle.get_stream());
  rmm::device_uvector<edge_t> edge_ids(tot_edge_count, handle.get_stream());
  rmm::device_uvector<edge_type_t> edge_types(tot_edge_count, handle.get_stream());

  size_t offset{0};
  for (size_t i = 0; i < src_chunks.size(); ++i) {
    raft::copy_async(
      srcs.data() + offset, src_chunks[i].data(), src_chunks[i].size(), handle.get_stream());
    raft::copy_async(
      dsts.data() + offset, dst_chunks[i].data(), dst_chunks[i].size(), handle.get_stream());
    raft::copy_async(weights.data() + offset,
                     weight_chunks[i].data(),
                     weight_chunks[i].size(),
                     handle.get_stream());
    raft::copy_async(edge_ids.data() + offset,
                     edge_id_chunks[i].data(),
                     edge_id_chunks[i].size(),
                     handle.get_stream());
    raft::copy_async(edge_types.data() + offset,
                     edge_type_chunks[i].data(),
                     edge_type_chunks[i].size(),
                     handle.get_stream());
    offset += src_chunks[i].size();
  }
  src_chunks.clear();
  dst_chunks.clear();
  weight_chunks.clear();
  edge_id_chunks.clear();
  edge_type_chunks.clear();

  return std::make_tuple(std::move(srcs),
                         std::move(dsts),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types));
}

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<std::vector<rmm::device_uvector<vertex_t>>,
           std::vector<rmm::device_uvector<vertex_t>>,
           std::vector<rmm::device_uvector<weight_t>>,
           std::vector<rmm::device_uvector<edge_t>>,
           std::vector<rmm::device_uvector<edge_type_t>>>
split_buffer(raft::handle_t const handle,
             raft::device_span<vertex_t const> srcs,
             raft::device_span<vertex_t const> dsts,
             raft::device_span<weight_t const> weights,
             raft::device_span<edge_t const> edge_ids,
             raft::device_span<edge_type_t const> edge_types,
             size_t num_chunks)
{
  std::vector<rmm::device_uvector<vertex_t>> src_chunks{};
  std::vector<rmm::device_uvector<vertex_t>> dst_chunks{};
  std::vector<rmm::device_uvector<weight_t>> weight_chunks{};
  std::vector<rmm::device_uvector<edge_t>> edge_id_chunks{};
  std::vector<rmm::device_uvector<edge_type_t>> edge_type_chunks{};

  src_chunks.reserve(num_chunks);
  dst_chunks.reserve(num_chunks);
  weight_chunks.reserve(num_chunks);
  edge_id_chunks.reserve(num_chunks);
  edge_type_chunks.reserve(num_chunks);

  for (size_t i = 0; i < num_chunks; ++i) {
    src_chunks.emplace_back(0, handle.get_stream());
    dst_chunks.emplace_back(0, handle.get_stream());
    weight_chunks.emplace_back(0, handle.get_stream());
    edge_id_chunks.emplace_back(0, handle.get_stream());
    edge_type_chunks.emplace_back(0, handle.get_stream());
  }

  size_t chunk_size = (srcs.size() + (num_chunks - 1)) / num_chunks;
  for (size_t i = 0; i < num_chunks; ++i) {
    auto start_offset = chunk_size * i;
    auto end_offset   = std::min(chunk_size * (i + 1), srcs.size());
    src_chunks[i].resize(end_offset - start_offset, handle.get_stream());
    raft::copy_async(src_chunks[i].data(),
                     srcs.data() + start_offset,
                     end_offset - start_offset,
                     handle.get_stream());
    dst_chunks[i].resize(end_offset - start_offset, handle.get_stream());
    raft::copy_async(dst_chunks[i].data(),
                     dsts.data() + start_offset,
                     end_offset - start_offset,
                     handle.get_stream());
    weight_chunks[i].resize(end_offset - start_offset, handle.get_stream());
    raft::copy_async(weight_chunks[i].data(),
                     weights.data() + start_offset,
                     end_offset - start_offset,
                     handle.get_stream());
    edge_id_chunks[i].resize(end_offset - start_offset, handle.get_stream());
    raft::copy_async(edge_id_chunks[i].data(),
                     edge_ids.data() + start_offset,
                     end_offset - start_offset,
                     handle.get_stream());
    edge_type_chunks[i].resize(end_offset - start_offset, handle.get_stream());
    raft::copy_async(edge_type_chunks[i].data(),
                     edge_types.data() + start_offset,
                     end_offset - start_offset,
                     handle.get_stream());
  }

  return std::make_tuple(std::move(src_chunks),
                         std::move(dst_chunks),
                         std::move(weight_chunks),
                         std::move(edge_id_chunks),
                         std::move(edge_type_chunks));
}

template <typename input_usecase_t>
class Tests_RemoveMultiEdges
  : public ::testing::TestWithParam<std::tuple<RemoveMultiEdges_Usecase, input_usecase_t>> {
 public:
  Tests_RemoveMultiEdges() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(
    std::tuple<RemoveMultiEdges_Usecase const&, input_usecase_t const&> const& param)
  {
    using weight_t    = float;
    using edge_type_t = int32_t;
    using edge_time_t = int32_t;

    bool constexpr store_transposed = false;
    bool constexpr multi_gpu        = false;
    bool constexpr test_weighted    = true;
    bool constexpr shuffle          = false;  // irrelevant if multi_gpu = false

    auto [remove_multi_edges_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};
    raft::random::RngState rng_state(0);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct edge list");
    }

    std::vector<rmm::device_uvector<vertex_t>> src_chunks{};
    std::vector<rmm::device_uvector<vertex_t>> dst_chunks{};
    std::vector<rmm::device_uvector<weight_t>> weight_chunks{};
    std::vector<rmm::device_uvector<edge_t>> edge_id_chunks{};
    std::vector<rmm::device_uvector<edge_type_t>> edge_type_chunks{};

    {
      std::optional<std::vector<rmm::device_uvector<weight_t>>> tmp_weight_chunks{};
      std::tie(src_chunks, dst_chunks, tmp_weight_chunks, std::ignore, std::ignore) =
        input_usecase.template construct_edgelist<vertex_t, weight_t>(
          handle, test_weighted, store_transposed, multi_gpu);
      weight_chunks = std::move(*tmp_weight_chunks);
    }

    edge_id_chunks.reserve(src_chunks.size());
    edge_type_chunks.reserve(src_chunks.size());
    for (size_t i = 0; i < src_chunks.size(); ++i) {
      edge_id_chunks.emplace_back(src_chunks[i].size(), handle.get_stream());
      edge_type_chunks.emplace_back(src_chunks[i].size(), handle.get_stream());
    }
    for (size_t i = 0; i < src_chunks.size(); ++i) {
      edge_id_chunks[i].resize(src_chunks[i].size(), handle.get_stream());
      edge_type_chunks[i].resize(src_chunks[i].size(), handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           edge_id_chunks[i].data(),
                                           edge_id_chunks[i].size(),
                                           edge_t{0},
                                           std::numeric_limits<edge_t>::max(),
                                           rng_state);
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           edge_type_chunks[i].data(),
                                           edge_type_chunks[i].size(),
                                           edge_type_t{0},
                                           std::numeric_limits<edge_type_t>::max(),
                                           rng_state);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto [org_srcs, org_dsts, org_weights, org_edge_ids, org_edge_types] =
      aggregate_buffers(handle,
                        std::move(src_chunks),
                        std::move(dst_chunks),
                        std::move(weight_chunks),
                        std::move(edge_id_chunks),
                        std::move(edge_type_chunks));

    rmm::device_uvector<vertex_t> result_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> result_dsts(0, handle.get_stream());
    rmm::device_uvector<weight_t> result_weights(0, handle.get_stream());
    rmm::device_uvector<edge_t> result_edge_ids(0, handle.get_stream());
    rmm::device_uvector<edge_type_t> result_edge_types(0, handle.get_stream());

    std::tie(src_chunks, dst_chunks, weight_chunks, edge_id_chunks, edge_type_chunks) =
      split_buffer(
        handle,
        raft::device_span<vertex_t const>(org_srcs.data(), org_srcs.size()),
        raft::device_span<vertex_t const>(org_dsts.data(), org_dsts.size()),
        raft::device_span<weight_t const>(org_weights.data(), org_weights.size()),
        raft::device_span<edge_t const>(org_edge_ids.data(), org_edge_ids.size()),
        raft::device_span<edge_type_t const>(org_edge_types.data(), org_edge_types.size()),
        remove_multi_edges_usecase.num_chunks);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Remove multi-edges (just src, dst)");
    }

    if (src_chunks.size() == 1) {
      std::tie(
        result_srcs, result_dsts, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          handle,
          std::move(src_chunks[0]),
          std::move(dst_chunks[0]),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          remove_multi_edges_usecase.keep_min_value_edge);
    } else {
      auto [result_src_chunks,
            result_dst_chunks,
            result_weight_chunks /* dummy */,
            result_edge_id_chunks /* dummy */,
            result_edge_type_chunks /* dummy */,
            result_edge_start_time_chunks /* dummy */,
            result_edge_end_time_chunks /* dummy */] =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          handle,
          std::move(src_chunks),
          std::move(dst_chunks),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          remove_multi_edges_usecase.keep_min_value_edge);
      result_srcs = aggregate_buffers(handle, std::move(result_src_chunks));
      result_dsts = aggregate_buffers(handle, std::move(result_dst_chunks));
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (remove_multi_edges_usecase.check_correctness) {
      auto h_org_srcs = cugraph::test::to_host(handle, org_srcs);
      auto h_org_dsts = cugraph::test::to_host(handle, org_dsts);
      std::vector<std::tuple<vertex_t, vertex_t>> h_org_edges(h_org_srcs.size());
      for (size_t i = 0; i < h_org_srcs.size(); ++i) {
        h_org_edges[i] = std::make_tuple(h_org_srcs[i], h_org_dsts[i]);
      }
      std::sort(h_org_edges.begin(), h_org_edges.end());
      h_org_edges.resize(
        std::distance(h_org_edges.begin(), std::unique(h_org_edges.begin(), h_org_edges.end())));

      auto h_result_srcs = cugraph::test::to_host(handle, result_srcs);
      auto h_result_dsts = cugraph::test::to_host(handle, result_dsts);
      std::vector<std::tuple<vertex_t, vertex_t>> h_result_edges(h_result_srcs.size());
      for (size_t i = 0; i < h_result_srcs.size(); ++i) {
        h_result_edges[i] = std::make_tuple(h_result_srcs[i], h_result_dsts[i]);
      }
      std::sort(h_result_edges.begin(), h_result_edges.end());

      ASSERT_EQ(h_org_edges.size(), h_result_edges.size());
      ASSERT_TRUE(std::equal(h_org_edges.begin(), h_org_edges.end(), h_result_edges.begin()));
    }

    std::tie(src_chunks, dst_chunks, weight_chunks, edge_id_chunks, edge_type_chunks) =
      split_buffer(
        handle,
        raft::device_span<vertex_t const>(org_srcs.data(), org_srcs.size()),
        raft::device_span<vertex_t const>(org_dsts.data(), org_dsts.size()),
        raft::device_span<weight_t const>(org_weights.data(), org_weights.size()),
        raft::device_span<edge_t const>(org_edge_ids.data(), org_edge_ids.size()),
        raft::device_span<edge_type_t const>(org_edge_types.data(), org_edge_types.size()),
        remove_multi_edges_usecase.num_chunks);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Remove multi-edges (src, dst, weight)");
    }

    if (src_chunks.size() == 1) {
      std::optional<rmm::device_uvector<weight_t>> tmp_weights{};
      std::tie(
        result_srcs, result_dsts, tmp_weights, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          handle,
          std::move(src_chunks[0]),
          std::move(dst_chunks[0]),
          std::move(weight_chunks[0]),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          remove_multi_edges_usecase.keep_min_value_edge);
      result_weights = std::move(*tmp_weights);
    } else {
      auto [result_src_chunks,
            result_dst_chunks,
            result_weight_chunks,
            result_edge_id_chunks /* dummy */,
            result_edge_type_chunks /* dummy */,
            result_edge_start_time_chunks /* dummy */,
            result_edge_end_time_chunks /* dummy */] =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          handle,
          std::move(src_chunks),
          std::move(dst_chunks),
          std::move(weight_chunks),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          remove_multi_edges_usecase.keep_min_value_edge);
      result_srcs    = aggregate_buffers(handle, std::move(result_src_chunks));
      result_dsts    = aggregate_buffers(handle, std::move(result_dst_chunks));
      result_weights = aggregate_buffers(handle, std::move(*result_weight_chunks));
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (remove_multi_edges_usecase.check_correctness) {
      auto h_org_srcs    = cugraph::test::to_host(handle, org_srcs);
      auto h_org_dsts    = cugraph::test::to_host(handle, org_dsts);
      auto h_org_weights = cugraph::test::to_host(handle, org_weights);
      std::vector<std::tuple<vertex_t, vertex_t, weight_t>> h_org_edges(h_org_srcs.size());
      for (size_t i = 0; i < h_org_srcs.size(); ++i) {
        h_org_edges[i] = std::make_tuple(h_org_srcs[i], h_org_dsts[i], h_org_weights[i]);
      }
      std::sort(h_org_edges.begin(), h_org_edges.end());
      h_org_edges.resize(std::distance(
        h_org_edges.begin(),
        std::unique(h_org_edges.begin(), h_org_edges.end(), [](auto lhs, auto rhs) {
          return (std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs));
        })));

      auto h_result_srcs    = cugraph::test::to_host(handle, result_srcs);
      auto h_result_dsts    = cugraph::test::to_host(handle, result_dsts);
      auto h_result_weights = cugraph::test::to_host(handle, result_weights);
      std::vector<std::tuple<vertex_t, vertex_t, weight_t>> h_result_edges(h_result_srcs.size());
      for (size_t i = 0; i < h_result_srcs.size(); ++i) {
        h_result_edges[i] =
          std::make_tuple(h_result_srcs[i], h_result_dsts[i], h_result_weights[i]);
      }
      std::sort(h_result_edges.begin(), h_result_edges.end());

      ASSERT_EQ(h_org_edges.size(), h_result_edges.size());
      if (remove_multi_edges_usecase.keep_min_value_edge) {
        ASSERT_TRUE(std::equal(h_org_edges.begin(), h_org_edges.end(), h_result_edges.begin()));
      } else {
        ASSERT_TRUE(std::equal(
          h_org_edges.begin(), h_org_edges.end(), h_result_edges.begin(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) == std::get<0>(rhs) && std::get<1>(lhs) == std::get<1>(rhs);
          }));
      }
    }

    std::tie(src_chunks, dst_chunks, weight_chunks, edge_id_chunks, edge_type_chunks) =
      split_buffer(
        handle,
        raft::device_span<vertex_t const>(org_srcs.data(), org_srcs.size()),
        raft::device_span<vertex_t const>(org_dsts.data(), org_dsts.size()),
        raft::device_span<weight_t const>(org_weights.data(), org_weights.size()),
        raft::device_span<edge_t const>(org_edge_ids.data(), org_edge_ids.size()),
        raft::device_span<edge_type_t const>(org_edge_types.data(), org_edge_types.size()),
        remove_multi_edges_usecase.num_chunks);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Remove multi-edges (src, dst, weight, edge_id, edge_type)");
    }

    if (src_chunks.size() == 1) {
      std::optional<rmm::device_uvector<weight_t>> tmp_weights{};
      std::optional<rmm::device_uvector<edge_t>> tmp_edge_ids{};
      std::optional<rmm::device_uvector<edge_type_t>> tmp_edge_types{};
      std::tie(result_srcs,
               result_dsts,
               tmp_weights,
               tmp_edge_ids,
               tmp_edge_types,
               std::ignore,
               std::ignore) =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          handle,
          std::move(src_chunks[0]),
          std::move(dst_chunks[0]),
          std::move(weight_chunks[0]),
          std::move(edge_id_chunks[0]),
          std::move(edge_type_chunks[0]),
          std::nullopt,
          std::nullopt,
          remove_multi_edges_usecase.keep_min_value_edge);
      result_weights    = std::move(*tmp_weights);
      result_edge_ids   = std::move(*tmp_edge_ids);
      result_edge_types = std::move(*tmp_edge_types);
    } else {
      auto [result_src_chunks,
            result_dst_chunks,
            result_weight_chunks,
            result_edge_id_chunks,
            result_edge_type_chunks,
            result_edge_start_time_chunks /* dummy */,
            result_edge_end_time_chunks /* dummy */] =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          handle,
          std::move(src_chunks),
          std::move(dst_chunks),
          std::move(weight_chunks),
          std::move(edge_id_chunks),
          std::move(edge_type_chunks),
          std::nullopt,
          std::nullopt,
          remove_multi_edges_usecase.keep_min_value_edge);
      result_srcs       = aggregate_buffers(handle, std::move(result_src_chunks));
      result_dsts       = aggregate_buffers(handle, std::move(result_dst_chunks));
      result_weights    = aggregate_buffers(handle, std::move(*result_weight_chunks));
      result_edge_ids   = aggregate_buffers(handle, std::move(*result_edge_id_chunks));
      result_edge_types = aggregate_buffers(handle, std::move(*result_edge_type_chunks));
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (remove_multi_edges_usecase.check_correctness) {
      auto h_org_srcs       = cugraph::test::to_host(handle, org_srcs);
      auto h_org_dsts       = cugraph::test::to_host(handle, org_dsts);
      auto h_org_weights    = cugraph::test::to_host(handle, org_weights);
      auto h_org_edge_ids   = cugraph::test::to_host(handle, org_edge_ids);
      auto h_org_edge_types = cugraph::test::to_host(handle, org_edge_types);
      std::vector<std::tuple<vertex_t, vertex_t, weight_t, edge_t, edge_type_t>> h_org_edges(
        h_org_srcs.size());
      for (size_t i = 0; i < h_org_srcs.size(); ++i) {
        h_org_edges[i] = std::make_tuple(
          h_org_srcs[i], h_org_dsts[i], h_org_weights[i], h_org_edge_ids[i], h_org_edge_types[i]);
      }
      std::sort(h_org_edges.begin(), h_org_edges.end());
      h_org_edges.resize(std::distance(
        h_org_edges.begin(),
        std::unique(h_org_edges.begin(), h_org_edges.end(), [](auto lhs, auto rhs) {
          return (std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs));
        })));

      auto h_result_srcs       = cugraph::test::to_host(handle, result_srcs);
      auto h_result_dsts       = cugraph::test::to_host(handle, result_dsts);
      auto h_result_weights    = cugraph::test::to_host(handle, result_weights);
      auto h_result_edge_ids   = cugraph::test::to_host(handle, result_edge_ids);
      auto h_result_edge_types = cugraph::test::to_host(handle, result_edge_types);
      std::vector<std::tuple<vertex_t, vertex_t, weight_t, edge_t, edge_type_t>> h_result_edges(
        h_result_srcs.size());
      for (size_t i = 0; i < h_result_srcs.size(); ++i) {
        h_result_edges[i] = std::make_tuple(h_result_srcs[i],
                                            h_result_dsts[i],
                                            h_result_weights[i],
                                            h_result_edge_ids[i],
                                            h_result_edge_types[i]);
      }
      std::sort(h_result_edges.begin(), h_result_edges.end());

      ASSERT_EQ(h_org_edges.size(), h_result_edges.size());
      if (remove_multi_edges_usecase.keep_min_value_edge) {
        ASSERT_TRUE(std::equal(h_org_edges.begin(), h_org_edges.end(), h_result_edges.begin()));
      } else {
        ASSERT_TRUE(std::equal(
          h_org_edges.begin(), h_org_edges.end(), h_result_edges.begin(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) == std::get<0>(rhs) && std::get<1>(lhs) == std::get<1>(rhs);
          }));
      }
    }
  }
};

using Tests_RemoveMultiEdges_File = Tests_RemoveMultiEdges<cugraph::test::File_Usecase>;
using Tests_RemoveMultiEdges_Rmat = Tests_RemoveMultiEdges<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_RemoveMultiEdges_File, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_RemoveMultiEdges_Rmat, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_RemoveMultiEdges_Rmat, CheckInt64Int64)
{
  run_current_test<int64_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_RemoveMultiEdges_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(RemoveMultiEdges_Usecase{1, false},
                      RemoveMultiEdges_Usecase{1, true},
                      RemoveMultiEdges_Usecase{4, false},
                      RemoveMultiEdges_Usecase{4, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_RemoveMultiEdges_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(RemoveMultiEdges_Usecase{1, false},
                      RemoveMultiEdges_Usecase{1, true},
                      RemoveMultiEdges_Usecase{4, false},
                      RemoveMultiEdges_Usecase{4, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_RemoveMultiEdges_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(RemoveMultiEdges_Usecase{1, false},
                      RemoveMultiEdges_Usecase{1, true},
                      RemoveMultiEdges_Usecase{4, false},
                      RemoveMultiEdges_Usecase{4, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_RemoveMultiEdges_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(RemoveMultiEdges_Usecase{1, false, false},
                      RemoveMultiEdges_Usecase{1, true, false},
                      RemoveMultiEdges_Usecase{4, false, false},
                      RemoveMultiEdges_Usecase{4, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
