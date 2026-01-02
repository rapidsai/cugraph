/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Include from source directory (target_include_directories adds src/)
#include "sampling/detail/window_edge_mask.cuh"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <vector>

namespace cugraph {
namespace test {

class WindowEdgeMaskTest : public ::testing::Test {
 protected:
  raft::handle_t handle_{};
};

// Test binary search window bounds
TEST_F(WindowEdgeMaskTest, BinarySearchBounds)
{
  using time_stamp_t = int64_t;

  // Create sorted timestamps
  std::vector<time_stamp_t> h_times = {100, 150, 200, 250, 300, 350, 400, 450, 500};
  rmm::device_uvector<time_stamp_t> d_times(h_times.size(), handle_.get_stream());
  raft::copy(d_times.data(), h_times.data(), h_times.size(), handle_.get_stream());

  // Test window [200, 400) - should include indices 2, 3, 4, 5 (times 200, 250, 300, 350)
  auto [start_idx, end_idx] = cugraph::detail::compute_window_bounds_binary_search<time_stamp_t>(
    handle_,
    d_times.data(),
    d_times.size(),
    200,
    400);

  handle_.sync_stream();

  EXPECT_EQ(start_idx, 2);  // First edge with time >= 200
  EXPECT_EQ(end_idx, 6);    // First edge with time >= 400
}

// Test binary search edge cases
TEST_F(WindowEdgeMaskTest, BinarySearchEdgeCases)
{
  using time_stamp_t = int64_t;

  std::vector<time_stamp_t> h_times = {100, 200, 300, 400, 500};
  rmm::device_uvector<time_stamp_t> d_times(h_times.size(), handle_.get_stream());
  raft::copy(d_times.data(), h_times.data(), h_times.size(), handle_.get_stream());

  // Test window at start
  {
    auto [start_idx, end_idx] = cugraph::detail::compute_window_bounds_binary_search<time_stamp_t>(
      handle_, d_times.data(), d_times.size(), 0, 150);
    handle_.sync_stream();
    EXPECT_EQ(start_idx, 0);
    EXPECT_EQ(end_idx, 1);  // Only edge with time 100
  }

  // Test window at end
  {
    auto [start_idx, end_idx] = cugraph::detail::compute_window_bounds_binary_search<time_stamp_t>(
      handle_, d_times.data(), d_times.size(), 450, 600);
    handle_.sync_stream();
    EXPECT_EQ(start_idx, 4);
    EXPECT_EQ(end_idx, 5);  // Only edge with time 500
  }

  // Test empty window
  {
    auto [start_idx, end_idx] = cugraph::detail::compute_window_bounds_binary_search<time_stamp_t>(
      handle_, d_times.data(), d_times.size(), 150, 200);
    handle_.sync_stream();
    EXPECT_EQ(start_idx, end_idx);  // No edges in range [150, 200)
  }
}

// Test set_mask_from_sorted_range
TEST_F(WindowEdgeMaskTest, SortedRangeMask)
{
  using edge_t = int32_t;

  // 10 edges, sorted indices: [3, 7, 1, 9, 0, 2, 8, 5, 4, 6]
  // (i.e., edge 3 has smallest time, edge 7 has second smallest, etc.)
  std::vector<edge_t> h_sorted_indices = {3, 7, 1, 9, 0, 2, 8, 5, 4, 6};
  rmm::device_uvector<edge_t> d_sorted_indices(h_sorted_indices.size(), handle_.get_stream());
  raft::copy(d_sorted_indices.data(), h_sorted_indices.data(), h_sorted_indices.size(), handle_.get_stream());

  // Create mask (10 edges = 1 word)
  rmm::device_uvector<uint32_t> d_mask(1, handle_.get_stream());

  // Set mask for sorted range [2, 5) - includes edges at sorted positions 2,3,4
  // which are original edge indices 1, 9, 0
  cugraph::detail::set_mask_from_sorted_range<edge_t>(
    handle_,
    d_mask.data(),
    static_cast<edge_t>(10),
    d_sorted_indices.data(),
    2,
    5);

  handle_.sync_stream();

  // Verify mask - bits 0, 1, 9 should be set
  uint32_t h_mask;
  raft::copy(&h_mask, d_mask.data(), 1, handle_.get_stream());
  handle_.sync_stream();

  EXPECT_TRUE(h_mask & (1u << 0));  // Edge 0
  EXPECT_TRUE(h_mask & (1u << 1));  // Edge 1
  EXPECT_TRUE(h_mask & (1u << 9));  // Edge 9
  EXPECT_FALSE(h_mask & (1u << 3)); // Edge 3 (outside range)
  EXPECT_FALSE(h_mask & (1u << 7)); // Edge 7 (outside range)
  EXPECT_FALSE(h_mask & (1u << 2)); // Edge 2 (outside range)
}

// Test incremental mask update
TEST_F(WindowEdgeMaskTest, IncrementalUpdate)
{
  using edge_t = int32_t;

  // 10 edges, sorted indices
  std::vector<edge_t> h_sorted_indices = {3, 7, 1, 9, 0, 2, 8, 5, 4, 6};
  rmm::device_uvector<edge_t> d_sorted_indices(h_sorted_indices.size(), handle_.get_stream());
  raft::copy(d_sorted_indices.data(), h_sorted_indices.data(), h_sorted_indices.size(), handle_.get_stream());

  // Create initial mask with edges [2, 5) set
  // This sets bits for edges 1, 9, 0 (indices at sorted positions 2, 3, 4)
  rmm::device_uvector<uint32_t> d_mask(1, handle_.get_stream());
  cugraph::detail::set_mask_from_sorted_range<edge_t>(
    handle_,
    d_mask.data(),
    static_cast<edge_t>(10),
    d_sorted_indices.data(),
    2,
    5);

  handle_.sync_stream();

  // Verify initial state
  uint32_t h_mask_before;
  raft::copy(&h_mask_before, d_mask.data(), 1, handle_.get_stream());
  handle_.sync_stream();
  EXPECT_TRUE(h_mask_before & (1u << 0));   // Edge 0
  EXPECT_TRUE(h_mask_before & (1u << 1));   // Edge 1
  EXPECT_TRUE(h_mask_before & (1u << 9));   // Edge 9

  // Now slide window: old [2, 5) -> new [3, 6)
  // Leaving: sorted position 2 (edge index 1)
  // Entering: sorted position 5 (edge index 2)
  cugraph::detail::update_mask_incremental<edge_t>(
    handle_,
    d_mask.data(),
    d_sorted_indices.data(),
    2, 3,  // leaving: position 2 (edge 1)
    5, 6); // entering: position 5 (edge 2)

  handle_.sync_stream();

  // Verify mask after update
  uint32_t h_mask_after;
  raft::copy(&h_mask_after, d_mask.data(), 1, handle_.get_stream());
  handle_.sync_stream();

  EXPECT_TRUE(h_mask_after & (1u << 0));   // Edge 0 (still in window)
  EXPECT_FALSE(h_mask_after & (1u << 1));  // Edge 1 (left window)
  EXPECT_TRUE(h_mask_after & (1u << 2));   // Edge 2 (entered window)
  EXPECT_TRUE(h_mask_after & (1u << 9));   // Edge 9 (still in window)
}

// Test multiple words in mask
TEST_F(WindowEdgeMaskTest, MultiWordMask)
{
  using edge_t = int64_t;

  // 100 edges spanning 4 mask words
  const size_t num_edges = 100;
  std::vector<edge_t> h_sorted_indices(num_edges);
  std::iota(h_sorted_indices.begin(), h_sorted_indices.end(), 0);
  // Shuffle to simulate non-sequential edge order
  std::mt19937 gen(42);
  std::shuffle(h_sorted_indices.begin(), h_sorted_indices.end(), gen);

  rmm::device_uvector<edge_t> d_sorted_indices(num_edges, handle_.get_stream());
  raft::copy(d_sorted_indices.data(), h_sorted_indices.data(), num_edges, handle_.get_stream());

  // Create mask
  size_t num_mask_words = (num_edges + 31) / 32;
  rmm::device_uvector<uint32_t> d_mask(num_mask_words, handle_.get_stream());

  // Set mask for range [25, 75) - 50 edges
  cugraph::detail::set_mask_from_sorted_range<edge_t>(
    handle_,
    d_mask.data(),
    static_cast<edge_t>(num_edges),
    d_sorted_indices.data(),
    25,
    75);

  handle_.sync_stream();

  // Count set bits
  std::vector<uint32_t> h_mask(num_mask_words);
  raft::copy(h_mask.data(), d_mask.data(), num_mask_words, handle_.get_stream());
  handle_.sync_stream();

  int set_count = 0;
  for (size_t i = 0; i < num_edges; ++i) {
    if (h_mask[i / 32] & (1u << (i % 32))) {
      set_count++;
    }
  }

  EXPECT_EQ(set_count, 50);  // Exactly 50 edges in window
}

// Performance test with larger data
TEST_F(WindowEdgeMaskTest, PerformanceTest)
{
  using edge_t = int64_t;
  using time_stamp_t = int64_t;

  const size_t num_edges = 1000000;  // 1M edges
  const int64_t time_range = 730 * 86400;  // 730 days in seconds
  const int64_t window_size = 365 * 86400; // 365 day window

  // Create random sorted timestamps
  std::vector<time_stamp_t> h_times(num_edges);
  std::mt19937 gen(42);
  std::uniform_int_distribution<time_stamp_t> dist(0, time_range);
  for (auto& t : h_times) { t = dist(gen); }
  std::sort(h_times.begin(), h_times.end());

  rmm::device_uvector<time_stamp_t> d_times(num_edges, handle_.get_stream());
  raft::copy(d_times.data(), h_times.data(), num_edges, handle_.get_stream());

  // Create sorted indices (identity since times are already sorted)
  rmm::device_uvector<edge_t> d_sorted_indices(num_edges, handle_.get_stream());
  thrust::sequence(thrust::device.on(handle_.get_stream()),
                   d_sorted_indices.data(),
                   d_sorted_indices.data() + num_edges);

  // Create mask
  size_t num_mask_words = (num_edges + 31) / 32;
  rmm::device_uvector<uint32_t> d_mask(num_mask_words, handle_.get_stream());

  handle_.sync_stream();

  using clock = std::chrono::high_resolution_clock;
  double binary_search_time_ms = 0.0;
  double set_mask_time_ms = 0.0;
  double incremental_time_ms = 0.0;

  // Test binary search
  auto t0 = clock::now();
  auto [start_idx, end_idx] = cugraph::detail::compute_window_bounds_binary_search<time_stamp_t>(
    handle_,
    d_times.data(),
    num_edges,
    window_size,  // window_start
    time_range);  // window_end
  handle_.sync_stream();
  auto t1 = clock::now();
  binary_search_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "Binary search time: " << binary_search_time_ms << " ms" << std::endl;
  std::cout << "Window edges: " << (end_idx - start_idx) << " / " << num_edges << std::endl;

  // Test full mask set
  t0 = clock::now();
  cugraph::detail::set_mask_from_sorted_range<edge_t>(
    handle_,
    d_mask.data(),
    static_cast<edge_t>(num_edges),
    d_sorted_indices.data(),
    start_idx,
    end_idx);
  handle_.sync_stream();
  t1 = clock::now();
  set_mask_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "Set mask from range time: " << set_mask_time_ms << " ms" << std::endl;

  // Test incremental update (simulate 1-day step)
  size_t delta_edges = num_edges / 730;  // ~1 day worth
  t0 = clock::now();
  cugraph::detail::update_mask_incremental<edge_t>(
    handle_,
    d_mask.data(),
    d_sorted_indices.data(),
    start_idx, start_idx + delta_edges,  // leaving
    end_idx, std::min(end_idx + delta_edges, num_edges));     // entering
  handle_.sync_stream();
  t1 = clock::now();
  incremental_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "Incremental update time: " << incremental_time_ms << " ms" << std::endl;
  std::cout << "Delta edges: " << delta_edges << std::endl;

  // Verify performance expectations
  // Binary search should be < 1ms for 1M edges
  EXPECT_LT(binary_search_time_ms, 10.0);  // Allow 10ms for GPU overhead
  
  // Incremental update should be faster than full set
  EXPECT_LT(incremental_time_ms, set_mask_time_ms * 2);  // Allow some variance

  SUCCEED();
}

}  // namespace test
}  // namespace cugraph

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
