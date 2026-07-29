/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"

#include <cugraph/dynamic/memory_manager/block_array.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/tuple>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

class Tests_BlockArray : public ::testing::Test {
 public:
  static constexpr size_t k_elements_per_block = 4;
  static constexpr size_t k_num_blocks         = 4;
  static constexpr size_t k_num_elements       = k_elements_per_block * k_num_blocks;

  Tests_BlockArray() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  void run_insert_until_full_test(raft::handle_t const& handle)
  {
    cugraph::dynamic::block_array_t<int32_t> block_array(
      k_elements_per_block, k_num_blocks, handle.get_stream());

    EXPECT_NE(block_array.block_array_key(), nullptr);
    EXPECT_EQ(block_array.num_elements_per_block(), k_elements_per_block);
    EXPECT_EQ(block_array.num_blocks(), k_num_blocks);
    EXPECT_EQ(block_array.num_elements(), k_num_elements);
    EXPECT_FALSE(block_array.full());

    std::vector<size_t> block_indices;
    block_indices.reserve(k_num_blocks);
    for (size_t i = 0; i < k_num_blocks; ++i) {
      auto const block_index = block_array.insert();
      EXPECT_EQ(std::find(block_indices.begin(), block_indices.end(), block_index),
                block_indices.end());
      block_indices.push_back(block_index);
      EXPECT_EQ(block_index, i);
    }

    EXPECT_TRUE(block_array.full());
  }

  void run_remove_and_reuse_test(raft::handle_t const& handle)
  {
    cugraph::dynamic::block_array_t<int32_t> block_array(
      k_elements_per_block, k_num_blocks, handle.get_stream());

    std::vector<size_t> block_indices;
    block_indices.reserve(k_num_blocks);
    for (size_t i = 0; i < k_num_blocks; ++i) {
      block_indices.push_back(block_array.insert());
    }

    block_array.remove(block_indices[1]);
    EXPECT_FALSE(block_array.full());

    auto const reused = block_array.insert();
    EXPECT_EQ(reused, block_indices[1]);
    EXPECT_TRUE(block_array.full());
  }
};

TEST_F(Tests_BlockArray, InsertUntilFull)
{
  raft::handle_t handle;
  run_insert_until_full_test(handle);
}

TEST_F(Tests_BlockArray, RemoveAndReuse)
{
  raft::handle_t handle;
  run_remove_and_reuse_test(handle);
}

TEST_F(Tests_BlockArray, BlockAndElementDataPointers)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<int32_t> block_array(
    Tests_BlockArray::k_elements_per_block, Tests_BlockArray::k_num_blocks, handle.get_stream());

  auto const block_index     = size_t{2};
  auto const in_block_offset = size_t{3};
  auto const element_index = block_index * Tests_BlockArray::k_elements_per_block + in_block_offset;

  EXPECT_EQ(block_array.block_data(block_index),
            block_array.column_data() + block_index * Tests_BlockArray::k_elements_per_block);
  EXPECT_EQ(block_array.element_data(element_index), block_array.column_data() + element_index);
  EXPECT_EQ(block_array.block_data(block_index) + in_block_offset,
            block_array.element_data(element_index));
}

TEST_F(Tests_BlockArray, SingleColumnReadWrite)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<int32_t> block_array(
    Tests_BlockArray::k_elements_per_block, Tests_BlockArray::k_num_blocks, handle.get_stream());

  std::vector<int32_t> expected(Tests_BlockArray::k_num_elements);
  std::iota(expected.begin(), expected.end(), 0);

  raft::update_device(
    block_array.column_data(), expected.data(), expected.size(), handle.get_stream());
  handle.sync_stream();

  auto actual = cugraph::test::to_host(
    handle,
    raft::device_span<int32_t const>(block_array.column_data(), Tests_BlockArray::k_num_elements));

  EXPECT_EQ(expected, actual);
}

TEST_F(Tests_BlockArray, BlockColumnReadWrite)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<int32_t> block_array(
    Tests_BlockArray::k_elements_per_block, Tests_BlockArray::k_num_blocks, handle.get_stream());

  auto const block_index = block_array.insert();

  std::vector<int32_t> expected(Tests_BlockArray::k_elements_per_block);
  std::iota(expected.begin(), expected.end(), static_cast<int32_t>(block_index * 10));

  raft::update_device(
    block_array.block_data(block_index), expected.data(), expected.size(), handle.get_stream());
  handle.sync_stream();

  auto actual = cugraph::test::to_host(
    handle,
    raft::device_span<int32_t const>(block_array.block_data(block_index),
                                     Tests_BlockArray::k_elements_per_block));

  EXPECT_EQ(expected, actual);
}

TEST_F(Tests_BlockArray, MultiColumnBuffer)
{
  raft::handle_t handle;

  constexpr size_t num_items = 4;
  cugraph::dynamic::block_array_t<cuda::std::tuple<int32_t, float>> block_array(
    Tests_BlockArray::k_elements_per_block, size_t{1}, handle.get_stream());

  std::vector<int32_t> int_values(num_items);
  std::iota(int_values.begin(), int_values.end(), 10);

  std::vector<float> float_values(num_items);
  for (size_t i = 0; i < num_items; ++i) {
    float_values[i] = static_cast<float>(i) * 0.5f;
  }

  raft::update_device(
    block_array.column_data<0>(), int_values.data(), int_values.size(), handle.get_stream());
  raft::update_device(
    block_array.column_data<1>(), float_values.data(), float_values.size(), handle.get_stream());
  handle.sync_stream();

  auto actual_int = cugraph::test::to_host(
    handle, raft::device_span<int32_t const>(block_array.column_data<0>(), num_items));
  auto actual_float = cugraph::test::to_host(
    handle, raft::device_span<float const>(block_array.column_data<1>(), num_items));

  EXPECT_EQ(int_values, actual_int);
  EXPECT_EQ(float_values, actual_float);
}

TEST_F(Tests_BlockArray, MultiColumnBlockDataPointers)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<cuda::std::tuple<int32_t, float>> block_array(
    Tests_BlockArray::k_elements_per_block, size_t{1}, handle.get_stream());

  EXPECT_EQ(block_array.block_data<0>(0), block_array.column_data<0>());
  EXPECT_EQ(block_array.block_data<1>(0), block_array.column_data<1>());
  EXPECT_EQ(block_array.element_data<0>(2), block_array.column_data<0>() + 2);
  EXPECT_EQ(block_array.element_data<1>(2), block_array.column_data<1>() + 2);
  EXPECT_NE(block_array.block_array_key(),
            reinterpret_cast<std::byte const*>(block_array.column_data<1>()));
  EXPECT_EQ(block_array.block_array_key(),
            reinterpret_cast<std::byte const*>(block_array.column_data<0>()));
}

CUGRAPH_TEST_PROGRAM_MAIN()
