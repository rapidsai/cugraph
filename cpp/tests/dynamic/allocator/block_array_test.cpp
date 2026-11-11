/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"

#include <cugraph/dynamic/allocator/block_array.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/tuple>

#include <gtest/gtest.h>

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
};

TEST_F(Tests_BlockArray, BufferSize)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<int32_t> block_array(
    Tests_BlockArray::k_elements_per_block, Tests_BlockArray::k_num_blocks, handle.get_stream());

  EXPECT_EQ(block_array.block_capacity(), Tests_BlockArray::k_elements_per_block);
  EXPECT_EQ(block_array.num_blocks(), Tests_BlockArray::k_num_blocks);
  EXPECT_EQ(block_array.num_elements(), Tests_BlockArray::k_num_elements);
}

TEST_F(Tests_BlockArray, IteratorArithmetic)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<int32_t> block_array(
    Tests_BlockArray::k_elements_per_block, Tests_BlockArray::k_num_blocks, handle.get_stream());

  auto const block_index     = size_t{2};
  auto const in_block_offset = size_t{3};
  auto const element_index = block_index * Tests_BlockArray::k_elements_per_block + in_block_offset;

  auto const block_first =
    block_array.begin() + block_index * Tests_BlockArray::k_elements_per_block;
  auto const element = block_array.begin() + element_index;
  EXPECT_EQ(block_first + in_block_offset, element);
}

TEST_F(Tests_BlockArray, SingleColumnReadWrite)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<int32_t> block_array(
    Tests_BlockArray::k_elements_per_block, Tests_BlockArray::k_num_blocks, handle.get_stream());

  std::vector<int32_t> expected(Tests_BlockArray::k_num_elements);
  std::iota(expected.begin(), expected.end(), 0);

  raft::update_device(block_array.begin(), expected.data(), expected.size(), handle.get_stream());
  handle.sync_stream();

  auto actual = cugraph::test::to_host(
    handle, raft::device_span<int32_t const>(block_array.cbegin(), block_array.num_elements()));

  EXPECT_EQ(expected, actual);
}

TEST_F(Tests_BlockArray, BlockColumnReadWrite)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<int32_t> block_array(
    Tests_BlockArray::k_elements_per_block, Tests_BlockArray::k_num_blocks, handle.get_stream());

  auto const block_index = size_t{0};

  std::vector<int32_t> expected(Tests_BlockArray::k_elements_per_block);
  std::iota(expected.begin(), expected.end(), static_cast<int32_t>(block_index * 10));

  auto const block_first =
    block_array.begin() + block_index * Tests_BlockArray::k_elements_per_block;
  raft::update_device(block_first, expected.data(), expected.size(), handle.get_stream());
  handle.sync_stream();

  auto actual = cugraph::test::to_host(
    handle, raft::device_span<int32_t const>(block_first, Tests_BlockArray::k_elements_per_block));

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

  auto const iterators = block_array.begin().get_iterator_tuple();
  raft::update_device(
    cuda::std::get<0>(iterators), int_values.data(), int_values.size(), handle.get_stream());
  raft::update_device(
    cuda::std::get<1>(iterators), float_values.data(), float_values.size(), handle.get_stream());
  handle.sync_stream();

  auto const const_iterators = block_array.cbegin().get_iterator_tuple();
  auto actual_int            = cugraph::test::to_host(
    handle, raft::device_span<int32_t const>(cuda::std::get<0>(const_iterators), num_items));
  auto actual_float = cugraph::test::to_host(
    handle, raft::device_span<float const>(cuda::std::get<1>(const_iterators), num_items));

  EXPECT_EQ(int_values, actual_int);
  EXPECT_EQ(float_values, actual_float);
}

TEST_F(Tests_BlockArray, MultiColumnIteratorArithmetic)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_t<cuda::std::tuple<int32_t, float>> block_array(
    Tests_BlockArray::k_elements_per_block, size_t{1}, handle.get_stream());

  auto const first   = block_array.begin().get_iterator_tuple();
  auto const element = (block_array.begin() + size_t{2}).get_iterator_tuple();
  EXPECT_EQ(cuda::std::get<0>(element), cuda::std::get<0>(first) + size_t{2});
  EXPECT_EQ(cuda::std::get<1>(element), cuda::std::get<1>(first) + size_t{2});
}

CUGRAPH_TEST_PROGRAM_MAIN()
