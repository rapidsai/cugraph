/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"

#include <cugraph/dynamic/memory_manager/block_array_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/tuple>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

class Tests_BlockArrayManager : public ::testing::Test {
 public:
  static constexpr size_t k_max_elements_per_block_array = size_t{16};

  Tests_BlockArrayManager() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(Tests_BlockArrayManager, InsertDegreeZero)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);
  auto const access = manager.insert(size_t{0}, handle.get_stream());

  EXPECT_EQ(access.block_array_key, nullptr);
  EXPECT_EQ(access.block_index, size_t{0});
  EXPECT_EQ(access.num_elements_per_block, size_t{0});
}

TEST_F(Tests_BlockArrayManager, InsertReusesBlockArray)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const first  = manager.insert(size_t{4}, handle.get_stream());
  auto const second = manager.insert(size_t{4}, handle.get_stream());

  EXPECT_NE(first.block_array_key, nullptr);
  EXPECT_EQ(first.block_array_key, second.block_array_key);
  EXPECT_NE(first.block_index, second.block_index);
  EXPECT_EQ(first.num_elements_per_block, size_t{4});
  EXPECT_EQ(second.num_elements_per_block, size_t{4});
}

TEST_F(Tests_BlockArrayManager, InsertRoundsUpElementsPerBlock)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const access = manager.insert(size_t{5}, handle.get_stream());

  EXPECT_NE(access.block_array_key, nullptr);
  EXPECT_EQ(access.block_index, size_t{0});
  EXPECT_EQ(access.num_elements_per_block, size_t{8});
}

TEST_F(Tests_BlockArrayManager, InsertDegreeOneUsesBinZero)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const access = manager.insert(size_t{1}, handle.get_stream());

  EXPECT_NE(access.block_array_key, nullptr);
  EXPECT_EQ(access.block_index, size_t{0});
  EXPECT_EQ(access.num_elements_per_block, size_t{1});
}

TEST_F(Tests_BlockArrayManager, InsertExceedsMaxElementsPerBlockArray)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  EXPECT_THROW(manager.insert(size_t{32}, handle.get_stream()), cugraph::logic_error);
}

TEST_F(Tests_BlockArrayManager, RemoveAndInsertOnSameBlockArray)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const first  = manager.insert(size_t{4}, handle.get_stream());
  auto const second = manager.insert(size_t{4}, handle.get_stream());
  EXPECT_NE(first.block_index, second.block_index);

  manager.remove(first);

  auto const third = manager.insert(size_t{4}, handle.get_stream());
  EXPECT_EQ(third.block_array_key, first.block_array_key);
  EXPECT_EQ(third.num_elements_per_block, first.num_elements_per_block);
  EXPECT_NE(third.block_index, second.block_index);
}

TEST_F(Tests_BlockArrayManager, DifferentBinsUseDifferentBlockArrays)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const small = manager.insert(size_t{4}, handle.get_stream());
  auto const large = manager.insert(size_t{8}, handle.get_stream());

  EXPECT_NE(small.block_array_key, large.block_array_key);
  EXPECT_EQ(small.num_elements_per_block, size_t{4});
  EXPECT_EQ(large.num_elements_per_block, size_t{8});
}

TEST_F(Tests_BlockArrayManager, RemoveAllClearsAllocations)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);
  manager.insert(size_t{4}, handle.get_stream());
  manager.remove_all();

  auto const access = manager.insert(size_t{4}, handle.get_stream());
  EXPECT_NE(access.block_array_key, nullptr);
  EXPECT_EQ(access.num_elements_per_block, size_t{4});
}

TEST_F(Tests_BlockArrayManager, MultiColumnInsertAndRemove)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<cuda::std::tuple<int32_t, float>> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const first  = manager.insert(size_t{4}, handle.get_stream());
  auto const second = manager.insert(size_t{4}, handle.get_stream());

  EXPECT_NE(first.block_array_key, nullptr);
  EXPECT_EQ(first.block_array_key, second.block_array_key);
  EXPECT_NE(first.block_index, second.block_index);

  manager.remove(first);

  auto const third = manager.insert(size_t{4}, handle.get_stream());
  EXPECT_EQ(third.block_array_key, first.block_array_key);
  EXPECT_NE(third.block_index, second.block_index);
}

CUGRAPH_TEST_PROGRAM_MAIN()
