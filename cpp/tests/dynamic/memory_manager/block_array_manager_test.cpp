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

TEST_F(Tests_BlockArrayManager, AllocateDegreeZero)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);
  auto const access = manager.allocate(size_t{0}, handle.get_stream());

  EXPECT_EQ(access.slab_id, cugraph::dynamic::block_access_data_t::invalid_slab_id);
  EXPECT_EQ(access.block_index, size_t{0});
  EXPECT_EQ(access.log_num_elements_per_block, size_t{0});
}

TEST_F(Tests_BlockArrayManager, AllocateReusesBlockArray)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const first  = manager.allocate(size_t{4}, handle.get_stream());
  auto const second = manager.allocate(size_t{4}, handle.get_stream());

  EXPECT_NE(first.slab_id, cugraph::dynamic::block_access_data_t::invalid_slab_id);
  EXPECT_EQ(first.slab_id, second.slab_id);
  EXPECT_NE(first.block_index, second.block_index);
  EXPECT_EQ(first.log_num_elements_per_block, size_t{2});
  EXPECT_EQ(second.log_num_elements_per_block, size_t{2});
}

TEST_F(Tests_BlockArrayManager, AllocateRejectsNonPowerOfTwo)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  EXPECT_THROW(manager.allocate(size_t{5}, handle.get_stream()), cugraph::logic_error);
}

TEST_F(Tests_BlockArrayManager, AllocateDegreeOneUsesBinZero)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const access = manager.allocate(size_t{1}, handle.get_stream());

  EXPECT_NE(access.slab_id, cugraph::dynamic::block_access_data_t::invalid_slab_id);
  EXPECT_EQ(access.block_index, size_t{0});
  EXPECT_EQ(access.log_num_elements_per_block, size_t{0});
}

TEST_F(Tests_BlockArrayManager, AllocateExceedsMaxElementsPerBlockArray)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  EXPECT_THROW(manager.allocate(size_t{32}, handle.get_stream()), cugraph::logic_error);
}

TEST_F(Tests_BlockArrayManager, ReleaseAndAllocateOnSameBlockArray)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const first  = manager.allocate(size_t{4}, handle.get_stream());
  auto const second = manager.allocate(size_t{4}, handle.get_stream());
  EXPECT_NE(first.block_index, second.block_index);

  manager.release(first);

  auto const third = manager.allocate(size_t{4}, handle.get_stream());
  EXPECT_EQ(third.slab_id, first.slab_id);
  EXPECT_EQ(third.log_num_elements_per_block, first.log_num_elements_per_block);
  EXPECT_NE(third.block_index, second.block_index);
}

TEST_F(Tests_BlockArrayManager, DifferentBinsUseDifferentBlockArrays)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const small = manager.allocate(size_t{4}, handle.get_stream());
  auto const large = manager.allocate(size_t{8}, handle.get_stream());

  EXPECT_EQ(small.log_num_elements_per_block, size_t{2});
  EXPECT_EQ(large.log_num_elements_per_block, size_t{3});
  EXPECT_EQ(small.slab_id, size_t{0});
  EXPECT_EQ(large.slab_id, size_t{0});
}

TEST_F(Tests_BlockArrayManager, ReleaseAllClearsAllocations)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<int32_t> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);
  manager.allocate(size_t{4}, handle.get_stream());
  manager.release_all();

  auto const access = manager.allocate(size_t{4}, handle.get_stream());
  EXPECT_EQ(access.slab_id, size_t{0});
  EXPECT_EQ(access.log_num_elements_per_block, size_t{2});
}

TEST_F(Tests_BlockArrayManager, MultiColumnAllocateAndRelease)
{
  raft::handle_t handle;

  cugraph::dynamic::block_array_manager_t<cuda::std::tuple<int32_t, float>> manager(
    Tests_BlockArrayManager::k_max_elements_per_block_array);

  auto const first  = manager.allocate(size_t{4}, handle.get_stream());
  auto const second = manager.allocate(size_t{4}, handle.get_stream());

  EXPECT_NE(first.slab_id, cugraph::dynamic::block_access_data_t::invalid_slab_id);
  EXPECT_EQ(first.slab_id, second.slab_id);
  EXPECT_NE(first.block_index, second.block_index);

  manager.release(first);

  auto const third = manager.allocate(size_t{4}, handle.get_stream());
  EXPECT_EQ(third.slab_id, first.slab_id);
  EXPECT_NE(third.block_index, second.block_index);
}

CUGRAPH_TEST_PROGRAM_MAIN()
