/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"

#include <cugraph/dynamic/allocator/block_allocator.hpp>
#include <cugraph/dynamic/allocator/block_storage.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <cuda/stream>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>

class Tests_BlockAllocator : public ::testing::Test {
 public:
  static constexpr size_t k_max_elements_per_slab = size_t{8};

  static void apply(cugraph::dynamic::block_storage_t<int32_t>& storage,
                    cugraph::dynamic::block_location_t const& block,
                    std::optional<size_t> const& slab_id,
                    cuda::stream_ref stream)
  {
    if (slab_id.has_value()) { storage.allocate_slab(block.bin, *slab_id, stream); }
  }
};

TEST_F(Tests_BlockAllocator, AllocateReusesBlockInLiveSlab)
{
  raft::handle_t handle;
  cugraph::dynamic::block_allocator_t allocator(Tests_BlockAllocator::k_max_elements_per_slab);
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockAllocator::k_max_elements_per_slab,
                                                     handle.get_stream());

  auto const [first, first_slab]   = allocator.allocate(size_t{4});
  auto const [second, second_slab] = allocator.allocate(size_t{4});
  Tests_BlockAllocator::apply(storage, first, first_slab, handle.get_stream());
  Tests_BlockAllocator::apply(storage, second, second_slab, handle.get_stream());

  EXPECT_EQ(first.bin, size_t{2});
  EXPECT_EQ(first.slab_id, size_t{0});
  EXPECT_EQ(second.slab_id, first.slab_id);
  EXPECT_NE(first.block_index, second.block_index);
  EXPECT_EQ(first_slab, std::optional<size_t>{size_t{0}});
  EXPECT_FALSE(second_slab.has_value());
  EXPECT_EQ(allocator.num_slabs(first.bin), size_t{1});
}

TEST_F(Tests_BlockAllocator, BulkAllocateReportsEachNewSlab)
{
  raft::handle_t handle;
  cugraph::dynamic::block_allocator_t allocator(Tests_BlockAllocator::k_max_elements_per_slab);
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockAllocator::k_max_elements_per_slab,
                                                     handle.get_stream());

  auto const [blocks, slab_ids] = allocator.allocate(size_t{4}, size_t{3});
  for (auto const slab_id : slab_ids) {
    storage.allocate_slab(blocks.front().bin, slab_id, handle.get_stream());
  }

  EXPECT_EQ(blocks.size(), size_t{3});
  EXPECT_EQ(slab_ids.size(), size_t{2});
  EXPECT_EQ(slab_ids[0], size_t{0});
  EXPECT_EQ(slab_ids[1], size_t{1});
  EXPECT_EQ(allocator.num_slabs(blocks.front().bin), size_t{2});
}

TEST_F(Tests_BlockAllocator, ReleaseEmptyMiddleSlabKeepsId)
{
  raft::handle_t handle;
  cugraph::dynamic::block_allocator_t allocator(Tests_BlockAllocator::k_max_elements_per_slab);
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockAllocator::k_max_elements_per_slab,
                                                     handle.get_stream());

  auto const [blocks, slab_ids] = allocator.allocate(size_t{4}, size_t{4});
  for (auto const slab_id : slab_ids) {
    storage.allocate_slab(blocks.front().bin, slab_id, handle.get_stream());
  }
  auto const bin = blocks.front().bin;

  EXPECT_FALSE(allocator.release(blocks[0]).has_value());
  auto const freed = allocator.release(blocks[1]);
  ASSERT_TRUE(freed.has_value());
  EXPECT_EQ(*freed, size_t{0});
  storage.release_slab(bin, *freed, handle.get_stream());

  EXPECT_EQ(allocator.num_slabs(bin), size_t{2});
}

TEST_F(Tests_BlockAllocator, ReusesFreedMiddleSlabBeforeGrowing)
{
  raft::handle_t handle;
  cugraph::dynamic::block_allocator_t allocator(Tests_BlockAllocator::k_max_elements_per_slab);
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockAllocator::k_max_elements_per_slab,
                                                     handle.get_stream());

  auto const [blocks, slab_ids] = allocator.allocate(size_t{4}, size_t{4});
  auto const bin                = blocks.front().bin;
  for (auto const slab_id : slab_ids) {
    storage.allocate_slab(bin, slab_id, handle.get_stream());
  }
  auto const freed_first = allocator.release(blocks[0]);
  auto const freed_slab  = allocator.release(blocks[1]);
  EXPECT_FALSE(freed_first.has_value());
  ASSERT_TRUE(freed_slab.has_value());
  storage.release_slab(bin, *freed_slab, handle.get_stream());

  auto const [block, slab_id] = allocator.allocate(size_t{4});
  ASSERT_TRUE(slab_id.has_value());
  EXPECT_EQ(*slab_id, size_t{0});
  EXPECT_EQ(block.slab_id, size_t{0});
  storage.allocate_slab(bin, *slab_id, handle.get_stream());

  EXPECT_EQ(allocator.num_slabs(bin), size_t{2});
}

TEST_F(Tests_BlockAllocator, ReleaseEmptyTailPopsAndCollapsesHoles)
{
  raft::handle_t handle;
  cugraph::dynamic::block_allocator_t allocator(Tests_BlockAllocator::k_max_elements_per_slab);
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockAllocator::k_max_elements_per_slab,
                                                     handle.get_stream());

  auto const [blocks, slab_ids] = allocator.allocate(size_t{4}, size_t{4});
  auto const bin                = blocks.front().bin;
  for (auto const slab_id : slab_ids) {
    storage.allocate_slab(bin, slab_id, handle.get_stream());
  }

  EXPECT_FALSE(allocator.release(blocks[1]).has_value());
  auto const freed_middle = allocator.release(blocks[0]);
  ASSERT_TRUE(freed_middle.has_value());
  storage.release_slab(bin, *freed_middle, handle.get_stream());

  EXPECT_FALSE(allocator.release(blocks[2]).has_value());
  auto const freed_tail = allocator.release(blocks[3]);
  ASSERT_TRUE(freed_tail.has_value());
  EXPECT_EQ(*freed_tail, size_t{1});
  storage.release_slab(bin, *freed_tail, handle.get_stream());

  EXPECT_EQ(allocator.num_slabs(bin), size_t{0});
}

TEST_F(Tests_BlockAllocator, BulkAllocateDrainsLiveSlabBeforeOpeningAnother)
{
  raft::handle_t handle;
  constexpr size_t max_elements = size_t{32};
  cugraph::dynamic::block_allocator_t allocator(max_elements);
  cugraph::dynamic::block_storage_t<int32_t> storage(max_elements, handle.get_stream());

  auto const [first, first_slab] = allocator.allocate(size_t{4});
  Tests_BlockAllocator::apply(storage, first, first_slab, handle.get_stream());

  auto const [blocks, slab_ids] = allocator.allocate(size_t{4}, size_t{10});
  for (auto const slab_id : slab_ids) {
    storage.allocate_slab(first.bin, slab_id, handle.get_stream());
  }

  EXPECT_EQ(blocks.size(), size_t{10});
  EXPECT_EQ(slab_ids.size(), size_t{1});
  size_t from_first_slab = 0;
  for (auto const& block : blocks) {
    EXPECT_EQ(block.bin, first.bin);
    if (block.slab_id == first.slab_id) { ++from_first_slab; }
  }
  EXPECT_EQ(from_first_slab, size_t{7});
  EXPECT_EQ(allocator.num_slabs(first.bin), size_t{2});
}

TEST_F(Tests_BlockAllocator, AllocateRejectsNonPowerOfTwo)
{
  cugraph::dynamic::block_allocator_t allocator(Tests_BlockAllocator::k_max_elements_per_slab);

  EXPECT_THROW(allocator.allocate(size_t{5}), cugraph::logic_error);
  EXPECT_THROW(allocator.allocate(size_t{16}), cugraph::logic_error);
}

CUGRAPH_TEST_PROGRAM_MAIN()
