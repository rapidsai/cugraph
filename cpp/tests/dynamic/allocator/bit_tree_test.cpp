/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"

#include <cugraph/dynamic/allocator/bit_tree.hpp>
#include <cugraph/utilities/error.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

class Tests_BitTree : public ::testing::Test {
 public:
  Tests_BitTree() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  void run_allocate_until_full_test()
  {
    cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{4});

    EXPECT_EQ(tree.log_block_capacity(), size_t{2});
    EXPECT_EQ(tree.num_occupied_blocks(), size_t{0});
    EXPECT_FALSE(tree.full());

    std::vector<size_t> slots;
    slots.reserve(4);
    for (size_t i = 0; i < size_t{4}; ++i) {
      auto const slot = tree.allocate();
      EXPECT_EQ(std::find(slots.begin(), slots.end(), slot), slots.end());
      slots.push_back(slot);
      EXPECT_EQ(tree.num_occupied_blocks(), i + 1);
    }

    EXPECT_TRUE(tree.full());
    EXPECT_EQ(tree.num_occupied_blocks(), size_t{4});
  }

  void run_release_and_reuse_test()
  {
    cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{4});

    std::vector<size_t> slots;
    slots.reserve(4);
    for (size_t i = 0; i < size_t{4}; ++i) {
      slots.push_back(tree.allocate());
    }

    tree.release(slots[1]);
    EXPECT_FALSE(tree.full());
    EXPECT_EQ(tree.num_occupied_blocks(), size_t{3});

    auto const reused = tree.allocate();
    EXPECT_EQ(reused, slots[1]);
    EXPECT_TRUE(tree.full());
  }
};

TEST_F(Tests_BitTree, AllocateUntilFull) { run_allocate_until_full_test(); }

TEST_F(Tests_BitTree, ReleaseAndReuse) { run_release_and_reuse_test(); }

TEST_F(Tests_BitTree, BulkAllocateFailsWhenFewerBlocksAreFree)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{4});

  EXPECT_THROW(tree.allocate(size_t{10}), cugraph::logic_error);
  EXPECT_EQ(tree.num_occupied_blocks(), size_t{0});

  auto const blocks = tree.allocate(tree.num_available_blocks());
  EXPECT_EQ(blocks.size(), size_t{4});
  EXPECT_TRUE(tree.full());
  EXPECT_THROW(tree.allocate(size_t{1}), cugraph::logic_error);
  EXPECT_EQ(tree.num_occupied_blocks(), size_t{4});
}

TEST_F(Tests_BitTree, BulkAllocateTakesEveryFreeBitInALeafWord)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{32});

  auto const first = tree.allocate(size_t{14});
  EXPECT_EQ(first.size(), size_t{14});

  EXPECT_THROW(tree.allocate(size_t{30}), cugraph::logic_error);
  EXPECT_EQ(tree.num_occupied_blocks(), size_t{14});

  auto const rest = tree.allocate(tree.num_available_blocks());
  EXPECT_EQ(rest.size(), size_t{18});
  EXPECT_EQ(tree.num_occupied_blocks(), size_t{32});
  EXPECT_TRUE(tree.full());
  EXPECT_EQ(rest.front(), size_t{14});
  EXPECT_EQ(rest.back(), size_t{31});
}

TEST_F(Tests_BitTree, BulkAllocateSpansLeafWords)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{64});

  auto const blocks = tree.allocate(size_t{40});

  EXPECT_EQ(blocks.size(), size_t{40});
  EXPECT_EQ(blocks.front(), size_t{0});
  EXPECT_EQ(blocks[31], size_t{31});
  EXPECT_EQ(blocks[32], size_t{32});
  EXPECT_EQ(blocks.back(), size_t{39});
  EXPECT_EQ(tree.num_occupied_blocks(), size_t{40});

  EXPECT_THROW(tree.allocate(size_t{100}), cugraph::logic_error);
  EXPECT_EQ(tree.num_occupied_blocks(), size_t{40});

  auto const rest = tree.allocate(tree.num_available_blocks());
  EXPECT_EQ(rest.size(), size_t{24});
  EXPECT_EQ(rest.front(), size_t{40});
  EXPECT_EQ(rest.back(), size_t{63});
  EXPECT_TRUE(tree.full());
}

TEST_F(Tests_BitTree, BulkAllocateSkipsFilledLeafWord)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{128});

  auto const filled = tree.allocate(size_t{64});
  tree.release(filled[10]);

  auto const blocks = tree.allocate(size_t{30});
  EXPECT_EQ(blocks.size(), size_t{30});
  EXPECT_EQ(blocks.front(), size_t{10});
  EXPECT_EQ(blocks[1], size_t{64});
  EXPECT_EQ(blocks.back(), size_t{92});
  EXPECT_EQ(tree.allocate(), size_t{93});
}

TEST_F(Tests_BitTree, BulkAllocateClearsSummaryPrefix)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{2048});

  auto const first = tree.allocate(size_t{1024});
  EXPECT_EQ(first.front(), size_t{0});
  EXPECT_EQ(first.back(), size_t{1023});

  auto const second = tree.allocate(size_t{1024});
  EXPECT_EQ(second.front(), size_t{1024});
  EXPECT_EQ(second.back(), size_t{2047});
  EXPECT_TRUE(tree.full());

  tree.release(size_t{10});
  tree.release(size_t{1500});
  EXPECT_EQ(tree.allocate(), size_t{10});
  EXPECT_EQ(tree.allocate(), size_t{1500});
  EXPECT_TRUE(tree.full());
}

TEST_F(Tests_BitTree, BulkAllocateAllRemainingFreeBlocks)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{64});

  EXPECT_EQ(tree.allocate(tree.num_blocks()).size(), size_t{64});
  tree.release(size_t{5});
  tree.release(size_t{40});

  auto const blocks = tree.allocate(tree.num_available_blocks());
  EXPECT_EQ(blocks.size(), size_t{2});
  EXPECT_EQ(blocks[0], size_t{5});
  EXPECT_EQ(blocks[1], size_t{40});
  EXPECT_TRUE(tree.full());

  tree.release(size_t{5});
  EXPECT_EQ(tree.allocate(), size_t{5});
  EXPECT_TRUE(tree.full());
}

TEST_F(Tests_BitTree, BulkAllocateAllRemainingSkipsFullLeadingLeafWords)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{128});

  EXPECT_EQ(tree.allocate(tree.num_blocks()).size(), size_t{128});
  tree.release(size_t{40});
  tree.release(size_t{100});

  auto const blocks = tree.allocate(tree.num_available_blocks());
  EXPECT_EQ(blocks.size(), size_t{2});
  EXPECT_EQ(blocks[0], size_t{40});
  EXPECT_EQ(blocks[1], size_t{100});
  EXPECT_TRUE(tree.full());

  tree.release(size_t{40});
  EXPECT_EQ(tree.allocate(), size_t{40});
  EXPECT_TRUE(tree.full());
}

TEST_F(Tests_BitTree, BulkAllocateFullEmptySlab)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{64});

  auto const blocks = tree.allocate(tree.num_blocks());
  EXPECT_EQ(blocks.size(), size_t{64});
  EXPECT_EQ(blocks.front(), size_t{0});
  EXPECT_EQ(blocks.back(), size_t{63});
  EXPECT_TRUE(tree.full());

  tree.release(size_t{40});
  EXPECT_EQ(tree.allocate(), size_t{40});
  EXPECT_TRUE(tree.full());
}

TEST_F(Tests_BitTree, BulkAllocateReleaseRestoresLeafWord)
{
  cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{64});

  auto const blocks = tree.allocate(size_t{32});
  EXPECT_EQ(blocks.size(), size_t{32});

  tree.release(blocks[0]);
  EXPECT_EQ(tree.allocate(), blocks[0]);
  EXPECT_EQ(tree.num_occupied_blocks(), size_t{32});
}

CUGRAPH_TEST_PROGRAM_MAIN()
