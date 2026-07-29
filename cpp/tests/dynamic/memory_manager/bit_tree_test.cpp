/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"

#include <cugraph/dynamic/memory_manager/bit_tree.hpp>

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

  void run_insert_until_full_test()
  {
    cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{4});

    EXPECT_EQ(tree.log_elements_per_block(), size_t{2});
    EXPECT_EQ(tree.num_occupied_blocks(), size_t{0});
    EXPECT_FALSE(tree.full());

    std::vector<size_t> slots;
    slots.reserve(4);
    for (size_t i = 0; i < size_t{4}; ++i) {
      auto const slot = tree.insert();
      EXPECT_EQ(std::find(slots.begin(), slots.end(), slot), slots.end());
      slots.push_back(slot);
      EXPECT_EQ(tree.num_occupied_blocks(), i + 1);
    }

    EXPECT_TRUE(tree.full());
    EXPECT_EQ(tree.num_occupied_blocks(), size_t{4});
  }

  void run_remove_and_reuse_test()
  {
    cugraph::dynamic::bit_tree_t tree(size_t{4}, size_t{4});

    std::vector<size_t> slots;
    slots.reserve(4);
    for (size_t i = 0; i < size_t{4}; ++i) {
      slots.push_back(tree.insert());
    }

    tree.remove(slots[1]);
    EXPECT_FALSE(tree.full());
    EXPECT_EQ(tree.num_occupied_blocks(), size_t{3});

    auto const reused = tree.insert();
    EXPECT_EQ(reused, slots[1]);
    EXPECT_TRUE(tree.full());
  }
};

TEST_F(Tests_BitTree, InsertUntilFull) { run_insert_until_full_test(); }

TEST_F(Tests_BitTree, RemoveAndReuse) { run_remove_and_reuse_test(); }

CUGRAPH_TEST_PROGRAM_MAIN()
