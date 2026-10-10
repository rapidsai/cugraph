/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"

#include <cugraph/dynamic/allocator/block_storage.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/tuple>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

class Tests_BlockStorage : public ::testing::Test {
 public:
  static constexpr size_t k_max_elements_per_slab = size_t{8};
  static constexpr size_t k_bin                   = size_t{2};

  Tests_BlockStorage() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(Tests_BlockStorage, AppendSlabGrowsVector)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab);

  EXPECT_EQ(storage.num_slabs(Tests_BlockStorage::k_bin), size_t{0});
  EXPECT_FALSE(storage.contains(Tests_BlockStorage::k_bin, size_t{0}));

  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{1}, handle.get_stream());

  EXPECT_EQ(storage.num_slabs(Tests_BlockStorage::k_bin), size_t{2});
  EXPECT_TRUE(storage.contains(Tests_BlockStorage::k_bin, size_t{0}));
  EXPECT_TRUE(storage.contains(Tests_BlockStorage::k_bin, size_t{1}));
  EXPECT_EQ(storage.num_slabs(size_t{0}), size_t{0});
}

TEST_F(Tests_BlockStorage, BlockArrayMatchesBin)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab);
  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{0}, handle.get_stream());

  auto const& block_array = storage.block_array(Tests_BlockStorage::k_bin, size_t{0});
  EXPECT_EQ(block_array.block_capacity(), size_t{1} << Tests_BlockStorage::k_bin);
  EXPECT_EQ(block_array.num_blocks(),
            Tests_BlockStorage::k_max_elements_per_slab / block_array.block_capacity());
  EXPECT_EQ(block_array.num_elements(), Tests_BlockStorage::k_max_elements_per_slab);

  cugraph::dynamic::block_storage_t<int32_t> const& const_storage = storage;
  EXPECT_EQ(const_storage.block_array(Tests_BlockStorage::k_bin, size_t{0}).num_elements(),
            Tests_BlockStorage::k_max_elements_per_slab);
}

TEST_F(Tests_BlockStorage, ReadWriteThroughBlockArray)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab);
  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{0}, handle.get_stream());

  auto& block_array = storage.block_array(Tests_BlockStorage::k_bin, size_t{0});
  std::vector<int32_t> expected(block_array.num_elements());
  std::iota(expected.begin(), expected.end(), 0);

  raft::update_device(
    block_array.column_data(), expected.data(), expected.size(), handle.get_stream());
  handle.sync_stream();

  auto actual = cugraph::test::to_host(
    handle, raft::device_span<int32_t const>(block_array.column_data(), expected.size()));

  EXPECT_EQ(expected, actual);
}

TEST_F(Tests_BlockStorage, TupleSlabBuffer)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<cuda::std::tuple<int32_t, float>> storage(
    Tests_BlockStorage::k_max_elements_per_slab);
  storage.allocate_slab(size_t{1}, size_t{0}, handle.get_stream());

  auto const& block_array = storage.block_array(size_t{1}, size_t{0});
  EXPECT_EQ(block_array.block_capacity(), size_t{2});
  EXPECT_EQ(block_array.num_blocks(), Tests_BlockStorage::k_max_elements_per_slab / size_t{2});
  EXPECT_NE(block_array.column_data<0>(), nullptr);
  EXPECT_NE(block_array.column_data<1>(), nullptr);
}

TEST_F(Tests_BlockStorage, ReleaseMiddleSlabKeepsId)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab);
  auto const bin = Tests_BlockStorage::k_bin;
  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(bin, size_t{1}, handle.get_stream());
  storage.allocate_slab(bin, size_t{2}, handle.get_stream());

  storage.release_slab(bin, size_t{1});

  EXPECT_EQ(storage.num_slabs(bin), size_t{3});
  EXPECT_TRUE(storage.contains(bin, size_t{0}));
  EXPECT_FALSE(storage.contains(bin, size_t{1}));
  EXPECT_TRUE(storage.contains(bin, size_t{2}));

  storage.allocate_slab(bin, size_t{1}, handle.get_stream());
  EXPECT_TRUE(storage.contains(bin, size_t{1}));
  EXPECT_EQ(storage.num_slabs(bin), size_t{3});
  EXPECT_EQ(storage.block_array(bin, size_t{1}).block_capacity(), size_t{1} << bin);
}

TEST_F(Tests_BlockStorage, ReleaseTailPopsAndCollapsesHoles)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab);
  auto const bin = Tests_BlockStorage::k_bin;
  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(bin, size_t{1}, handle.get_stream());
  storage.allocate_slab(bin, size_t{2}, handle.get_stream());

  storage.release_slab(bin, size_t{0});
  storage.release_slab(bin, size_t{1});
  EXPECT_EQ(storage.num_slabs(bin), size_t{3});

  storage.release_slab(bin, size_t{2});
  EXPECT_EQ(storage.num_slabs(bin), size_t{0});
  EXPECT_FALSE(storage.contains(bin, size_t{0}));
  EXPECT_FALSE(storage.contains(bin, size_t{2}));
}

TEST_F(Tests_BlockStorage, ReleaseOnlyTailPopsThatSlab)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab);
  auto const bin = Tests_BlockStorage::k_bin;
  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(bin, size_t{1}, handle.get_stream());

  storage.release_slab(bin, size_t{1});

  EXPECT_EQ(storage.num_slabs(bin), size_t{1});
  EXPECT_TRUE(storage.contains(bin, size_t{0}));
  EXPECT_FALSE(storage.contains(bin, size_t{1}));
}

TEST_F(Tests_BlockStorage, RejectsInvalidArguments)
{
  raft::handle_t handle;
  EXPECT_THROW(cugraph::dynamic::block_storage_t<int32_t>(size_t{3}), cugraph::logic_error);

  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab);
  auto const bin = Tests_BlockStorage::k_bin;

  EXPECT_THROW(storage.allocate_slab(size_t{4}, size_t{0}, handle.get_stream()),
               cugraph::logic_error);
  EXPECT_THROW(storage.allocate_slab(bin, size_t{1}, handle.get_stream()), cugraph::logic_error);
  EXPECT_THROW(storage.num_slabs(cugraph::dynamic::block_storage_t<int32_t>::num_bins_v),
               cugraph::logic_error);

  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  EXPECT_THROW(storage.allocate_slab(bin, size_t{0}, handle.get_stream()), cugraph::logic_error);
  EXPECT_THROW(storage.block_array(bin, size_t{1}), cugraph::logic_error);

  storage.release_slab(bin, size_t{0});
  EXPECT_THROW(storage.release_slab(bin, size_t{0}), cugraph::logic_error);
  EXPECT_THROW(storage.block_array(bin, size_t{0}), cugraph::logic_error);
}

CUGRAPH_TEST_PROGRAM_MAIN()
