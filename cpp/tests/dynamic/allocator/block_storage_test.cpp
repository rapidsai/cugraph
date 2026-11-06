/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <typename T>
auto get_slab_iterator_ptr(cugraph::dynamic::block_storage_t<T> const& storage,
                           size_t bin,
                           cuda::stream_ref stream)
{
  auto const slab_iterator_ptrs = storage.slab_iterator_ptrs();
  typename cugraph::dynamic::block_storage_t<T>::iterator const* slab_iterator_ptr{};
  raft::update_host(&slab_iterator_ptr, slab_iterator_ptrs.data() + bin, size_t{1}, stream);
  stream.sync();
  return slab_iterator_ptr;
}

template <typename T>
auto get_slab_iterator(cugraph::dynamic::block_storage_t<T> const& storage,
                       size_t bin,
                       size_t slab_id,
                       cuda::stream_ref stream)
{
  auto const slab_iterator_ptr = get_slab_iterator_ptr(storage, bin, stream);
  typename cugraph::dynamic::block_storage_t<T>::iterator slab_iterator{};
  raft::update_host(&slab_iterator, slab_iterator_ptr + slab_id, size_t{1}, stream);
  stream.sync();
  return slab_iterator;
}

template <typename T>
auto get_block_iterator(cugraph::dynamic::block_storage_t<T> const& storage,
                        size_t bin,
                        size_t slab_id,
                        size_t block_index,
                        cuda::stream_ref stream)
{
  return get_slab_iterator(storage, bin, slab_id, stream) + block_index * (size_t{1} << bin);
}

TEST_F(Tests_BlockStorage, AppendSlabGrowsVector)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());

  EXPECT_EQ(get_slab_iterator_ptr(storage, Tests_BlockStorage::k_bin, handle.get_stream()),
            nullptr);

  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{1}, handle.get_stream());

  EXPECT_NE(get_slab_iterator_ptr(storage, Tests_BlockStorage::k_bin, handle.get_stream()),
            nullptr);
  auto const slab0 =
    get_slab_iterator(storage, Tests_BlockStorage::k_bin, size_t{0}, handle.get_stream());
  auto const slab1 =
    get_slab_iterator(storage, Tests_BlockStorage::k_bin, size_t{1}, handle.get_stream());
  EXPECT_NE(slab0, nullptr);
  EXPECT_NE(slab1, nullptr);
  EXPECT_NE(slab0, slab1);
  EXPECT_EQ(get_slab_iterator_ptr(storage, size_t{0}, handle.get_stream()), nullptr);
}

TEST_F(Tests_BlockStorage, BlockIteratorMatchesBin)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());
  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{0}, handle.get_stream());

  auto const block0 = get_block_iterator(
    storage, Tests_BlockStorage::k_bin, size_t{0}, size_t{0}, handle.get_stream());
  auto const block1 = get_block_iterator(
    storage, Tests_BlockStorage::k_bin, size_t{0}, size_t{1}, handle.get_stream());
  EXPECT_EQ(block1 - block0, size_t{1} << Tests_BlockStorage::k_bin);
}

TEST_F(Tests_BlockStorage, ReadWriteThroughSlabDirectory)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());
  storage.allocate_slab(Tests_BlockStorage::k_bin, size_t{0}, handle.get_stream());

  auto const block_index = size_t{1};
  auto const block_first = get_block_iterator(
    storage, Tests_BlockStorage::k_bin, size_t{0}, block_index, handle.get_stream());
  std::vector<int32_t> expected(size_t{1} << Tests_BlockStorage::k_bin);
  std::iota(expected.begin(), expected.end(), 0);

  raft::update_device(block_first, expected.data(), expected.size(), handle.get_stream());
  handle.sync_stream();

  auto actual =
    cugraph::test::to_host(handle, raft::device_span<int32_t const>(block_first, expected.size()));

  EXPECT_EQ(expected, actual);
}

TEST_F(Tests_BlockStorage, TupleSlabBuffer)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<cuda::std::tuple<int32_t, float>> storage(
    Tests_BlockStorage::k_max_elements_per_slab, handle.get_stream());
  storage.allocate_slab(size_t{1}, size_t{0}, handle.get_stream());

  auto const slab_iterator = get_slab_iterator(storage, size_t{1}, size_t{0}, handle.get_stream());
  EXPECT_NE(cuda::std::get<0>(slab_iterator.get_iterator_tuple()), nullptr);
  EXPECT_NE(cuda::std::get<1>(slab_iterator.get_iterator_tuple()), nullptr);
}

TEST_F(Tests_BlockStorage, ReleaseMiddleSlabKeepsId)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());
  auto const bin = Tests_BlockStorage::k_bin;
  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(bin, size_t{1}, handle.get_stream());
  storage.allocate_slab(bin, size_t{2}, handle.get_stream());
  auto const slab0 = get_slab_iterator(storage, bin, size_t{0}, handle.get_stream());
  auto const slab2 = get_slab_iterator(storage, bin, size_t{2}, handle.get_stream());

  storage.release_slab(bin, size_t{1}, handle.get_stream());

  EXPECT_NE(get_slab_iterator_ptr(storage, bin, handle.get_stream()), nullptr);
  EXPECT_EQ(get_slab_iterator(storage, bin, size_t{0}, handle.get_stream()), slab0);
  EXPECT_EQ(get_slab_iterator(storage, bin, size_t{1}, handle.get_stream()), nullptr);
  EXPECT_EQ(get_slab_iterator(storage, bin, size_t{2}, handle.get_stream()), slab2);

  storage.allocate_slab(bin, size_t{1}, handle.get_stream());
  EXPECT_NE(get_slab_iterator_ptr(storage, bin, handle.get_stream()), nullptr);
  EXPECT_NE(get_slab_iterator(storage, bin, size_t{1}, handle.get_stream()), nullptr);
}

TEST_F(Tests_BlockStorage, ReleaseTailPopsAndCollapsesHoles)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());
  auto const bin = Tests_BlockStorage::k_bin;
  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(bin, size_t{1}, handle.get_stream());
  storage.allocate_slab(bin, size_t{2}, handle.get_stream());

  storage.release_slab(bin, size_t{0}, handle.get_stream());
  storage.release_slab(bin, size_t{1}, handle.get_stream());
  EXPECT_NE(get_slab_iterator_ptr(storage, bin, handle.get_stream()), nullptr);

  storage.release_slab(bin, size_t{2}, handle.get_stream());
  EXPECT_EQ(get_slab_iterator_ptr(storage, bin, handle.get_stream()), nullptr);
}

TEST_F(Tests_BlockStorage, ReleaseOnlyTailPopsThatSlab)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());
  auto const bin = Tests_BlockStorage::k_bin;
  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  storage.allocate_slab(bin, size_t{1}, handle.get_stream());
  auto const slab0 = get_slab_iterator(storage, bin, size_t{0}, handle.get_stream());

  storage.release_slab(bin, size_t{1}, handle.get_stream());

  EXPECT_NE(get_slab_iterator_ptr(storage, bin, handle.get_stream()), nullptr);
  EXPECT_EQ(get_slab_iterator(storage, bin, size_t{0}, handle.get_stream()), slab0);
}

TEST_F(Tests_BlockStorage, SlabDirectoryFollowsTableGrowth)
{
  raft::handle_t handle;
  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());
  auto const bin             = Tests_BlockStorage::k_bin;
  constexpr size_t num_slabs = size_t{32};
  for (size_t slab_id = 0; slab_id < num_slabs; ++slab_id) {
    storage.allocate_slab(bin, slab_id, handle.get_stream());
  }

  EXPECT_NE(get_slab_iterator_ptr(storage, bin, handle.get_stream()), nullptr);
  for (size_t slab_id = 0; slab_id < num_slabs; ++slab_id) {
    EXPECT_NE(get_slab_iterator(storage, bin, slab_id, handle.get_stream()), nullptr);
  }
}

TEST_F(Tests_BlockStorage, RejectsInvalidArguments)
{
  raft::handle_t handle;
  EXPECT_THROW(cugraph::dynamic::block_storage_t<int32_t>(size_t{3}, handle.get_stream()),
               cugraph::logic_error);

  cugraph::dynamic::block_storage_t<int32_t> storage(Tests_BlockStorage::k_max_elements_per_slab,
                                                     handle.get_stream());
  auto const bin = Tests_BlockStorage::k_bin;

  EXPECT_THROW(storage.allocate_slab(size_t{4}, size_t{0}, handle.get_stream()),
               cugraph::logic_error);
  EXPECT_THROW(storage.allocate_slab(bin, size_t{1}, handle.get_stream()), cugraph::logic_error);

  storage.allocate_slab(bin, size_t{0}, handle.get_stream());
  EXPECT_THROW(storage.allocate_slab(bin, size_t{0}, handle.get_stream()), cugraph::logic_error);

  storage.release_slab(bin, size_t{0}, handle.get_stream());
  EXPECT_THROW(storage.release_slab(bin, size_t{0}, handle.get_stream()), cugraph::logic_error);
}

CUGRAPH_TEST_PROGRAM_MAIN()
