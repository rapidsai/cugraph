/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <utilities/error.hpp>
#include <utilities/host_scalar_comm.cuh>
#include <utilities/thrust_tuple_utils.cuh>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cinttypes>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr move_and_invalidate_if_block_size = 128;

// FIXME: better move to another file for reusability
inline size_t round_up(size_t number_to_round, size_t modulus)
{
  return ((number_to_round + (modulus - 1)) / modulus) * modulus;
}

template <size_t num_buckets, typename RowIterator, typename vertex_t, typename SplitOp>
__global__ void move_and_invalidate_if(RowIterator row_first,
                                       RowIterator row_last,
                                       vertex_t** bucket_ptrs,
                                       size_t* bucket_sizes_ptr,
                                       size_t this_bucket_idx,
                                       size_t invalid_bucket_idx,
                                       vertex_t invalid_vertex,
                                       SplitOp split_op)
{
  static_assert(
    std::is_same<typename std::iterator_traits<RowIterator>::value_type, vertex_t>::value);
  auto const tid    = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx        = tid;
  size_t block_idx  = blockIdx.x;
  auto num_elements = thrust::distance(row_first, row_last);
  // FIXME: it might be more performant to process more than one element per thread
  auto num_blocks = (num_elements + blockDim.x - 1) / blockDim.x;

  using BlockScan = cub::BlockScan<size_t, move_and_invalidate_if_block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ size_t bucket_block_start_offsets[num_buckets];

  size_t bucket_block_local_offsets[num_buckets];
  size_t bucket_block_aggregate_sizes[num_buckets];

  while (block_idx < num_blocks) {
    for (size_t i = 0; i < num_buckets; ++i) { bucket_block_local_offsets[i] = 0; }

    size_t selected_bucket_idx{invalid_bucket_idx};
    vertex_t key{invalid_vertex};

    if (idx < num_elements) {
      key                 = *(row_first + idx);
      selected_bucket_idx = split_op(key);
      if (selected_bucket_idx != this_bucket_idx) {
        *(row_first + idx) = invalid_vertex;
        if (selected_bucket_idx != invalid_bucket_idx) {
          bucket_block_local_offsets[selected_bucket_idx] = 1;
        }
      }
    }

    for (size_t i = 0; i < num_buckets; ++i) {
      BlockScan(temp_storage)
        .ExclusiveSum(bucket_block_local_offsets[i],
                      bucket_block_local_offsets[i],
                      bucket_block_aggregate_sizes[i]);
    }

    if (threadIdx.x == 0) {
      for (size_t i = 0; i < num_buckets; ++i) {
        static_assert(sizeof(unsigned long long int) == sizeof(size_t));
        bucket_block_start_offsets[i] =
          atomicAdd(reinterpret_cast<unsigned long long int*>(bucket_sizes_ptr + i),
                    static_cast<unsigned long long int>(bucket_block_aggregate_sizes[i]));
      }
    }

    __syncthreads();

    // FIXME: better use shared memory buffer to aggreaget global memory writes
    if ((selected_bucket_idx != this_bucket_idx) && (selected_bucket_idx != invalid_bucket_idx)) {
      bucket_ptrs[selected_bucket_idx][bucket_block_start_offsets[selected_bucket_idx] +
                                       bucket_block_local_offsets[selected_bucket_idx]] = key;
    }

    idx += gridDim.x * blockDim.x;
    block_idx += gridDim.x;
  }
}

}  // namespace detail

template <typename vertex_t, bool is_multi_gpu = false>
class Bucket {
 public:
  Bucket(raft::handle_t const& handle, size_t capacity)
    : handle_ptr_(&handle), elements_(capacity, handle.get_stream())
  {
    thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                 elements_.begin(),
                 elements_.end(),
                 invalid_vertex_id<vertex_t>::value);
  }

  void insert(vertex_t v)
  {
    raft::update_device(elements_.data() + size_, &v, 1, handle_ptr_->get_stream());
    ++size_;
  }

  size_t size() const { return size_; }

  void set_size(size_t size) { size_ = size; }

  template <bool do_aggregate = is_multi_gpu>
  std::enable_if_t<do_aggregate, size_t> aggregate_size() const
  {
    return host_scalar_allreduce(handle_ptr_->get_comms(), size_, handle_ptr_->get_stream());
  }

  template <bool do_aggregate = is_multi_gpu>
  std::enable_if_t<!do_aggregate, size_t> aggregate_size() const
  {
    return size_;
  }

  void clear() { size_ = 0; }

  size_t capacity() const { return elements_.size(); }

  auto const data() const { return elements_.data(); }

  auto data() { return elements_.data(); }

  auto const begin() const { return elements_.begin(); }

  auto begin() { return elements_.begin(); }

  auto const end() const { return elements_.begin() + size_; }

  auto end() { return elements_.begin() + size_; }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::device_uvector<vertex_t> elements_;
  size_t size_{0};
};

template <typename vertex_t, bool is_multi_gpu = false, size_t num_buckets = 1>
class VertexFrontier {
 public:
  static size_t constexpr kNumBuckets = num_buckets;
  static size_t constexpr kInvalidBucketIdx{std::numeric_limits<size_t>::max()};

  VertexFrontier(raft::handle_t const& handle, std::vector<size_t> bucket_capacities)
    : handle_ptr_(&handle),
      tmp_bucket_ptrs_(num_buckets, handle.get_stream()),
      tmp_bucket_sizes_(num_buckets, handle.get_stream())
  {
    CUGRAPH_EXPECTS(bucket_capacities.size() == num_buckets,
                    "invalid input argument bucket_capacities (size mismatch)");
    thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                 tmp_bucket_ptrs_.begin(),
                 tmp_bucket_ptrs_.end(),
                 static_cast<vertex_t*>(nullptr));
    thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                 tmp_bucket_sizes_.begin(),
                 tmp_bucket_sizes_.end(),
                 size_t{0});
    for (size_t i = 0; i < num_buckets; ++i) {
      buckets_.emplace_back(handle, bucket_capacities[i]);
    }
  }

  Bucket<vertex_t, is_multi_gpu>& get_bucket(size_t bucket_idx) { return buckets_[bucket_idx]; }

  Bucket<vertex_t, is_multi_gpu> const& get_bucket(size_t bucket_idx) const
  {
    return buckets_[bucket_idx];
  }

  void swap_buckets(size_t bucket_idx0, size_t bucket_idx1)
  {
    std::swap(buckets_[bucket_idx0], buckets_[bucket_idx1]);
  }

  template <typename SplitOp>
  void split_bucket(size_t bucket_idx, SplitOp split_op)
  {
    auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

    auto bucket_and_bucket_size_device_ptrs = get_bucket_and_bucket_size_device_pointers();

    auto& this_bucket = get_bucket(bucket_idx);
    if (this_bucket.size() > 0) {
      raft::grid_1d_thread_t move_and_invalidate_if_grid(
        this_bucket.size(),
        detail::move_and_invalidate_if_block_size,
        handle_ptr_->get_device_properties().maxGridSize[0]);

      detail::move_and_invalidate_if<kNumBuckets>
        <<<move_and_invalidate_if_grid.num_blocks,
           move_and_invalidate_if_grid.block_size,
           0,
           handle_ptr_->get_stream()>>>(this_bucket.begin(),
                                        this_bucket.end(),
                                        std::get<0>(bucket_and_bucket_size_device_ptrs),
                                        std::get<1>(bucket_and_bucket_size_device_ptrs),
                                        bucket_idx,
                                        kInvalidBucketIdx,
                                        invalid_vertex,
                                        split_op);
    }

    // FIXME: if we adopt CUDA cooperative group https://devblogs.nvidia.com/cooperative-groups
    // and global sync(), we can merge this step with the above kernel (and rename the above kernel
    // to move_if)
    auto it =
      thrust::remove_if(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                        get_bucket(bucket_idx).begin(),
                        get_bucket(bucket_idx).end(),
                        [] __device__(auto value) { return value == invalid_vertex; });

    auto bucket_sizes_device_ptr = std::get<1>(bucket_and_bucket_size_device_ptrs);
    std::vector<size_t> bucket_sizes(kNumBuckets);
    raft::update_host(
      bucket_sizes.data(), bucket_sizes_device_ptr, kNumBuckets, handle_ptr_->get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle_ptr_->get_stream()));
    for (size_t i = 0; i < kNumBuckets; ++i) {
      if (i != bucket_idx) { get_bucket(i).set_size(bucket_sizes[i]); }
    }

    auto size = thrust::distance(get_bucket(bucket_idx).begin(), it);
    get_bucket(bucket_idx).set_size(size);

    return;
  }

  auto get_bucket_and_bucket_size_device_pointers()
  {
    std::vector<vertex_t*> tmp_ptrs(buckets_.size(), nullptr);
    std::vector<size_t> tmp_sizes(buckets_.size(), 0);
    for (size_t i = 0; i < buckets_.size(); ++i) {
      tmp_ptrs[i]  = get_bucket(i).data();
      tmp_sizes[i] = get_bucket(i).size();
    }
    raft::update_device(
      tmp_bucket_ptrs_.data(), tmp_ptrs.data(), tmp_ptrs.size(), handle_ptr_->get_stream());
    raft::update_device(
      tmp_bucket_sizes_.data(), tmp_sizes.data(), tmp_sizes.size(), handle_ptr_->get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle_ptr_->get_stream()));
    return std::make_tuple(tmp_bucket_ptrs_.data(), tmp_bucket_sizes_.data());
  }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  std::vector<Bucket<vertex_t, is_multi_gpu>> buckets_{};
  rmm::device_uvector<vertex_t*> tmp_bucket_ptrs_;
  rmm::device_uvector<size_t> tmp_bucket_sizes_;
};

}  // namespace experimental
}  // namespace cugraph
