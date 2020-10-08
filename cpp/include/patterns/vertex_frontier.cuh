/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <utilities/comm_utils.cuh>
#include <utilities/error.hpp>
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

template <typename TupleType, typename vertex_t, size_t... Is>
auto make_buffer_zip_iterator_impl(std::vector<void*>& buffer_ptrs,
                                   size_t offset,
                                   std::index_sequence<Is...>)
{
  auto key_ptr    = reinterpret_cast<vertex_t*>(buffer_ptrs[0]) + offset;
  auto payload_it = thrust::make_zip_iterator(
    thrust::make_tuple(reinterpret_cast<typename thrust::tuple_element<Is, TupleType>::type*>(
      buffer_ptrs[1 + Is])...));
  return std::make_tuple(key_ptr, payload_it);
}

template <typename TupleType, typename vertex_t>
auto make_buffer_zip_iterator(std::vector<void*>& buffer_ptrs, size_t offset)
{
  size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
  return make_buffer_zip_iterator_impl<TupleType, vertex_t>(
    buffer_ptrs, offset, std::make_index_sequence<tuple_size>());
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
    : handle_ptr_(&handle), elements_(capacity, invalid_vertex_id<vertex_t>::value)
  {
  }

  void insert(vertex_t v)
  {
    elements_[size_] = v;
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

  auto const data() const { return elements_.data().get(); }

  auto data() { return elements_.data().get(); }

  auto const begin() const { return elements_.begin(); }

  auto begin() { return elements_.begin(); }

  auto const end() const { return elements_.begin() + size_; }

  auto end() { return elements_.begin() + size_; }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::device_vector<vertex_t> elements_{};
  size_t size_{0};
};

template <typename ReduceInputTupleType,
          typename vertex_t,
          bool is_multi_gpu  = false,
          size_t num_buckets = 1>
class VertexFrontier {
 public:
  static size_t constexpr kNumBuckets = num_buckets;
  static size_t constexpr kInvalidBucketIdx{std::numeric_limits<size_t>::max()};

  VertexFrontier(raft::handle_t const& handle, std::vector<size_t> bucket_capacities)
    : handle_ptr_(&handle),
      tmp_bucket_ptrs_(num_buckets, nullptr),
      tmp_bucket_sizes_(num_buckets, 0),
      buffer_ptrs_(kReduceInputTupleSize + 1 /* to store destination column number */, nullptr),
      buffer_idx_(0, handle_ptr_->get_stream())
  {
    CUGRAPH_EXPECTS(bucket_capacities.size() == num_buckets,
                    "invalid input argument bucket_capacities (size mismatch)");
    for (size_t i = 0; i < num_buckets; ++i) {
      buckets_.emplace_back(handle, bucket_capacities[i]);
    }
    buffer_.set_stream(handle_ptr_->get_stream());
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
                                        std::get<0>(bucket_and_bucket_size_device_ptrs).get(),
                                        std::get<1>(bucket_and_bucket_size_device_ptrs).get(),
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
    thrust::host_vector<size_t> bucket_sizes(bucket_sizes_device_ptr,
                                             bucket_sizes_device_ptr + kNumBuckets);
    for (size_t i = 0; i < kNumBuckets; ++i) {
      if (i != bucket_idx) { get_bucket(i).set_size(bucket_sizes[i]); }
    }

    auto size = thrust::distance(get_bucket(bucket_idx).begin(), it);
    get_bucket(bucket_idx).set_size(size);

    return;
  }

  auto get_bucket_and_bucket_size_device_pointers()
  {
    thrust::host_vector<vertex_t*> tmp_ptrs(buckets_.size(), nullptr);
    thrust::host_vector<size_t> tmp_sizes(buckets_.size(), 0);
    for (size_t i = 0; i < buckets_.size(); ++i) {
      tmp_ptrs[i]  = get_bucket(i).data();
      tmp_sizes[i] = get_bucket(i).size();
    }
    tmp_bucket_ptrs_  = tmp_ptrs;
    tmp_bucket_sizes_ = tmp_sizes;
    return std::make_tuple(tmp_bucket_ptrs_.data(), tmp_bucket_sizes_.data());
  }

  void resize_buffer(size_t size)
  {
    // FIXME: rmm::device_buffer resize incurs copy if memory is reallocated, which is unnecessary
    // in this case.
    buffer_.resize(compute_aggregate_buffer_size_in_bytes(size), handle_ptr_->get_stream());
    if (size > buffer_capacity_) {
      buffer_capacity_ = size;
      update_buffer_ptrs();
    }
    buffer_size_ = size;
  }

  void clear_buffer() { resize_buffer(0); }

  void shrink_to_fit_buffer()
  {
    if (buffer_size_ != buffer_capacity_) {
      // FIXME: rmm::device_buffer shrink_to_fit incurs copy if memory is reallocated, which is
      // unnecessary in this case.
      buffer_.shrink_to_fit(handle_ptr_->get_stream());
      update_buffer_ptrs();
      buffer_capacity_ = buffer_size_;
    }
  }

  auto buffer_begin()
  {
    return detail::make_buffer_zip_iterator<ReduceInputTupleType, vertex_t>(buffer_ptrs_, 0);
  }

  auto buffer_end()
  {
    return detail::make_buffer_zip_iterator<ReduceInputTupleType, vertex_t>(buffer_ptrs_,
                                                                            buffer_size_);
  }

  auto get_buffer_idx_ptr() { return buffer_idx_.data(); }

  size_t get_buffer_idx_value() { return buffer_idx_.value(handle_ptr_->get_stream()); }

  void set_buffer_idx_value(size_t value)
  {
    buffer_idx_.set_value(value, handle_ptr_->get_stream());
  }

 private:
  static size_t constexpr kReduceInputTupleSize = thrust::tuple_size<ReduceInputTupleType>::value;
  static size_t constexpr kBufferAlignment      = 128;

  raft::handle_t const* handle_ptr_{nullptr};
  std::vector<Bucket<vertex_t, is_multi_gpu>> buckets_{};
  rmm::device_vector<vertex_t*> tmp_bucket_ptrs_{};
  rmm::device_vector<size_t> tmp_bucket_sizes_{};

  std::array<size_t, kReduceInputTupleSize> tuple_element_sizes_ =
    compute_thrust_tuple_element_sizes<ReduceInputTupleType>()();
  std::vector<void*> buffer_ptrs_{};
  rmm::device_buffer buffer_{};
  size_t buffer_size_{0};
  size_t buffer_capacity_{0};
  rmm::device_scalar<size_t> buffer_idx_{};

  // FIXME: better pick between this apporach or the approach used in allocate_comm_buffer
  size_t compute_aggregate_buffer_size_in_bytes(size_t size)
  {
    size_t aggregate_buffer_size_in_bytes =
      detail::round_up(sizeof(vertex_t) * size, kBufferAlignment);
    for (size_t i = 0; i < kReduceInputTupleSize; ++i) {
      aggregate_buffer_size_in_bytes +=
        detail::round_up(tuple_element_sizes_[i] * size, kBufferAlignment);
    }
    return aggregate_buffer_size_in_bytes;
  }

  void update_buffer_ptrs()
  {
    uintptr_t ptr   = reinterpret_cast<uintptr_t>(buffer_.data());
    buffer_ptrs_[0] = reinterpret_cast<void*>(ptr);
    ptr += detail::round_up(sizeof(vertex_t) * buffer_capacity_, kBufferAlignment);
    for (size_t i = 0; i < kReduceInputTupleSize; ++i) {
      buffer_ptrs_[1 + i] = reinterpret_cast<void*>(ptr);
      ptr += detail::round_up(tuple_element_sizes_[i] * buffer_capacity_, kBufferAlignment);
    }
  }
};

}  // namespace experimental
}  // namespace cugraph
