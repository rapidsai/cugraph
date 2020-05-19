/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

// FIXME: better move this file to include/utilities (following cuDF) and rename to error.hpp
#include <utilities/error_utils.h>

#include <graph.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cinttypes>
#include <type_traits>
#include <vector>


namespace {

// FIXME: copied from include/cudf/detail/utilities/integer_utils.hpp, better move to another file
// for reusability
/**
 * Finds the smallest integer not less than `number_to_round` and modulo `S` is
 * zero. This function assumes that `number_to_round` is non-negative and
 * `modulus` is positive.
 */
template <typename S>
inline S round_up_safe(S number_to_round, S modulus) {
  auto remainder = number_to_round % modulus;
  if (remainder == 0) { return number_to_round; }
  auto rounded_up = number_to_round - remainder + modulus;
  if (rounded_up < number_to_round) {
    throw std::invalid_argument("Attempt to round up beyond the type's maximum value");
  }
  return rounded_up;
}

template <typename TupleType, size_t tuple_size>
struct compute_tuple_element_sizes {
  auto compute() {
    CUGRAPH_FAIL("unimplemented.");
    return std::array<size_t, 0>{};
  }
};

template <typename TupleType>
struct compute_tuple_element_sizes<TupleType, 1> {
  auto compute() {
    return std::array<size_t, 1>{sizeof(typename thrust::tuple_element<0, TupleType>::type)};
  }
};

template <typename TupleType>
struct compute_tuple_element_sizes<TupleType, 2> {
  auto compute() {
    return std::array<size_t, 2>{
      sizeof(typename thrust::tuple_element<0, TupleType>::type),
      sizeof(typename thrust::tuple_element<1, TupleType>::type)
    };
  }
};

template <typename TupleType, typename vertex_t, size_t tuple_size>
struct make_buffer_zip_iterator {
  auto make(std::vector<void*> buffer_ptrs, size_t offset) {
    CUGRAPH_FAIL("unimplemented.");
    return thrust::make_zip_iterator(
      thrust::make_tuple(reinterpret_cast<vertex_t*>(buffer_ptrs[0]) + offset));
  }
};

template <typename TupleType, typename vertex_t>
struct make_buffer_zip_iterator<TupleType, vertex_t, 0> {
  auto make(std::vector<void*> buffer_ptrs, size_t offset) {
    return thrust::make_zip_iterator(
      thrust::make_tuple(reinterpret_cast<vertex_t*>(buffer_ptrs[0]) + offset));
  }
};

template <typename TupleType, typename vertex_t>
struct make_buffer_zip_iterator<TupleType, vertex_t, 1> {
  auto make(std::vector<void*> buffer_ptrs, size_t offset) {
    return thrust::make_zip_iterator(
      thrust::make_tuple(
        reinterpret_cast<vertex_t*>(buffer_ptrs[0]) + offset,
        reinterpret_cast<typename thrust::tuple_element<0, TupleType>::type*>(
          buffer_ptrs[1]) + offset));
  }
};

template <typename TupleType, typename vertex_t>
struct make_buffer_zip_iterator<TupleType, vertex_t, 2> {
  auto make(std::vector<void*> buffer_ptrs, size_t offset) {
    return thrust::make_zip_iterator(
      thrust::make_tuple(
        reinterpret_cast<vertex_t*>(buffer_ptrs[0]) + offset,
        reinterpret_cast<typename thrust::tuple_element<0, TupleType>::type*>(
          buffer_ptrs[1]) + offset,
        reinterpret_cast<typename thrust::tuple_element<1, TupleType>::type*>(
          buffer_ptrs[2]) + offset));
  }
};

}

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType, typename vertex_t>
class Bucket {
 public:
  Bucket(HandleType const& handle, size_t capacity)
    : p_handle_(&handle), elements_(capacity, invalid_vertex_id<vertex_t>::value) {}

  void insert(vertex_t v) {
    elements_[size_] = v;
    ++size_;
  }

  size_t size() {
    return size_;
  }

  void set_size(size_t size) {
    size_ = size;
  }

  template <bool opg = HandleType::is_opg>
  std::enable_if_t<opg, size_t> aggregate_size() {
    CUGRAPH_FAIL("unimplemented.");
    return size_;
  }

  template <bool opg = HandleType::is_opg>
  std::enable_if_t<!opg, size_t> aggregate_size() {
    return size_;
  }

  size_t capacity() {
    return elements_.size();
  }

  auto const begin() const {
    return elements_.begin();
  }

  auto const end() const {
    return elements_.begin() + size_;
  }

 private:
  HandleType const* p_handle_{nullptr};
  rmm::device_vector<vertex_t> elements_{};
  size_t size_{0};
};

template <typename HandleType, typename ReduceInputTupleType, typename vertex_t>
class AdjMatrixRowFrontier {
 public:
  static size_t constexpr kInvalidBucketIdx{std::numeric_limits<size_t>::max()};

  AdjMatrixRowFrontier(HandleType const& handle, std::vector<size_t> bucket_capacities)
    : p_handle_(&handle),
      buffer_ptrs_(kReduceInputTupleSize + 1/* to store destination column number */, nullptr),
      buffer_idx_(0, p_handle_->get_default_stream()) {
    for (size_t i = 0; i < bucket_capacities.size(); ++i) {
      buckets_.emplace_back(handle, bucket_capacities[i]);
    }
    buffer_.set_stream(p_handle_->get_default_stream());
  }

  Bucket<HandleType, vertex_t>& get_bucket(size_t bucket_idx) {
    return buckets_[bucket_idx];
  }

  Bucket<HandleType, vertex_t> const& get_bucket(size_t bucket_idx) const {
    return buckets_[bucket_idx];
  }

  void resize_buffer(size_t size) {
    // FIXME: rmm::device_buffer resize incurs copy if memory is reallocated, which is unnecessary
    // in this case.
    buffer_.resize(
      compute_aggregate_buffer_size_in_bytes(size), p_handle_->get_default_stream());
    if (size > buffer_capacity_) {
      buffer_capacity_ = size;
      update_buffer_ptrs();
    }
    buffer_size_ = size;
  }

  void clear_buffer() {
    resize_buffer(0);
  }

  void shrink_to_fit_buffer() {
    if (buffer_size_ != buffer_capacity_) {
      // FIXME: rmm::device_buffer shrink_to_fit incurs copy if memory is reallocated, which is
      // unnecessary in this case.
      buffer_.shrink_to_fit(p_handle_->get_default_stream());
      update_buffer_ptrs();
      buffer_capacity_ = buffer_size_;
    }
  }

  auto buffer_begin() {
    return make_buffer_zip_iterator<ReduceInputTupleType, vertex_t, kReduceInputTupleSize>{}.make(
      buffer_ptrs_, 0);
  }

  auto buffer_end() {
    return make_buffer_zip_iterator<ReduceInputTupleType, vertex_t, kReduceInputTupleSize>{}.make(
      buffer_ptrs_, buffer_size_);
  }

  auto get_buffer_idx_ptr() {
    return buffer_idx_.data();
  }

  void set_buffer_idx_value(size_t value) {
    buffer_idx_.set_value(value, p_handle_->get_default_stream());
  }

 private:
  static size_t constexpr kReduceInputTupleSize = thrust::tuple_size<ReduceInputTupleType>::value;
  static size_t constexpr kBufferAlignment = 128;

  HandleType const* p_handle_{nullptr};
  std::vector<Bucket<HandleType, vertex_t>> buckets_{};

  std::array<size_t, kReduceInputTupleSize> tuple_element_sizes_ =
    compute_tuple_element_sizes<ReduceInputTupleType, kReduceInputTupleSize>{}.compute();
  std::vector<void*> buffer_ptrs_{};
  rmm::device_buffer buffer_{};
  size_t buffer_size_{0};
  size_t buffer_capacity_{0};
  rmm::device_scalar<size_t> buffer_idx_{};

  size_t compute_aggregate_buffer_size_in_bytes(size_t size) {
    size_t aggregate_buffer_size_in_bytes = round_up_safe(sizeof(vertex_t) * size, kBufferAlignment);
    for (size_t i = 0; i < kReduceInputTupleSize; ++i) {
      aggregate_buffer_size_in_bytes +=
        round_up_safe(tuple_element_sizes_[i] * size, kBufferAlignment);
    }
    return aggregate_buffer_size_in_bytes;
  }

  void update_buffer_ptrs() {
    uintptr_t ptr = reinterpret_cast<uintptr_t>(buffer_.data());
    buffer_ptrs_[0] = reinterpret_cast<void*>(ptr);
    ptr += round_up_safe(sizeof(vertex_t) * buffer_capacity_, kBufferAlignment);
    for (size_t i = 0; i < kReduceInputTupleSize; ++i) {
      buffer_ptrs_[i] = reinterpret_cast<void*>(ptr);
      ptr += round_up_safe(tuple_element_sizes_[i] * buffer_capacity_, kBufferAlignment);
    }
  }
};

}  // namesapce detail
}  // namespace experimental
}  // namespace cugraph