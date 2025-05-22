/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "prims/detail/optional_dataframe_buffer.hpp"

#include <cugraph/graph.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cuco/static_map.cuh>

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>

// FIXME: this can be used in edge_partition_device_view_t &
// edge_partition_endpoint_property_device_view_t as well but this requires placing this header
// under cpp/include/cugraph; this requires exposing cuco in the public interface and is currently
// problematic.

namespace cugraph {

namespace detail {

using cuco_storage_type = cuco::storage<1>;  ///< cuco window storage type

template <typename KeyIterator, typename ValueIterator>
struct kv_binary_search_find_op_t {
  using key_type   = typename thrust::iterator_traits<KeyIterator>::value_type;
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  KeyIterator store_key_first{};
  KeyIterator store_key_last{};
  ValueIterator store_value_first{};

  value_type invalid_value{};

  __device__ value_type operator()(key_type key) const
  {
    auto it = thrust::lower_bound(thrust::seq, store_key_first, store_key_last, key);
    if (it != store_key_last && *it == key) {
      return *(store_value_first + cuda::std::distance(store_key_first, it));
    } else {
      return invalid_value;
    }
  }
};

template <typename KeyIterator>
struct kv_binary_search_contains_op_t {
  using key_type = typename thrust::iterator_traits<KeyIterator>::value_type;

  KeyIterator store_key_first{};
  KeyIterator store_key_last{};

  __device__ bool operator()(key_type key) const
  {
    return thrust::binary_search(thrust::seq, store_key_first, store_key_last, key);
  }
};

template <typename RefType, typename KeyIterator>
struct kv_cuco_insert_and_increment_t {
  RefType device_ref{};
  KeyIterator key_first{};
  size_t* counter{nullptr};
  size_t invalid_idx{};

  __device__ size_t operator()(size_t i)
  {
    auto pair             = thrust::make_tuple(*(key_first + i), size_t{0} /* dummy */);
    auto [iter, inserted] = device_ref.insert_and_find(pair);
    if (inserted) {
      cuda::atomic_ref<size_t, cuda::thread_scope_device> atomic_counter(*counter);
      auto idx = atomic_counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
      cuda::atomic_ref<typename RefType::mapped_type, cuda::thread_scope_device> ref(
        (*iter).second);
      ref.store(idx, cuda::std::memory_order_relaxed);
      return idx;
    } else {
      return invalid_idx;
    }
  }
};

template <typename RefType, typename KeyIterator, typename StencilIterator, typename PredOp>
struct kv_cuco_insert_if_and_increment_t {
  RefType device_ref{};
  KeyIterator key_first{};
  StencilIterator stencil_first{};
  PredOp pred_op{};
  size_t* counter{nullptr};
  size_t invalid_idx{};

  __device__ size_t operator()(size_t i)
  {
    if (pred_op(*(stencil_first + i)) == false) { return invalid_idx; }

    auto pair             = thrust::make_tuple(*(key_first + i), size_t{0} /* dummy */);
    auto [iter, inserted] = device_ref.insert_and_find(pair);
    if (inserted) {
      cuda::atomic_ref<size_t, cuda::thread_scope_device> atomic_counter(*counter);
      auto idx = atomic_counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
      cuda::atomic_ref<typename RefType::mapped_type, cuda::thread_scope_device> ref(
        (*iter).second);
      ref.store(idx, cuda::std::memory_order_relaxed);
      return idx;
    } else {
      return invalid_idx;
    }
  }
};

template <typename RefType, typename key_t, typename value_t>
struct kv_cuco_insert_and_assign_t {
  RefType device_ref{};

  __device__ void operator()(thrust::tuple<key_t, value_t> pair)
  {
    auto [iter, inserted] = device_ref.insert_and_find(pair);
    if (!inserted) {
      cuda::atomic_ref<typename RefType::mapped_type, cuda::thread_scope_device> ref(
        (*iter).second);
      ref.store(thrust::get<1>(pair), cuda::std::memory_order_relaxed);
    }
  }
};

template <typename ViewType>
struct kv_binary_search_store_device_view_t {
  using key_type   = typename ViewType::key_type;
  using value_type = typename ViewType::value_type;

  static_assert(ViewType::binary_search);

  __host__ kv_binary_search_store_device_view_t(ViewType view)
    : store_key_first(view.store_key_first()),
      store_key_last(view.store_key_last()),
      store_value_first(view.store_value_first()),
      invalid_value(view.invalid_value())
  {
  }

  __device__ value_type find(key_type key) const
  {
    auto it = thrust::lower_bound(thrust::seq, store_key_first, store_key_last, key);
    if (it != store_key_last && *it == key) {
      return *(store_value_first + cuda::std::distance(store_key_first, it));
    } else {
      return invalid_value;
    }
  }

  typename ViewType::key_iterator store_key_first{};
  typename ViewType::key_iterator store_key_last{};
  typename ViewType::value_iterator store_value_first{};

  value_type invalid_value{};
};

template <typename ViewType>
struct kv_cuco_store_find_device_view_t {
  using key_type                   = typename ViewType::key_type;
  using value_type                 = typename ViewType::value_type;
  using cuco_store_device_ref_type = typename ViewType::cuco_map_type::ref_type<cuco::find_tag>;

  static_assert(!ViewType::binary_search);

  __host__ kv_cuco_store_find_device_view_t(ViewType view)
    : cuco_store_device_ref(view.cuco_store_find_device_ref())
  {
    if constexpr (std::is_arithmetic_v<value_type>) {
      invalid_value = cuco_store_device_ref.empty_value_sentinel();
    } else {
      store_value_first = view.store_value_first();
      invalid_value     = view.invalid_value();
    }
  }

  __device__ value_type find(key_type key) const
  {
    auto found = cuco_store_device_ref.find(key);
    if (found == cuco_store_device_ref.end()) {
      return invalid_value;
    } else {
      auto val = (*found).second;
      if constexpr (std::is_arithmetic_v<value_type>) {
        return val;
      } else {
        return *(store_value_first + val);
      }
    }
  }

  cuco_store_device_ref_type cuco_store_device_ref{};
  std::conditional_t<!std::is_arithmetic_v<value_type>,
                     typename ViewType::value_iterator,
                     std::byte /* dummy */>
    store_value_first{};

  value_type invalid_value{};
};

template <typename KeyIterator, typename ValueIterator>
class kv_binary_search_store_view_t {
 public:
  using key_type   = std::remove_cv_t<typename thrust::iterator_traits<KeyIterator>::value_type>;
  using value_type = std::remove_cv_t<typename thrust::iterator_traits<ValueIterator>::value_type>;
  using key_iterator   = KeyIterator;
  using value_iterator = ValueIterator;

  static constexpr bool binary_search = true;

  kv_binary_search_store_view_t(KeyIterator key_first,
                                KeyIterator key_last,
                                ValueIterator value_first,
                                value_type invalid_value)
    : store_key_first_(key_first),
      store_key_last_(key_last),
      store_value_first_(value_first),
      invalid_value_(invalid_value)
  {
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void find(QueryKeyIterator key_first,
            QueryKeyIterator key_last,
            ResultValueIterator value_first,
            rmm::cuda_stream_view stream) const
  {
    thrust::transform(rmm::exec_policy(stream),
                      key_first,
                      key_last,
                      value_first,
                      kv_binary_search_find_op_t<KeyIterator, ValueIterator>{
                        store_key_first_, store_key_last_, store_value_first_, invalid_value_});
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void contains(QueryKeyIterator key_first,
                QueryKeyIterator key_last,
                ResultValueIterator value_first,
                rmm::cuda_stream_view stream) const
  {
    thrust::transform(
      rmm::exec_policy(stream),
      key_first,
      key_last,
      value_first,
      kv_binary_search_contains_op_t<KeyIterator>{store_key_first_, store_key_last_});
  }

  KeyIterator store_key_first() const { return store_key_first_; }

  KeyIterator store_key_last() const { return store_key_last_; }

  ValueIterator store_value_first() const { return store_value_first_; }

  value_type invalid_value() const { return invalid_value_; }

 private:
  KeyIterator store_key_first_{};
  KeyIterator store_key_last_{};
  ValueIterator store_value_first_{};

  value_type invalid_value_{};
};

template <typename key_t, typename ValueIterator>
class kv_cuco_store_view_t {
 public:
  using key_type   = key_t;
  using value_type = std::remove_cv_t<typename thrust::iterator_traits<ValueIterator>::value_type>;
  using value_iterator = ValueIterator;

  static constexpr bool binary_search = false;

  using cuco_map_type =
    cuco::static_map<key_t,
                     std::conditional_t<std::is_arithmetic_v<value_type>, value_type, size_t>,
                     cuco::extent<std::size_t>,
                     cuda::thread_scope_device,
                     thrust::equal_to<key_t>,
                     cuco::linear_probing<1,  // CG size
                                          cuco::murmurhash3_32<key_t>>,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<std::byte>>,
                     cuco_storage_type>;

  template <typename type = value_type>
  kv_cuco_store_view_t(cuco_map_type const* store,
                       std::enable_if_t<std::is_arithmetic_v<type>, int32_t> = 0)
    : cuco_store_(store)
  {
  }

  template <typename type = value_type>
  kv_cuco_store_view_t(cuco_map_type const* store,
                       ValueIterator value_first,
                       type invalid_value,
                       std::enable_if_t<!std::is_arithmetic_v<type>, int32_t> = 0)
    : cuco_store_(store), store_value_first_(value_first), invalid_value_(invalid_value)
  {
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void find(QueryKeyIterator key_first,
            QueryKeyIterator key_last,
            ResultValueIterator value_first,
            rmm::cuda_stream_view stream) const
  {
    if constexpr (std::is_arithmetic_v<value_type>) {
      cuco_store_->find(key_first, key_last, value_first, stream.value());
    } else {
      rmm::device_uvector<size_t> indices(cuda::std::distance(key_first, key_last), stream);
      auto invalid_idx = cuco_store_->empty_value_sentinel();
      cuco_store_->find(key_first, key_last, indices.begin(), stream.value());
      thrust::transform(rmm::exec_policy(stream),
                        indices.begin(),
                        indices.end(),
                        value_first,
                        indirection_if_idx_valid_t<size_t, decltype(store_value_first_)>{
                          store_value_first_, invalid_idx, invalid_value_});
    }
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void contains(QueryKeyIterator key_first,
                QueryKeyIterator key_last,
                ResultValueIterator value_first,
                rmm::cuda_stream_view stream) const
  {
    cuco_store_->contains(key_first, key_last, value_first, stream.value());
  }

  auto cuco_store_find_device_ref() const { return cuco_store_->ref(cuco::find); }

  template <typename type = value_type>
  std::enable_if_t<!std::is_arithmetic_v<type>, ValueIterator> store_value_first() const
  {
    return store_value_first_;
  }

  key_t invalid_key() const { return cuco_store_->empty_key_sentinel(); }

  value_type invalid_value() const
  {
    if constexpr (std::is_arithmetic_v<value_type>) {
      return cuco_store_->empty_value_sentinel();
    } else {
      return invalid_value_;
    }
  }

 private:
  cuco_map_type const* cuco_store_{};
  std::conditional_t<!std::is_arithmetic_v<value_type>, ValueIterator, std::byte /* dummy */>
    store_value_first_{};

  std::conditional_t<!std::is_arithmetic_v<value_type>, value_type, std::byte /* dummy */>
    invalid_value_{};
};

template <typename key_t, typename value_t>
class kv_binary_search_store_t {
 public:
  using key_type   = key_t;
  using value_type = value_t;

  kv_binary_search_store_t(rmm::cuda_stream_view stream)
    : store_keys_(0, stream), store_values_(allocate_dataframe_buffer<value_t>(0, stream))
  {
  }

  template <typename KeyIterator, typename ValueIterator>
  kv_binary_search_store_t(
    KeyIterator key_first,
    KeyIterator key_last,
    ValueIterator value_first,
    value_t invalid_value /* invalid_value is returned when match fails for the given key */,
    bool key_sorted /* if set to true, assume that the input data is sorted and skip sorting (which
                       is necessary for binary-search) */
    ,
    rmm::cuda_stream_view stream)
    : store_keys_(static_cast<size_t>(cuda::std::distance(key_first, key_last)), stream),
      store_values_(allocate_dataframe_buffer<value_t>(
        static_cast<size_t>(cuda::std::distance(key_first, key_last)), stream)),
      invalid_value_(invalid_value)
  {
    thrust::copy(rmm::exec_policy(stream), key_first, key_last, store_keys_.begin());
    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    thrust::copy(rmm::exec_policy(stream),
                 value_first,
                 value_first + num_keys,
                 get_dataframe_buffer_begin(store_values_));
    if (!key_sorted) {
      thrust::sort_by_key(rmm::exec_policy(stream),
                          store_keys_.begin(),
                          store_keys_.end(),
                          get_dataframe_buffer_begin(store_values_));
    }
  }

  kv_binary_search_store_t(
    rmm::device_uvector<key_t>&& keys,
    decltype(allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))&& values,
    value_t invalid_value /* invalid_value is returned when match fails for the given key */,
    bool key_sorted /* if set to true, assume that the input data is sorted and skip sorting (which
                       is necessary for binary-search) */
    ,
    rmm::cuda_stream_view stream)
    : store_keys_(std::move(keys)), store_values_(std::move(values)), invalid_value_(invalid_value)
  {
    if (!key_sorted) {
      thrust::sort_by_key(rmm::exec_policy(stream),
                          store_keys_.begin(),
                          store_keys_.end(),
                          get_dataframe_buffer_begin(store_values_));
    }
  }

  auto retrieve_all(rmm::cuda_stream_view stream)
  {
    rmm::device_uvector<key_t> tmp_store_keys(store_keys_.size(), stream);
    auto tmp_store_values =
      allocate_dataframe_buffer<value_t>(size_dataframe_buffer(store_values_), stream);
    thrust::copy(
      rmm::exec_policy(stream), store_keys_.begin(), store_keys_.end(), tmp_store_keys.begin());
    thrust::copy(rmm::exec_policy(stream),
                 get_dataframe_buffer_begin(store_values_),
                 get_dataframe_buffer_end(store_values_),
                 get_dataframe_buffer_begin(tmp_store_values));
    return std::make_tuple(std::move(tmp_store_keys), std::move(tmp_store_values));
  }

  auto release(rmm::cuda_stream_view stream)
  {
    auto tmp_store_keys   = std::move(store_keys_);
    auto tmp_store_values = std::move(store_values_);
    store_keys_           = rmm::device_uvector<key_t>(0, stream);
    store_values_         = allocate_dataframe_buffer<value_t>(0, stream);
    return std::make_tuple(std::move(tmp_store_keys), std::move(tmp_store_values));
  }

  key_t const* store_key_first() const { return store_keys_.cbegin(); }

  key_t const* store_key_last() const { return store_keys_.cend(); }

  auto store_value_first() const { return get_dataframe_buffer_cbegin(store_values_); }

  value_t invalid_value() const { return invalid_value_; }

  size_t size() const { return store_keys_.size(); }

  size_t capacity() const { return store_keys_.size(); }

 private:
  rmm::device_uvector<key_t> store_keys_;
  decltype(allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{})) store_values_;

  value_t invalid_value_{};
};

template <typename key_t, typename value_t>
class kv_cuco_store_t {
 public:
  using key_type   = key_t;
  using value_type = value_t;
  using value_buffer_type =
    decltype(allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}));
  using const_value_iterator =
    std::invoke_result_t<decltype(get_dataframe_buffer_cbegin<value_buffer_type>),
                         value_buffer_type&>;

  using cuco_map_type =
    cuco::static_map<key_t,
                     std::conditional_t<std::is_arithmetic_v<value_t>, value_t, size_t>,
                     cuco::extent<std::size_t>,
                     cuda::thread_scope_device,
                     thrust::equal_to<key_t>,
                     cuco::linear_probing<1,  // CG size
                                          cuco::murmurhash3_32<key_t>>,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<std::byte>>,
                     cuco_storage_type>;

  kv_cuco_store_t(rmm::cuda_stream_view stream)
    : store_values_(allocate_optional_dataframe_buffer<
                    std::conditional_t<!std::is_arithmetic_v<value_t>, value_t, void>>(0, stream))
  {
  }

  kv_cuco_store_t(size_t capacity,
                  key_t invalid_key,
                  value_t invalid_value,
                  rmm::cuda_stream_view stream)
    : store_values_(allocate_optional_dataframe_buffer<
                    std::conditional_t<!std::is_arithmetic_v<value_t>, value_t, void>>(0, stream))
  {
    allocate(capacity, invalid_key, invalid_value, stream);
    if constexpr (!std::is_arithmetic_v<value_t>) { invalid_value_ = invalid_value; }
    capacity_ = capacity;
    size_     = 0;
  }

  template <typename KeyIterator, typename ValueIterator>
  kv_cuco_store_t(KeyIterator key_first,
                  KeyIterator key_last,
                  ValueIterator value_first,
                  key_t invalid_key,
                  value_t invalid_value,
                  rmm::cuda_stream_view stream)
    : store_values_(allocate_optional_dataframe_buffer<
                    std::conditional_t<!std::is_arithmetic_v<value_t>, value_t, void>>(0, stream))
  {
    static_assert(std::is_same_v<typename thrust::iterator_traits<KeyIterator>::value_type, key_t>);
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t>);

    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    allocate(num_keys, invalid_key, invalid_value, stream);
    if constexpr (!std::is_arithmetic_v<value_t>) { invalid_value_ = invalid_value; }
    capacity_ = num_keys;
    size_     = 0;

    insert(key_first, key_last, value_first, stream);
  }

  template <typename KeyIterator, typename ValueIterator>
  void insert(KeyIterator key_first,
              KeyIterator key_last,
              ValueIterator value_first,
              rmm::cuda_stream_view stream)
  {
    static_assert(std::is_same_v<typename thrust::iterator_traits<KeyIterator>::value_type, key_t>);
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t>);

    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    if (num_keys == 0) return;

    if constexpr (std::is_arithmetic_v<value_t>) {
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(key_first, value_first));
      size_ += cuco_store_->insert(pair_first, pair_first + num_keys, stream.value());
    } else {
      auto old_store_value_size = size_optional_dataframe_buffer<value_t>(store_values_);
      // FIXME: we can use cuda::atomic instead but currently on a system with x86 + GPU, this
      // requires placing the atomic variable on managed memory and this adds additional
      // complication.
      rmm::device_scalar<size_t> counter(old_store_value_size, stream);
      auto mutable_device_ref = cuco_store_->ref(cuco::insert_and_find);
      rmm::device_uvector<size_t> store_value_offsets(num_keys, stream);
      thrust::tabulate(
        rmm::exec_policy(stream),
        store_value_offsets.begin(),
        store_value_offsets.end(),
        kv_cuco_insert_and_increment_t<decltype(mutable_device_ref), KeyIterator>{
          mutable_device_ref, key_first, counter.data(), std::numeric_limits<size_t>::max()});
      size_ = counter.value(stream);
      resize_optional_dataframe_buffer<value_t>(store_values_, size_, stream);
      thrust::scatter_if(rmm::exec_policy(stream),
                         value_first,
                         value_first + num_keys,
                         store_value_offsets.begin() /* map */,
                         store_value_offsets.begin() /* stencil */,
                         get_optional_dataframe_buffer_begin<value_t>(store_values_),
                         is_not_equal_t<size_t>{std::numeric_limits<size_t>::max()});
    }
  }

  template <typename KeyIterator, typename ValueIterator, typename StencilIterator, typename PredOp>
  void insert_if(KeyIterator key_first,
                 KeyIterator key_last,
                 ValueIterator value_first,
                 StencilIterator stencil_first,
                 PredOp pred_op,
                 rmm::cuda_stream_view stream)
  {
    static_assert(std::is_same_v<typename thrust::iterator_traits<KeyIterator>::value_type, key_t>);
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t>);

    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    if (num_keys == 0) return;

    if constexpr (std::is_arithmetic_v<value_t>) {
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(key_first, value_first));
      size_ += cuco_store_->insert_if(
        pair_first, pair_first + num_keys, stencil_first, pred_op, stream.value());
    } else {
      auto old_store_value_size = size_optional_dataframe_buffer<value_t>(store_values_);
      // FIXME: we can use cuda::atomic instead but currently on a system with x86 + GPU, this
      // requires placing the atomic variable on managed memory and this adds additional
      // complication.
      rmm::device_scalar<size_t> counter(old_store_value_size, stream);
      auto mutable_device_ref = cuco_store_->ref(cuco::insert_and_find);
      rmm::device_uvector<size_t> store_value_offsets(num_keys, stream);
      thrust::tabulate(
        rmm::exec_policy(stream),
        store_value_offsets.begin(),
        store_value_offsets.end(),
        kv_cuco_insert_if_and_increment_t<decltype(mutable_device_ref),
                                          KeyIterator,
                                          StencilIterator,
                                          PredOp>{mutable_device_ref,
                                                  key_first,
                                                  stencil_first,
                                                  pred_op,
                                                  counter.data(),
                                                  std::numeric_limits<size_t>::max()});
      size_ = counter.value(stream);
      resize_optional_dataframe_buffer<value_t>(store_values_, size_, stream);
      thrust::scatter_if(rmm::exec_policy(stream),
                         value_first,
                         value_first + num_keys,
                         store_value_offsets.begin() /* map */,
                         store_value_offsets.begin() /* stencil */,
                         get_optional_dataframe_buffer_begin<value_t>(store_values_),
                         is_not_equal_t<size_t>{std::numeric_limits<size_t>::max()});
    }
  }

  template <typename KeyIterator, typename ValueIterator>
  void insert_and_assign(KeyIterator key_first,
                         KeyIterator key_last,
                         ValueIterator value_first,
                         rmm::cuda_stream_view stream)
  {
    static_assert(std::is_same_v<typename thrust::iterator_traits<KeyIterator>::value_type, key_t>);
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t>);

    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    if (num_keys == 0) return;

    if constexpr (std::is_arithmetic_v<value_t>) {
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(key_first, value_first));
      // FIXME: a temporary solution till insert_and_assign is added to
      // cuco::static_map
      auto mutable_device_ref = cuco_store_->ref(cuco::insert_and_find);
      thrust::for_each(
        rmm::exec_policy(stream),
        pair_first,
        pair_first + num_keys,
        detail::kv_cuco_insert_and_assign_t<decltype(mutable_device_ref), key_t, value_t>{
          mutable_device_ref});
      // FIXME: this is an upper bound of size_, as some inserts may fail due to existing keys
      size_ += num_keys;
    } else {
      auto old_store_value_size = size_optional_dataframe_buffer<value_t>(store_values_);
      // FIXME: we can use cuda::atomic instead but currently on a system with x86 + GPU, this
      // requires placing the atomic variable on managed memory and this adds additional
      // complication.
      rmm::device_scalar<size_t> counter(old_store_value_size, stream);
      auto mutable_device_ref = cuco_store_->ref(cuco::insert_and_find);
      rmm::device_uvector<size_t> store_value_offsets(num_keys, stream);
      thrust::tabulate(
        rmm::exec_policy(stream),
        store_value_offsets.begin(),
        store_value_offsets.end(),
        kv_cuco_insert_and_increment_t<decltype(mutable_device_ref), KeyIterator>{
          mutable_device_ref, key_first, counter.data(), std::numeric_limits<size_t>::max()});
      size_ = counter.value(stream);
      resize_optional_dataframe_buffer<value_t>(store_values_, size_, stream);
      thrust::scatter_if(rmm::exec_policy(stream),
                         value_first,
                         value_first + num_keys,
                         store_value_offsets.begin() /* map */,
                         store_value_offsets.begin() /* stencil */,
                         get_optional_dataframe_buffer_begin<value_t>(store_values_),
                         is_not_equal_t<size_t>{std::numeric_limits<size_t>::max()});

      // now perform assigns (for k,v pairs that failed to insert)

      rmm::device_uvector<size_t> kv_indices(num_keys, stream);
      thrust::sequence(rmm::exec_policy(), kv_indices.begin(), kv_indices.end(), size_t{0});
      auto pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(store_value_offsets.begin(), kv_indices.begin()));
      kv_indices.resize(
        cuda::std::distance(
          pair_first,
          thrust::remove_if(rmm::exec_policy(stream),
                            pair_first,
                            pair_first + num_keys,
                            [invalid_idx = std::numeric_limits<size_t>::max()] __device__(
                              auto pair) { return thrust::get<0>(pair) != invalid_idx; })),
        stream);
      store_value_offsets.resize(0, stream);
      store_value_offsets.shrink_to_fit(stream);

      thrust::sort(rmm::exec_policy(stream),
                   kv_indices.begin(),
                   kv_indices.end(),
                   [key_first] __device__(auto lhs, auto rhs) {
                     return *(key_first + lhs) < *(key_first + rhs);
                   });
      kv_indices.resize(
        cuda::std::distance(kv_indices.begin(),
                            thrust::unique(rmm::exec_policy(stream),
                                           kv_indices.begin(),
                                           kv_indices.end(),
                                           [key_first] __device__(auto lhs, auto rhs) {
                                             return *(key_first + lhs) == *(key_first + rhs);
                                           })),
        stream);

      thrust::for_each(
        rmm::exec_policy(stream),
        kv_indices.begin(),
        kv_indices.end(),
        [key_first,
         value_first,
         store_value_first = get_optional_dataframe_buffer_begin<value_t>(store_values_),
         device_ref        = cuco_store_->ref(cuco::find)] __device__(auto kv_idx) {
          size_t store_value_offset{};
          auto found = device_ref.find(*(key_first + kv_idx));
          assert(found != device_ref.end());
          store_value_offset                        = (*found).second;
          *(store_value_first + store_value_offset) = *(value_first + kv_idx);
        });
    }
  }

  auto retrieve_all(rmm::cuda_stream_view stream)
  {
    rmm::device_uvector<key_t> keys(size_, stream);
    auto values = allocate_dataframe_buffer<value_t>(0, stream);
    if constexpr (std::is_arithmetic_v<value_t>) {
      values.resize(size_, stream);
      auto pair_last = cuco_store_->retrieve_all(keys.begin(), values.begin(), stream.value());
      // FIXME: this resize (& shrink_to_fit) shouldn't be necessary if size_ is exact
      keys.resize(cuda::std::distance(keys.begin(), std::get<0>(pair_last)), stream);
      values.resize(keys.size(), stream);
    } else {
      rmm::device_uvector<size_t> indices(size_, stream);
      auto pair_last = cuco_store_->retrieve_all(keys.begin(), indices.begin(), stream.value());
      // FIXME: this resize (& shrink_to_fit) shouldn't be necessary if size_ is exact
      keys.resize(cuda::std::distance(keys.begin(), std::get<0>(pair_last)), stream);
      indices.resize(keys.size(), stream);
      resize_dataframe_buffer(values, keys.size(), stream);
      thrust::gather(rmm::exec_policy(stream),
                     indices.begin(),
                     indices.end(),
                     get_optional_dataframe_buffer_begin<value_t>(store_values_),
                     get_dataframe_buffer_begin(values));
    }
    return std::make_tuple(std::move(keys), std::move(values));
  }

  auto release(rmm::cuda_stream_view stream)
  {
    auto [retrieved_keys, retrieved_values] = retrieve_all(stream);
    allocate(0, invalid_key(), invalid_value(), stream);
    capacity_ = 0;
    size_     = 0;
    return std::make_tuple(std::move(retrieved_keys), std::move(retrieved_values));
  }

  cuco_map_type const* cuco_store_ptr() const { return cuco_store_.get(); }

  template <typename type = value_t>
  std::enable_if_t<!std::is_arithmetic_v<type>, const_value_iterator> store_value_first() const
  {
    return get_optional_dataframe_buffer_cbegin<value_t>(store_values_);
  }

  key_t invalid_key() const { return cuco_store_->empty_key_sentinel(); }

  value_t invalid_value() const
  {
    if constexpr (std::is_arithmetic_v<value_t>) {
      return cuco_store_->empty_value_sentinel();
    } else {
      return invalid_value_;
    }
  }

  // FIXME: currently this returns an upper-bound
  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

 private:
  void allocate(size_t num_keys,
                key_t invalid_key,
                value_t invalid_value,
                rmm::cuda_stream_view stream)
  {
    double constexpr load_factor = 0.7;
    auto cuco_size               = std::max(
      static_cast<size_t>(static_cast<double>(num_keys) / load_factor),
      static_cast<size_t>(num_keys) + 1);  // cuco::static_map requires at least one empty slot

    auto stream_adapter = rmm::mr::stream_allocator_adaptor(
      rmm::mr::polymorphic_allocator<std::byte>(rmm::mr::get_current_device_resource()), stream);
    if constexpr (std::is_arithmetic_v<value_t>) {
      cuco_store_ =
        std::make_unique<cuco_map_type>(cuco_size,
                                        cuco::empty_key<key_t>{invalid_key},
                                        cuco::empty_value<value_t>{invalid_value},
                                        thrust::equal_to<key_t>{},
                                        cuco::linear_probing<1,  // CG size
                                                             cuco::murmurhash3_32<key_t>>{},
                                        cuco::thread_scope_device,
                                        cuco_storage_type{},
                                        stream_adapter,
                                        stream.value());
    } else {
      cuco_store_ = std::make_unique<cuco_map_type>(
        cuco_size,
        cuco::empty_key<key_t>{invalid_key},
        cuco::empty_value<size_t>{std::numeric_limits<size_t>::max()},
        thrust::equal_to<key_t>{},
        cuco::linear_probing<1,  // CG size
                             cuco::murmurhash3_32<key_t>>{},
        cuco::thread_scope_device,
        cuco_storage_type{},
        stream_adapter,
        stream.value());
      reserve_optional_dataframe_buffer<value_t>(store_values_, num_keys, stream);
    }
  }

  std::unique_ptr<cuco_map_type> cuco_store_{nullptr};
  decltype(allocate_optional_dataframe_buffer<
           std::conditional_t<!std::is_arithmetic_v<value_t>, value_t, void>>(
    0, rmm::cuda_stream_view{})) store_values_;

  std::conditional_t<!std::is_arithmetic_v<value_t>, value_t, std::byte /* dummy */>
    invalid_value_{};
  size_t capacity_{0};
  size_t size_{
    0};  // caching as cuco_store_->size() is expensive (this scans the entire slots to handle
         // user inserts through a device reference (and currently this is an upper bound (this
         // will become exact once we fully switch to cuco::static_map and use the
         // static_map class's insert_and_assign function; this function will be added soon)
};

}  // namespace detail

/* a class to store (key, value) pairs, the actual storage can either be implemented based on binary
 * tree (when use_binary_search == true) or hash-table (cuCollection, when use_binary_search =
 * false) */
template <typename key_t, typename value_t, bool use_binary_search = true>
class kv_store_t {
 public:
  using key_type   = key_t;
  using value_type = value_t;

  static_assert(std::is_arithmetic_v<key_t>);
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

  kv_store_t(rmm::cuda_stream_view stream) : store_(stream) {}

  /* when use_binary_search = false */
  template <bool binary_search = use_binary_search>
  kv_store_t(
    size_t capacity /* one can expect good performance till the capacity, the actual underlying
                       capacity can be larger (for performance & correctness reasons) */
    ,
    key_t invalid_key /* invalid key shouldn't appear in any *iter in [key_first, key_last) */,
    value_t invalid_value /* invalid_value shouldn't appear in any *iter in [value_first,
                             value_first + cuda::std::distance(key_first, key_last)), invalid_value
                             is returned when match fails for the given key */
    ,
    rmm::cuda_stream_view stream,
    std::enable_if_t<!binary_search, int32_t> = 0)
    : store_(capacity, invalid_key, invalid_value, stream)
  {
  }

  /* when use_binary_search = true */
  template <typename KeyIterator, typename ValueIterator, bool binary_search = use_binary_search>
  kv_store_t(
    KeyIterator key_first,
    KeyIterator key_last,
    ValueIterator value_first,
    value_t invalid_value /* invalid_value is returned when match fails for the given key */,
    bool key_sorted /* if set to true, assume that the input data is sorted and skip sorting (which
                       is necessary for binary-search) */
    ,
    rmm::cuda_stream_view stream,
    std::enable_if_t<binary_search, int32_t> = 0)
    : store_(key_first, key_last, value_first, invalid_value, key_sorted, stream)
  {
  }

  /* when use_binary_search = false */
  template <typename KeyIterator, typename ValueIterator, bool binary_search = use_binary_search>
  kv_store_t(
    KeyIterator key_first,
    KeyIterator key_last,
    ValueIterator value_first,
    key_t invalid_key /* invalid key shouldn't appear in any *iter in [key_first, key_last) */,
    value_t invalid_value /* invalid_value shouldn't appear in any *iter in [value_first,
                             value_first + cuda::std::distance(key_first, key_last)), invalid_value
                             is returned when match fails for the given key */
    ,
    rmm::cuda_stream_view stream,
    std::enable_if_t<!binary_search, int32_t> = 0)
    : store_(key_first, key_last, value_first, invalid_key, invalid_value, stream)
  {
  }

  /* when use_binary_search = true */
  template <bool binary_search = use_binary_search>
  kv_store_t(
    rmm::device_uvector<key_t>&& keys,
    decltype(allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))&& values,
    value_t invalid_value /* invalid_value is returned when match fails for the given key */,
    bool key_sorted /* if set to true, assume that the input data is sorted and skip sorting (which
                       is necessary for binary-search) */
    ,
    rmm::cuda_stream_view stream,
    std::enable_if_t<binary_search, int32_t> = 0)
    : store_(std::move(keys), std::move(values), invalid_value, key_sorted, stream)
  {
  }

  /* when use binary_search = false, this requires that the capacity is large enough */
  template <typename KeyIterator, typename ValueIterator, bool binary_search = use_binary_search>
  std::enable_if_t<!binary_search, void> insert(KeyIterator key_first,
                                                KeyIterator key_last,
                                                ValueIterator value_first,
                                                rmm::cuda_stream_view stream)
  {
    store_.insert(key_first, key_last, value_first, stream);
  }

  /* when use binary_search = false, this requires that the capacity is large enough */
  template <typename KeyIterator,
            typename ValueIterator,
            typename StencilIterator,
            typename PredOp,
            bool binary_search = use_binary_search>
  std::enable_if_t<!binary_search, void> insert_if(KeyIterator key_first,
                                                   KeyIterator key_last,
                                                   ValueIterator value_first,
                                                   StencilIterator stencil_first,
                                                   PredOp pred_op,
                                                   rmm::cuda_stream_view stream)
  {
    store_.insert_if(key_first, key_last, value_first, stencil_first, pred_op, stream);
  }

  /* when use binary_search = false, this requires that the capacity is large enough */
  template <typename KeyIterator, typename ValueIterator, bool binary_search = use_binary_search>
  std::enable_if_t<!binary_search, void> insert_and_assign(KeyIterator key_first,
                                                           KeyIterator key_last,
                                                           ValueIterator value_first,
                                                           rmm::cuda_stream_view stream)
  {
    store_.insert_and_assign(key_first, key_last, value_first, stream);
  }

  auto retrieve_all(rmm::cuda_stream_view stream) const { return store_.retrieve_all(stream); }

  // kv_store_t becomes empty after release
  auto release(rmm::cuda_stream_view stream) { return store_.release(stream); }

  auto view() const
  {
    if constexpr (use_binary_search) {
      return detail::kv_binary_search_store_view_t(store_.store_key_first(),
                                                   store_.store_key_last(),
                                                   store_.store_value_first(),
                                                   store_.invalid_value());
    } else {
      if constexpr (std::is_arithmetic_v<value_t>) {
        return detail::kv_cuco_store_view_t<key_t, value_t const*>(store_.cuco_store_ptr());
      } else {
        return detail::kv_cuco_store_view_t<key_t, decltype(store_.store_value_first())>(
          store_.cuco_store_ptr(), store_.store_value_first(), store_.invalid_value());
      }
    }
  }

  template <bool binary_search = use_binary_search>
  std::enable_if_t<!binary_search, key_t> invalid_key() const
  {
    return store_.invalid_key();
  }

  value_t invalid_value() const { return store_.invalid_value(); }

  size_t size() const { return store_.size(); }

  size_t capacity() const { return store_.capacity(); }

 private:
  std::conditional_t<use_binary_search,
                     detail::kv_binary_search_store_t<key_t, value_t>,
                     detail::kv_cuco_store_t<key_t, value_t>>
    store_;
};

}  // namespace cugraph
