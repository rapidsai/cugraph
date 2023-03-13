/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.hpp>

#include <cuco/static_map.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

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

template <typename KeyIterator, typename ValueIterator>
struct binary_search_find_op_t {
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
      return *(store_value_first + thrust::distance(store_key_first, it));
    } else {
      return invalid_value;
    }
  }
};

template <typename KeyIterator>
struct binary_search_contains_op_t {
  using key_type = typename thrust::iterator_traits<KeyIterator>::value_type;

  KeyIterator store_key_first{};
  KeyIterator store_key_last{};

  __device__ bool operator()(key_type key) const
  {
    return thrust::binary_search(thrust::seq, store_key_first, store_key_last, key);
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
      return *(store_value_first + thrust::distance(store_key_first, it));
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
struct kv_cuco_store_device_view_t {
  using key_type                    = typename ViewType::key_type;
  using value_type                  = typename ViewType::value_type;
  using cuco_store_device_view_type = typename ViewType::cuco_store_type::device_view;

  static_assert(!ViewType::binary_search);

  __host__ kv_cuco_store_device_view_t(ViewType view)
    : cuco_store_device_view(view.cuco_store_device_view())
  {
    if constexpr (std::is_arithmetic_v<value_type>) {
      invalid_value = cuco_store_device_view.get_empty_value_sentinel();
    } else {
      store_value_first = view.store_value_first();
      invalid_value     = view.invalid_value();
    }
  }

  __device__ value_type find(key_type key) const
  {
    auto found = cuco_store_device_view.find(key);
    if (found == cuco_store_device_view.end()) {
      return invalid_value;
    } else {
      auto val = found->second.load(cuda::std::memory_order_relaxed);
      if constexpr (std::is_arithmetic_v<value_type>) {
        return val;
      } else {
        return *((*store_value_first) + val);
      }
    }
  }

  cuco_store_device_view_type cuco_store_device_view{};
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
                      binary_search_find_op_t<KeyIterator, ValueIterator>{
                        store_key_first_, store_key_last_, store_value_first_, invalid_value_});
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void contains(QueryKeyIterator key_first,
                QueryKeyIterator key_last,
                ResultValueIterator value_first,
                rmm::cuda_stream_view stream) const
  {
    thrust::transform(rmm::exec_policy(stream),
                      key_first,
                      key_last,
                      value_first,
                      binary_search_contains_op_t<KeyIterator>{store_key_first_, store_key_last_});
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

  using cuco_store_type =
    cuco::static_map<key_t,
                     std::conditional_t<std::is_arithmetic_v<value_type>, value_type, size_t>,
                     cuda::thread_scope_device,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<char>>>;

  // FIXME: const_cast as a temporary workaround for
  // https://github.com/NVIDIA/cuCollections/issues/242 (cuco find() is not a const function)
  template <typename type = value_type>
  kv_cuco_store_view_t(cuco_store_type const* store,
                       std::enable_if_t<std::is_arithmetic_v<type>, int32_t> = 0)
    : cuco_store_(const_cast<cuco_store_type*>(store))
  {
  }

  // FIXME: const_cast as a temporary workaround for
  // https://github.com/NVIDIA/cuCollections/issues/242 (cuco find() is not a const function)
  template <typename type = value_type>
  kv_cuco_store_view_t(cuco_store_type const* store,
                       ValueIterator value_first,
                       type invalid_value,
                       std::enable_if_t<!std::is_arithmetic_v<type>, int32_t> = 0)
    : cuco_store_(const_cast<cuco_store_type*>(store)),
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
    if constexpr (std::is_arithmetic_v<value_type>) {
      cuco_store_->find(key_first,
                        key_last,
                        value_first,
                        cuco::detail::MurmurHash3_32<key_t>{},
                        thrust::equal_to<key_t>{},
                        stream);
    } else {
      rmm::device_uvector<size_t> indices(thrust::distance(key_first, key_last), stream);
      cuco_store_->find(key_first,
                        key_last,
                        indices.begin(),
                        cuco::detail::MurmurHash3_32<key_t>{},
                        thrust::equal_to<key_t>{},
                        stream);
      auto invalid_idx = cuco_store_->get_empty_value_sentinel();
      thrust::transform(rmm::exec_policy(stream),
                        indices.begin(),
                        indices.end(),
                        value_first,
                        [store_value_first = store_value_first_,
                         invalid_value     = invalid_value_,
                         invalid_idx] __device__(auto idx) {
                          if (idx != invalid_idx) {
                            return *(store_value_first + idx);
                          } else {
                            return invalid_value;
                          }
                        });
    }
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void contains(QueryKeyIterator key_first,
                QueryKeyIterator key_last,
                ResultValueIterator value_first,
                rmm::cuda_stream_view stream) const
  {
    cuco_store_->contains(key_first,
                          key_last,
                          value_first,
                          cuco::detail::MurmurHash3_32<key_t>{},
                          thrust::equal_to<key_t>{},
                          stream);
  }

  auto cuco_store_device_view() const { return cuco_store_->get_device_view(); }

  template <typename type = value_type>
  std::enable_if_t<!std::is_arithmetic_v<type>, ValueIterator> store_value_first() const
  {
    return store_value_first_;
  }

  key_t invalid_key() const { return cuco_store_->get_empty_key_sentinel(); }

  value_type invalid_value() const
  {
    if constexpr (std::is_arithmetic_v<value_type>) {
      return cuco_store_->get_empty_value_sentinel();
    } else {
      return invalid_value_;
    }
  }

 private:
  // FIXME: cuco_store should be a const pointer but we can't due to
  // https://github.com/NVIDIA/cuCollections/issues/242 (cuco find() is not a const function)
  cuco_store_type* cuco_store_{};
  std::conditional_t<std::is_arithmetic_v<value_type>, ValueIterator, std::byte /* dummy */>
    store_value_first_{};

  std::conditional_t<std::is_arithmetic_v<value_type>, value_type, std::byte /* dummy */>
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
    : store_keys_(static_cast<size_t>(thrust::distance(key_first, key_last)), stream),
      store_values_(allocate_dataframe_buffer<value_t>(
        static_cast<size_t>(thrust::distance(key_first, key_last)), stream)),
      invalid_value_(invalid_value)
  {
    thrust::copy(rmm::exec_policy(stream), key_first, key_last, store_keys_.begin());
    auto num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
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

  key_t const* store_key_first() const { return store_keys_.cbegin(); }

  key_t const* store_key_last() const { return store_keys_.cend(); }

  auto store_value_first() const { return get_dataframe_buffer_cbegin(store_values_); }

  value_t invalid_value() const { return invalid_value_; }

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

  using cuco_store_type =
    cuco::static_map<key_t,
                     std::conditional_t<std::is_arithmetic_v<value_t>, value_t, size_t>,
                     cuda::thread_scope_device,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<char>>>;

  kv_cuco_store_t(rmm::cuda_stream_view stream) {}

  template <typename KeyIterator, typename ValueIterator>
  kv_cuco_store_t(KeyIterator key_first,
                  KeyIterator key_last,
                  ValueIterator value_first,
                  key_t invalid_key,
                  value_t invalid_value,
                  rmm::cuda_stream_view stream)
  {
    double constexpr load_factor = 0.7;
    auto num_keys                = static_cast<size_t>(thrust::distance(key_first, key_last));
    auto cuco_size               = std::max(
      static_cast<size_t>(static_cast<double>(num_keys) / load_factor),
      static_cast<size_t>(num_keys) + 1);  // cuco::static_map requires at least one empty slot
    auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(
      rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource()), stream);
    if constexpr (std::is_arithmetic_v<value_t>) {
      cuco_store_ =
        std::make_unique<cuco_store_type>(cuco_size,
                                          cuco::sentinel::empty_key<key_t>{invalid_key},
                                          cuco::sentinel::empty_value<value_t>{invalid_value},
                                          stream_adapter,
                                          stream);

      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(key_first, value_first));
      cuco_store_->insert(pair_first,
                          pair_first + num_keys,
                          cuco::detail::MurmurHash3_32<key_t>{},
                          thrust::equal_to<key_t>{},
                          stream);
    } else {
      cuco_store_ = std::make_unique<cuco_store_type>(
        cuco_size,
        cuco::sentinel::empty_key<key_t>{invalid_key},
        cuco::sentinel::empty_value<size_t>{std::numeric_limits<size_t>::max()},
        stream_adapter,
        stream);
      store_values_  = allocate_dataframe_buffer<value_t>(num_keys, stream);
      invalid_value_ = invalid_value;

      auto pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(key_first, thrust::make_counting_iterator(size_t{0})));
      cuco_store_->insert(pair_first,
                          pair_first + num_keys,
                          cuco::detail::MurmurHash3_32<key_t>{},
                          thrust::equal_to<key_t>{},
                          stream);
      thrust::copy(rmm::exec_policy(stream),
                   value_first,
                   value_first + num_keys,
                   get_dataframe_buffer_begin(store_values_));
    }
  }

  cuco_store_type const* cuco_store_ptr() const { return cuco_store_.get(); }

  template <typename type = value_t>
  std::enable_if_t<!std::is_arithmetic_v<type>, const_value_iterator> store_value_first() const
  {
    return get_dataframe_buffer_cbegin(store_values_);
  }

  key_t invalid_key() const { return cuco_store_.get_empty_key_sentinel(); }

  value_t invalid_value() const
  {
    if constexpr (std::is_arithmetic_v<value_t>) {
      return cuco_store_.get_empty_value_sentinel();
    } else {
      return invalid_value_;
    }
  }

 private:
  std::unique_ptr<cuco_store_type> cuco_store_{nullptr};
  std::conditional_t<!std::is_arithmetic_v<value_t>,
                     decltype(allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{})),
                     std::byte /* dummy */>
    store_values_{};

  std::conditional_t<!std::is_arithmetic_v<value_t>, value_t, std::byte /* dummy */>
    invalid_value_{};
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
                             value_first + thrust::distance(key_first, key_last)), invalid_value is
                             returned when match fails for the given key */
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

 private:
  std::conditional_t<use_binary_search,
                     detail::kv_binary_search_store_t<key_t, value_t>,
                     detail::kv_cuco_store_t<key_t, value_t>>
    store_;
};

}  // namespace cugraph
