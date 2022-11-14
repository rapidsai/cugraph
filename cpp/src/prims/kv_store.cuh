/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
    : store_key_first(view.store_key_first),
      store_key_last(view.store_key_last),
      store_value_first(view.store_value_first),
      invalid_value(view.invalid_value)
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
    : cuco_store(view.cuco_store->get_device_view()),
      store_value_first(view.store_value_first ? thrust::make_optional(*(view.store_value_first))
                                               : thrust::nullopt),
      invalid_value(view.cuco_store->get_empty_value_sentinel())
  {
  }

  __device__ value_type find(key_type key) const
  {
    auto found = cuco_store.find(key);
    if (found == cuco_store.end()) {
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

  cuco_store_device_view_type cuco_store{};
  thrust::optional<typename ViewType::value_iterator> store_value_first{thrust::nullopt};
  value_type invalid_value{};
};

template <typename KeyIterator, typename ValueIterator>
struct kv_binary_search_store_view_t {
  using key_type       = typename thrust::iterator_traits<KeyIterator>::value_type;
  using value_type     = typename thrust::iterator_traits<ValueIterator>::value_type;
  using key_iterator   = KeyIterator;
  using value_iterator = ValueIterator;

  static constexpr bool binary_search = true;

  kv_binary_search_store_view_t(KeyIterator key_first,
                                KeyIterator key_last,
                                ValueIterator value_first,
                                value_type invalid_val)
    : store_key_first(key_first),
      store_key_last(key_last),
      store_value_first(value_first),
      invalid_value(invalid_val)
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
                        store_key_first, store_key_last, store_value_first, invalid_value});
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
                      binary_search_contains_op_t<KeyIterator>{store_key_first, store_key_last});
  }

  KeyIterator store_key_first{};
  KeyIterator store_key_last{};
  ValueIterator store_value_first{};
  value_type invalid_value{};
};

template <typename key_t, typename ValueIterator>
struct kv_cuco_store_view_t {
  using key_type       = key_t;
  using value_type     = typename thrust::iterator_traits<ValueIterator>::value_type;
  using value_iterator = ValueIterator;

  static constexpr bool binary_search = false;

  using cuco_store_type =
    cuco::static_map<key_t,
                     std::conditional_t<std::is_arithmetic_v<value_type>, value_type, size_t>,
                     cuda::thread_scope_device,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<char>>>;

  // FIXME: const_cast as a temporary workaround for
  // https://github.com/NVIDIA/cuCollections/issues/242 (cuco find() is not a const function)
  kv_cuco_store_view_t(cuco_store_type const* store, std::optional<ValueIterator> value_first)
    : cuco_store(const_cast<cuco_store_type*>(store)), store_value_first(value_first)
  {
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void find(QueryKeyIterator key_first,
            QueryKeyIterator key_last,
            ResultValueIterator value_first,
            rmm::cuda_stream_view stream) const
  {
    if constexpr (std::is_arithmetic_v<value_type>) {
      cuco_store->find(key_first,
                       key_last,
                       value_first,
                       cuco::detail::MurmurHash3_32<key_t>{},
                       thrust::equal_to<key_t>{},
                       stream);
    } else {
      rmm::device_uvector<size_t> indices(thrust::distance(key_first, key_last), stream);
      cuco_store->find(key_first,
                       key_last,
                       indices.begin(),
                       cuco::detail::MurmurHash3_32<key_t>{},
                       thrust::equal_to<key_t>{},
                       stream);
      thrust::gather(
        rmm::exec_policy(stream), indices.begin(), indices.end(), *store_value_first, value_first);
    }
  }

  template <typename QueryKeyIterator, typename ResultValueIterator>
  void contains(QueryKeyIterator key_first,
                QueryKeyIterator key_last,
                ResultValueIterator value_first,
                rmm::cuda_stream_view stream) const
  {
    cuco_store->contains(key_first,
                         key_last,
                         value_first,
                         cuco::detail::MurmurHash3_32<key_t>{},
                         thrust::equal_to<key_t>{},
                         stream);
  }

  // FIXME: cuco_store should be a const pointer but we can't due to
  // https://github.com/NVIDIA/cuCollections/issues/242 (cuco find() is not a const function)
  cuco_store_type* cuco_store{};
  std::optional<ValueIterator> store_value_first{};
};

template <typename key_t, typename value_t>
struct kv_binary_search_store_t {
  using key_type   = key_t;
  using value_type = value_t;

  kv_binary_search_store_t(rmm::cuda_stream_view stream)
    : keys(0, stream), values(allocate_dataframe_buffer<value_t>(0, stream))
  {
  }

  kv_binary_search_store_t(size_t size, value_t invalid_value, rmm::cuda_stream_view stream)
    : keys(rmm::device_uvector<key_t>(size, stream)),
      values(allocate_dataframe_buffer<value_t>(size, stream))
  {
  }

  rmm::device_uvector<key_t> keys;
  decltype(allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{})) values;
  value_t invalid_value{};
};

template <typename key_t, typename value_t>
struct kv_cuco_store_t {
  using key_type   = key_t;
  using value_type = value_t;

  using cuco_store_type =
    cuco::static_map<key_t,
                     std::conditional_t<std::is_arithmetic_v<value_t>, value_t, size_t>,
                     cuda::thread_scope_device,
                     rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<char>>>;

  kv_cuco_store_t(rmm::cuda_stream_view stream) {}

  kv_cuco_store_t(size_t size,
                  key_t invalid_key,
                  value_t invalid_value,
                  rmm::cuda_stream_view stream)
  {
    double constexpr load_factor = 0.7;
    auto cuco_size =
      std::max(static_cast<size_t>(static_cast<double>(size) / load_factor),
               static_cast<size_t>(size) + 1);  // cuco::static_map requires at least one empty slot
    auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(
      rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource()), stream);
    if constexpr (std::is_arithmetic_v<value_t>) {
      cuco_store =
        std::make_unique<cuco_store_type>(cuco_size,
                                          cuco::sentinel::empty_key<key_t>{invalid_key},
                                          cuco::sentinel::empty_value<value_t>{invalid_value},
                                          stream_adapter,
                                          stream);
    } else {
      cuco_store = std::make_unique<cuco_store_type>(
        cuco_size,
        cuco::sentinel::empty_key<key_t>{invalid_key},
        cuco::sentinel::empty_value<size_t>{std::numeric_limits<size_t>::max()},
        stream_adapter,
        stream);
      values = allocate_dataframe_buffer<value_t>(size, stream);
    }
  }

  std::unique_ptr<cuco_store_type> cuco_store{nullptr};
  std::optional<decltype(allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))> values{
    std::nullopt};  // valid only when std::is_arithmetic_v<value_t> is false
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
    : store_(static_cast<size_t>(thrust::distance(key_first, key_last)), invalid_value, stream)
  {
    thrust::copy(rmm::exec_policy(stream), key_first, key_last, store_.keys.begin());
    auto num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
    thrust::copy(rmm::exec_policy(stream),
                 value_first,
                 value_first + num_keys,
                 get_dataframe_buffer_begin(store_.values));
    if (!key_sorted) {
      thrust::sort_by_key(rmm::exec_policy(stream),
                          store_.keys.begin(),
                          store_.keys.end(),
                          get_dataframe_buffer_begin(store_.values));
    }
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
    : store_(static_cast<size_t>(thrust::distance(key_first, key_last)),
             invalid_key,
             invalid_value,
             stream)
  {
    auto num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
    if constexpr (std::is_arithmetic_v<value_t>) {
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(key_first, value_first));
      store_.cuco_store->insert(pair_first,
                                pair_first + num_keys,
                                cuco::detail::MurmurHash3_32<key_t>{},
                                thrust::equal_to<key_t>{},
                                stream);
    } else {
      auto pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(key_first, thrust::make_counting_iterator(size_t{0})));
      store_.cuco_store->insert(pair_first,
                                pair_first + num_keys,
                                cuco::detail::MurmurHash3_32<key_t>{},
                                thrust::equal_to<key_t>{},
                                stream);
      thrust::copy(rmm::exec_policy(stream),
                   value_first,
                   value_first + num_keys,
                   get_dataframe_buffer_begin(*(store_.values)));
    }
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
    : store_(stream)
  {
    store_.keys          = std::move(keys);
    store_.values        = std::move(values);
    store_.invalid_value = invalid_value;
    if (!key_sorted) {
      thrust::sort_by_key(rmm::exec_policy(stream),
                          store_.keys.begin(),
                          store_.keys.end(),
                          get_dataframe_buffer_begin(store_.values));
    }
  }

  auto view() const
  {
    if constexpr (use_binary_search) {
      return detail::kv_binary_search_store_view_t(store_.keys.begin(),
                                                   store_.keys.end(),
                                                   get_dataframe_buffer_begin(store_.values),
                                                   store_.invalid_value);
    } else {
      return detail::kv_cuco_store_view_t(
        store_.cuco_store.get(),
        store_.values ? std::make_optional(get_dataframe_buffer_begin(*(store_.values)))
                      : std::nullopt);
    }
  }

 private:
  std::conditional_t<use_binary_search,
                     detail::kv_binary_search_store_t<key_t, value_t>,
                     detail::kv_cuco_store_t<key_t, value_t>>
    store_;
};

}  // namespace cugraph
