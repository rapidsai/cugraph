/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <cuco/detail/hash_functions.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <tuple>
#include <vector>

namespace cugraph {
namespace test {

namespace detail {

template <typename TupleType, typename T, std::size_t... Is>
__host__ __device__ auto make_type_casted_tuple_from_scalar(T val, std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    static_cast<typename thrust::tuple_element<Is, TupleType>::type>(val)...);
}

template <typename property_t, typename T>
__host__ __device__ auto make_property_value(T val)
{
  property_t ret{};
  if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
    ret = make_type_casted_tuple_from_scalar<property_t>(
      val, std::make_index_sequence<thrust::tuple_size<property_t>::value>{});
  } else {
    ret = static_cast<property_t>(val);
  }
  return ret;
}

template <typename vertex_t, typename property_t>
struct property_transform {
  int32_t mod{};

  constexpr __device__ property_t operator()(vertex_t v) const
  {
    static_assert(cugraph::is_thrust_tuple_of_arithmetic<property_t>::value ||
                  std::is_arithmetic_v<property_t>);
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    return make_property_value<property_t>(hash_func(v) % mod);
  }
};

}  // namespace detail

template <typename vertex_t, typename property_t>
struct generate {
 private:
  using property_buffer_type =
    decltype(allocate_dataframe_buffer<property_t>(size_t{0}, rmm::cuda_stream_view{}));

 public:
  static property_t initial_value(int32_t init)
  {
    return detail::make_property_value<property_t>(init);
  }

  static auto vertex_property(raft::handle_t const& handle,
                              rmm::device_uvector<vertex_t> const& labels,
                              int32_t hash_bin_count)
  {
    auto data = cugraph::allocate_dataframe_buffer<property_t>(labels.size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      labels.begin(),
                      labels.end(),
                      cugraph::get_dataframe_buffer_begin(data),
                      detail::property_transform<vertex_t, property_t>{hash_bin_count});
    return data;
  }

  static auto vertex_property(raft::handle_t const& handle,
                              thrust::counting_iterator<vertex_t> begin,
                              thrust::counting_iterator<vertex_t> end,
                              int32_t hash_bin_count)
  {
    auto length = thrust::distance(begin, end);
    auto data   = cugraph::allocate_dataframe_buffer<property_t>(length, handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      begin,
                      end,
                      cugraph::get_dataframe_buffer_begin(data),
                      detail::property_transform<vertex_t, property_t>{hash_bin_count});
    return data;
  }

  template <typename graph_view_type>
  static auto src_property(raft::handle_t const& handle,
                           graph_view_type const& graph_view,
                           property_buffer_type const& property)
  {
    auto output_property =
      cugraph::edge_src_property_t<graph_view_type, property_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, cugraph::get_dataframe_buffer_begin(property), output_property);
    return output_property;
  }

  template <typename graph_view_type>
  static auto dst_property(raft::handle_t const& handle,
                           graph_view_type const& graph_view,
                           property_buffer_type const& property)
  {
    auto output_property =
      cugraph::edge_dst_property_t<graph_view_type, property_t>(handle, graph_view);
    update_edge_dst_property(
      handle, graph_view, cugraph::get_dataframe_buffer_begin(property), output_property);
    return output_property;
  }
};

}  // namespace test
}  // namespace cugraph
