/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_property.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

#include <variant>

namespace cugraph {

using arithmetic_type_t = std::variant<std::monostate, float, double, int32_t, int64_t, size_t>;

using arithmetic_device_uvector_t    = std::variant<std::monostate,
                                                    rmm::device_uvector<float>,
                                                    rmm::device_uvector<double>,
                                                    rmm::device_uvector<int32_t>,
                                                    rmm::device_uvector<int64_t>,
                                                    rmm::device_uvector<size_t>>;
using arithmetic_device_span_t       = std::variant<std::monostate,
                                                    raft::device_span<float>,
                                                    raft::device_span<double>,
                                                    raft::device_span<int32_t>,
                                                    raft::device_span<int64_t>,
                                                    raft::device_span<size_t>>;
using const_arithmetic_device_span_t = std::variant<std::monostate,
                                                    raft::device_span<float const>,
                                                    raft::device_span<double const>,
                                                    raft::device_span<int32_t const>,
                                                    raft::device_span<int64_t const>,
                                                    raft::device_span<size_t const>>;

template <typename edge_t>
using edge_arithmetic_property_t = std::variant<std::monostate,
                                                cugraph::edge_property_t<edge_t, float>,
                                                cugraph::edge_property_t<edge_t, double>,
                                                cugraph::edge_property_t<edge_t, int32_t>,
                                                cugraph::edge_property_t<edge_t, int64_t>,
                                                cugraph::edge_property_t<edge_t, size_t>>;

template <typename edge_t>
using edge_arithmetic_property_view_t =
  std::variant<std::monostate,
               cugraph::edge_property_view_t<edge_t, float const*>,
               cugraph::edge_property_view_t<edge_t, double const*>,
               cugraph::edge_property_view_t<edge_t, int32_t const*>,
               cugraph::edge_property_view_t<edge_t, int64_t const*>,
               cugraph::edge_property_view_t<edge_t, size_t const*>>;

template <typename edge_t>
using edge_arithmetic_property_mutable_view_t =
  std::variant<std::monostate,
               cugraph::edge_property_view_t<edge_t, float*>,
               cugraph::edge_property_view_t<edge_t, double*>,
               cugraph::edge_property_view_t<edge_t, int32_t*>,
               cugraph::edge_property_view_t<edge_t, int64_t*>,
               cugraph::edge_property_view_t<edge_t, size_t*>>;

template <typename dispatched_type_t, typename func_t>
auto variant_type_dispatch(dispatched_type_t& property, func_t func)
{
  switch (property.index()) {
    case 1: return func(std::get<1>(property));
    case 2: return func(std::get<2>(property));
    case 3: return func(std::get<3>(property));
    case 4: return func(std::get<4>(property));
    case 5: return func(std::get<5>(property));
    default: CUGRAPH_FAIL("Variant not initialized");
  }
}

struct sizeof_arithmetic_element {
  template <typename T>
  size_t operator()(rmm::device_uvector<T> const&) const
  {
    return sizeof(T);
  }
  template <typename T>
  size_t operator()(raft::device_span<T> const&) const
  {
    return sizeof(T);
  }
  template <typename T>
  size_t operator()(raft::device_span<T const> const&) const
  {
    return sizeof(T);
  }
};

inline arithmetic_device_span_t make_arithmetic_device_span(arithmetic_device_uvector_t& v)
{
  return variant_type_dispatch(v, [](auto& v) {
    using T = typename std::remove_reference<decltype(v)>::type::value_type;
    return arithmetic_device_span_t(raft::device_span<T>(v.data(), v.size()));
  });
}

inline std::vector<arithmetic_device_span_t> make_arithmetic_device_span_vector(
  std::vector<arithmetic_device_uvector_t>& v)
{
  std::vector<arithmetic_device_span_t> results(v.size());
  std::transform(
    v.begin(), v.end(), results.begin(), [](auto& c) { return make_arithmetic_device_span(c); });
  return results;
}

inline const_arithmetic_device_span_t make_const_arithmetic_device_span(
  arithmetic_device_uvector_t& v)
{
  return variant_type_dispatch(v, [](auto& v) {
    using T = typename std::remove_reference<decltype(v)>::type::value_type;
    return const_arithmetic_device_span_t(raft::device_span<T const>(v.data(), v.size()));
  });
}

inline std::vector<const_arithmetic_device_span_t> make_const_arithmetic_device_span_vector(
  std::vector<arithmetic_device_uvector_t>& v)
{
  std::vector<const_arithmetic_device_span_t> results(v.size());
  std::transform(v.begin(), v.end(), results.begin(), [](auto& c) {
    return make_const_arithmetic_device_span(c);
  });
  return results;
}

}  // namespace cugraph
