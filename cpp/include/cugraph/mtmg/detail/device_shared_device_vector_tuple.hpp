/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_device_span_tuple.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace mtmg {
namespace detail {

/**
 * @brief  Manage a tuple of device vector on each GPU
 *
 * Uses the device_shared_wrapper to manage a tuple of rmm::device_uvector
 * instances on each GPU.
 */
template <typename... Ts>
class device_shared_device_vector_tuple_t
  : public device_shared_wrapper_t<std::tuple<rmm::device_uvector<Ts>...>> {
  using parent_t = detail::device_shared_wrapper_t<std::tuple<rmm::device_uvector<Ts>...>>;

 public:
  /**
   * @brief Create a device_shared_device_span (read only view)
   */
  auto view()
  {
    std::lock_guard<std::mutex> lock(parent_t::lock_);

    device_shared_device_span_tuple_t<Ts...> result;

    std::for_each(parent_t::objects_.begin(), parent_t::objects_.end(), [&result, this](auto& p) {
      convert_to_span(std::index_sequence_for<Ts...>(), result, p);
      // std::size_t Is... = std::index_sequence_for<Ts...>;
      // result.set(p.first, std::make_tuple(raft::device_span<Ts
      // const>{std::get<Is>(p.second).data(), std::get<Is>(p.second).size()}...));
    });

    return result;
  }

 private:
  template <std::size_t... Is>
  void convert_to_span(std::index_sequence<Is...>,
                       device_shared_device_span_tuple_t<Ts...>& result,
                       std::pair<int32_t const, std::tuple<rmm::device_uvector<Ts>...>>& p)
  {
    result.set(p.first,
               std::make_tuple(raft::device_span<Ts>{std::get<Is>(p.second).data(),
                                                     std::get<Is>(p.second).size()}...));
  }
};

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph
