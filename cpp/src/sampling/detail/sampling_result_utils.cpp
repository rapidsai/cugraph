/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/sampling_result_utils.hpp"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/export.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cugraph {
namespace detail {

CUGRAPH_EXPORT arithmetic_device_uvector_t
concatenate_spans(raft::handle_t const& handle,
                  std::vector<arithmetic_device_span_t> const& spans,
                  size_t result_size)
{
  arithmetic_device_uvector_t result{std::monostate{}};
  if (spans.empty()) { return result; }

  // Dispatch on the first span to recover the concrete element type, then concatenate every span
  // (each holding that same alternative) into one contiguous device vector.  The lambda returns the
  // common arithmetic_device_uvector_t variant so variant_type_dispatch deduces a single type.
  result = variant_type_dispatch(
    spans[0], [&handle, result_size, &spans](auto& first_span) -> arithmetic_device_uvector_t {
      using T = typename std::decay_t<decltype(first_span)>::value_type;
      rmm::device_uvector<T> out(result_size, handle.get_stream());
      size_t output_offset{0};
      for (auto const& span_variant : spans) {
        auto const& span = std::get<raft::device_span<T>>(span_variant);
        raft::copy(out.begin() + output_offset, span.begin(), span.size(), handle.get_stream());
        output_offset += span.size();
      }
      return arithmetic_device_uvector_t{std::move(out)};
    });

  return result;
}

}  // namespace detail
}  // namespace cugraph
