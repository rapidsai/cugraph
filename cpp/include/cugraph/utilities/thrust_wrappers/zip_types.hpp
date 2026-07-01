/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

/** Thrust zip_iterator types matching @c thrust_wrappers/sort.cu explicit @c sort_impl
 * instantiations.
 */
template <typename... Ts>
using zip_iterator_t = thrust::zip_iterator<cuda::std::tuple<Ts*...>>;

using zip_i32_i32 = zip_iterator_t<std::int32_t, std::int32_t>;
using zip_i64_i64 = zip_iterator_t<std::int64_t, std::int64_t>;
using zip_i64_i32 = zip_iterator_t<std::int64_t, std::int32_t>;
using zip_i32_i64 = zip_iterator_t<std::int32_t, std::int64_t>;
using zip_sz_i32  = zip_iterator_t<std::size_t, std::int32_t>;
using zip_sz_i64  = zip_iterator_t<std::size_t, std::int64_t>;
using zip_f_sz    = zip_iterator_t<float, std::size_t>;
using zip_d_sz    = zip_iterator_t<double, std::size_t>;

using zip_i32_i32_f   = zip_iterator_t<std::int32_t, std::int32_t, float>;
using zip_i32_i32_d   = zip_iterator_t<std::int32_t, std::int32_t, double>;
using zip_i64_i64_f   = zip_iterator_t<std::int64_t, std::int64_t, float>;
using zip_i64_i64_d   = zip_iterator_t<std::int64_t, std::int64_t, double>;
using zip_sz_i32_i32  = zip_iterator_t<std::size_t, std::int32_t, std::int32_t>;
using zip_sz_i64_i64  = zip_iterator_t<std::size_t, std::int64_t, std::int64_t>;
using zip_i32_i32_sz  = zip_iterator_t<std::int32_t, std::int32_t, std::size_t>;
using zip_i64_i64_sz  = zip_iterator_t<std::int64_t, std::int64_t, std::size_t>;
using zip_i32_i32_i32 = zip_iterator_t<std::int32_t, std::int32_t, std::int32_t>;
using zip_i32_i64_i32 = zip_iterator_t<std::int32_t, std::int64_t, std::int32_t>;
using zip_i64_i32_i32 = zip_iterator_t<std::int64_t, std::int32_t, std::int32_t>;
using zip_i64_i64_i32 = zip_iterator_t<std::int64_t, std::int64_t, std::int32_t>;

using zip_i32_i32_sz_i   = zip_iterator_t<std::int32_t, std::int32_t, std::size_t, int>;
using zip_i64_i64_sz_i   = zip_iterator_t<std::int64_t, std::int64_t, std::size_t, int>;
using zip_i32_i32_i32_sz = zip_iterator_t<std::int32_t, std::int32_t, std::int32_t, std::size_t>;
using zip_i64_i64_i64_sz = zip_iterator_t<std::int64_t, std::int64_t, std::int64_t, std::size_t>;

}  // namespace detail
}  // namespace cugraph
