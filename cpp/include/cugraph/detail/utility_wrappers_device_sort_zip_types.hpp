/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <thrust/iterator/zip_iterator.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

/** Thrust zip_iterator types that match @c utility_wrappers_zip_device_sort_inst.cu explicit
 *  @c device_sort instantiations (raw column pointers / @c rmm::device_uvector::iterator). */
using zip_i32_i32 = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                       static_cast<std::int32_t*>(nullptr)));
using zip_i64_i64 = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                       static_cast<std::int64_t*>(nullptr)));
using zip_i64_i32 = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                       static_cast<std::int32_t*>(nullptr)));
using zip_i32_i64 = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                       static_cast<std::int64_t*>(nullptr)));
using zip_sz_i32  = decltype(thrust::make_zip_iterator(static_cast<std::size_t*>(nullptr),
                                                      static_cast<std::int32_t*>(nullptr)));
using zip_sz_i64  = decltype(thrust::make_zip_iterator(static_cast<std::size_t*>(nullptr),
                                                      static_cast<std::int64_t*>(nullptr)));
using zip_f_sz    = decltype(thrust::make_zip_iterator(static_cast<float*>(nullptr),
                                                    static_cast<std::size_t*>(nullptr)));
using zip_d_sz    = decltype(thrust::make_zip_iterator(static_cast<double*>(nullptr),
                                                    static_cast<std::size_t*>(nullptr)));

using zip_i32_i32_f   = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                         static_cast<std::int32_t*>(nullptr),
                                                         static_cast<float*>(nullptr)));
using zip_i32_i32_d   = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                         static_cast<std::int32_t*>(nullptr),
                                                         static_cast<double*>(nullptr)));
using zip_i64_i64_f   = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                         static_cast<std::int64_t*>(nullptr),
                                                         static_cast<float*>(nullptr)));
using zip_i64_i64_d   = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                         static_cast<std::int64_t*>(nullptr),
                                                         static_cast<double*>(nullptr)));
using zip_sz_i32_i32  = decltype(thrust::make_zip_iterator(static_cast<std::size_t*>(nullptr),
                                                          static_cast<std::int32_t*>(nullptr),
                                                          static_cast<std::int32_t*>(nullptr)));
using zip_sz_i64_i64  = decltype(thrust::make_zip_iterator(static_cast<std::size_t*>(nullptr),
                                                          static_cast<std::int64_t*>(nullptr),
                                                          static_cast<std::int64_t*>(nullptr)));
using zip_i32_i32_sz  = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                          static_cast<std::int32_t*>(nullptr),
                                                          static_cast<std::size_t*>(nullptr)));
using zip_i64_i64_sz  = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                          static_cast<std::int64_t*>(nullptr),
                                                          static_cast<std::size_t*>(nullptr)));
using zip_i32_i32_i32 = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                           static_cast<std::int32_t*>(nullptr),
                                                           static_cast<std::int32_t*>(nullptr)));
using zip_i32_i64_i32 = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                           static_cast<std::int64_t*>(nullptr),
                                                           static_cast<std::int32_t*>(nullptr)));
using zip_i64_i32_i32 = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                           static_cast<std::int32_t*>(nullptr),
                                                           static_cast<std::int32_t*>(nullptr)));
using zip_i64_i64_i32 = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                           static_cast<std::int64_t*>(nullptr),
                                                           static_cast<std::int32_t*>(nullptr)));

using zip_i32_i32_sz_i   = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                            static_cast<std::int32_t*>(nullptr),
                                                            static_cast<std::size_t*>(nullptr),
                                                            static_cast<int*>(nullptr)));
using zip_i64_i64_sz_i   = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                            static_cast<std::int64_t*>(nullptr),
                                                            static_cast<std::size_t*>(nullptr),
                                                            static_cast<int*>(nullptr)));
using zip_i32_i32_i32_sz = decltype(thrust::make_zip_iterator(static_cast<std::int32_t*>(nullptr),
                                                              static_cast<std::int32_t*>(nullptr),
                                                              static_cast<std::int32_t*>(nullptr),
                                                              static_cast<std::size_t*>(nullptr)));
using zip_i64_i64_i64_sz = decltype(thrust::make_zip_iterator(static_cast<std::int64_t*>(nullptr),
                                                              static_cast<std::int64_t*>(nullptr),
                                                              static_cast<std::int64_t*>(nullptr),
                                                              static_cast<std::size_t*>(nullptr)));

}  // namespace detail
}  // namespace cugraph
