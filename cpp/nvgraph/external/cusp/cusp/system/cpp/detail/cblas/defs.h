/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file defs.h
 *  \brief CBLAS utility definitions for interface routines
 */

#pragma once

#include <cblas.h>

namespace cusp
{
namespace system
{
namespace cpp
{
namespace detail
{
namespace cblas
{

struct cblas_format {};

struct upper   : public cblas_format {};
struct lower   : public cblas_format {};
struct unit    : public cblas_format {};
struct nonunit : public cblas_format {};

struct cblas_row_major { const static CBLAS_ORDER order = CblasRowMajor; };
struct cblas_col_major { const static CBLAS_ORDER order = CblasColMajor; };

template< typename LayoutFormat >
struct Orientation : thrust::detail::eval_if<
                        thrust::detail::is_same<LayoutFormat, cusp::row_major>::value,
                        thrust::detail::identity_<cblas_row_major>,
                        thrust::detail::identity_<cblas_col_major>
                     >
{};

} // end namespace cblas
} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace cusp

