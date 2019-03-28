/*
 *  Copyright 2008-2014 NVIDIA Corporation
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


/*! \file format.h
 *  \brief Format types
 */

#pragma once

#include <cusp/detail/config.h>

#include <thrust/detail/type_traits.h>

namespace cusp
{

struct known_format   {};
struct unknown_format {};

struct dense_format       : public known_format  {};
struct permutation_format : public known_format  {};
struct array1d_format     : public dense_format  {};
struct array2d_format     : public dense_format  {};

struct sparse_format      : public known_format  {};
struct coo_format         : public sparse_format {};
struct csr_format         : public sparse_format {};
struct dia_format         : public sparse_format {};
struct ell_format         : public sparse_format {};
struct hyb_format         : public sparse_format {};

template<typename is_transpose>
struct orientation {
  typedef is_transpose transpose;
};

template<typename is_transpose>
struct row_major_base : public orientation<is_transpose> {};

template<typename is_transpose>
struct column_major_base : public orientation<is_transpose> {};

typedef row_major_base<thrust::detail::false_type>    row_major;
typedef column_major_base<thrust::detail::false_type> column_major;

} // end namespace cusp

