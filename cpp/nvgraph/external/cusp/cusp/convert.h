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

/*! \file convert.h
 *  \brief Matrix format conversion
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  \brief Algorithms for processing matrices in sparse and dense
 *  formats
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void convert(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             const SourceType& src,
                   DestinationType& dst);
/*! \endcond */

/**
 * \brief Convert between matrix formats
 *
 * \tparam SourceType Type of the input matrix to convert
 * \tparam DestinationType Type of the output matrix to create
 *
 * \param src Input matrix to convert
 * \param dst Output matrix created by converting src to the specified format
 *
 * \note DestinationType will be resized as necessary
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp convert header file
 * #include <cusp/convert.h>
 *
 * int main()
 * {
 *   // create an empty sparse matrix structure
 *   cusp::coo_matrix<int,float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an empty array2d structure
 *   cusp::array2d<float,cusp::host_memory> B;
 *
 *   // convert coo_matrix to array2d
 *   cusp::convert(A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 * }
 * \endcode
 *
 * \see \p copy
 */
template <typename SourceType,
          typename DestinationType>
void convert(const SourceType& src,
                   DestinationType& dst);
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/convert.inl>
