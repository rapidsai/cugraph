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

/*! \file poisson.h
 *  \brief Poisson matrix generators
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/gallery/stencil.h>

namespace cusp
{
namespace gallery
{
/**
 *  \addtogroup gallery Gallery
 *  \ingroup utilities
 *  \{
 */

/*! \p poisson5pt: Create a matrix representing a 5pt Poisson problem
 * discretized on an \p m by \p n grid with the standard 2D 5-point
 * finite-difference stencil.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/poisson.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *
 *     // create a matrix for a Poisson problem on a 4x4 grid
 *     cusp::gallery::poisson5pt(A, 4, 4);
 *
 *     // print matrix
 *     cusp::print(A);
 *
 *     return 0;
 * }
 * \endcode
 */
template <typename MatrixType>
void poisson5pt(MatrixType& matrix,
                const size_t m,
                const size_t n);

/*! \p poisson9pt: Create a matrix representing a 9pt Poisson problem
 * discretized on an \p m by \p n grid with the standard 2D 9-point
 * finite-difference stencil.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/poisson.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *
 *     // create a matrix for a Poisson problem on a 4x4 grid
 *     cusp::gallery::poisson9pt(A, 4, 4);
 *
 *     // print matrix
 *     cusp::print(A);
 *
 *     return 0;
 * }
 * \endcode
 */
template <typename MatrixType>
void poisson9pt(MatrixType& matrix,
                const size_t m,
                const size_t n);

/*! \p poisson7pt: Create a matrix representing a 7pt Poisson problem
 * discretized on an \p m by \p n by \p k grid with the standard 3D 7-point
 * finite-difference stencil.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \param k number of grid layers
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/poisson.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *
 *     // create a matrix for a Poisson problem on a 4x4x4 grid
 *     cusp::gallery::poisson7pt(A, 4, 4, 4);
 *
 *     // print matrix
 *     cusp::print(A);
 *
 *     return 0;
 * }
 * \endcode
 */
template <typename MatrixType>
void poisson7pt(MatrixType& matrix,
                const size_t m,
                const size_t n,
                const size_t k);

/*! \p poisson27pt: Create a matrix representing a 27pt Poisson problem
 * discretized on an \p m by \p n by \p l grid with the standard 3D 27-point
 * finite-difference stencil.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \param l number of grid layers
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/poisson.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *
 *     // create a matrix for a Poisson problem on a 4x4x4 grid
 *     cusp::gallery::poisson27pt(A, 4, 4, 4);
 *
 *     // print matrix
 *     cusp::print(A);
 *
 *     return 0;
 * }
 * \endcode
 */
template <typename MatrixType>
void poisson27pt(MatrixType& matrix,
                 const size_t m,
                 const size_t n,
                 const size_t l);

/*! \}
 */

} // end namespace gallery
} // end namespace cusp

#include <cusp/gallery/detail/poisson.inl>
