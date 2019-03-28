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

/*! \file grid.h
 *  \brief Grid matrix generators
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

/*! \p grid2d: Create a matrix representing a 2d \p m by \p n grid.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/grid.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *
 *     // create a matrix for a 4x4 grid
 *     cusp::gallery::grid2d(A, 4, 4);
 *
 *     // print matrix
 *     cusp::print(A);
 *
 *     return 0;
 * }
 * \endcode
 *
 */
template <typename MatrixType>
void grid2d(MatrixType& matrix,
            const size_t m,
            const size_t n);

/*! \p grid3d: Create a matrix representing a 3d \p m by \p n by \p l grid.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \param l number of grid layers
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/grid.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *
 *     // create a matrix for a 4x4x4 grid
 *     cusp::gallery::grid3d(A, 4, 4, 4);
 *
 *     // print matrix
 *     cusp::print(A);
 *
 *     return 0;
 * }
 * \endcode
 *
 */
template <typename MatrixType>
void grid3d(MatrixType& matrix,
            const size_t m,
            const size_t n,
            const size_t l);

/*! \}
 */

} // end namespace gallery
} // end namespace cusp

#include <cusp/gallery/detail/grid.inl>
