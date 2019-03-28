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

/*! \file random.h
 *  \brief Random matrix generators
 */

#pragma once

#include <cusp/detail/config.h>

#include <cstddef>

namespace cusp
{
namespace gallery
{
/**
 *  \addtogroup gallery Gallery
 *  \ingroup utilities
 *  \{
 */

// TODO use thrust RNGs, add seed parameter defaulting to num_rows ^ num_cols ^ num_samples
/*! \p random: Create a matrix with random connections
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \param num_samples number of random edges
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/random.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *
 *     // create a matrix for a Poisson problem on a 4x4 grid
 *     cusp::gallery::random(A, 4, 4, 12);
 *
 *     // print matrix
 *     cusp::print(A);
 *
 *     return 0;
 * }
 * \endcode
 */
template <typename MatrixType>
void random(MatrixType& matrix,
            const size_t m,
            const size_t n,
            const size_t num_samples);
/*! \}
 */

} // end namespace gallery
} // end namespace cusp

#include <cusp/gallery/detail/random.inl>
