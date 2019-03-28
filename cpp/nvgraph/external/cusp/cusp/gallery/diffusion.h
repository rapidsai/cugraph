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

/*! \file diffusion.h
 *  \brief Anisotropic diffusion matrix generators
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/gallery/stencil.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES 1  // make sure M_PI is defined
#endif
#include <math.h>

namespace cusp
{
namespace gallery
{

/*! \cond */
struct disc_type {};
struct FD : public disc_type {};
struct FE : public disc_type {};
/*! \endcond */

/**
 *  \addtogroup gallery Gallery
 *  \brief Collection of example matrices
 *  \ingroup utilities
 *  \{
 */

/*! \p diffusion: Create a matrix representing an anisotropic
 * Poisson problem discretized on an \p m by \p n grid with
 * the a given direction.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns
 * \param eps magnitude of anisotropy
 * \param theta angle of anisotropy in radians
 *
 * \tparam MatrixType matrix container
 *
 */
template <typename Method,
          typename MatrixType>
void diffusion(MatrixType& matrix,
               const size_t m,
               const size_t n,
               const double eps = 1e-5,
               const double theta = M_PI/4.0);
/*! \}
 */

} // end namespace gallery
} // end namespace cusp

#include <cusp/gallery/detail/diffusion.inl>
