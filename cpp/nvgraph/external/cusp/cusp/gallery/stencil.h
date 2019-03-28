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

/*! \file stencil.h
 *  \brief Generate sparse matrix from grid stencil
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>

namespace cusp
{
namespace gallery
{

/**
 *  \addtogroup gallery Gallery
 *  \brief Collection of example matrices
 *  \ingroup utilities
 *  \{
 */

/*! \p generate_matrix_from_stencil: generate a matrix using the provided
 * stencil operator and grid dimensions.
 *
 * \param matrix output
 * \param stencil stencil object
 * \param grid grid dimensions
 * \tparam MatrixType matrix container
 * \tparam StencilPoint stencil descriptor
 * \tparam GridDimension dimensions of the grid
 */
template <typename MatrixType,
          typename StencilPoint,
          typename MemorySpace,
          typename GridDimension>
void generate_matrix_from_stencil(MatrixType& matrix,
                                  const cusp::array1d<StencilPoint,MemorySpace>& stencil,
                                  const GridDimension& grid);
/*! \}
 */

} // end namespace gallery
} // end namespace cusp

#include <cusp/gallery/detail/stencil.inl>

