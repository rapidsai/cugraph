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

/*! \file connected_components.h
 *  \brief Compute the connected components of a graph
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

#include <cstddef>

namespace cusp
{
namespace graph
{
/*! \addtogroup algorithms Algorithms
 *  \addtogroup graph_algorithms Graph Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
size_t connected_components(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                            const MatrixType& G,
                                  ArrayType& components);
/*! \endcond */

/**
 * \brief Computes the connected components of a graph
 *
 * \tparam MatrixType Type of input matrix
 * \tparam ArrayType Type of components array
 *
 * \param G A symmetric matrix that represents the graph
 * \param components Array containing the number indicating the component
 * that each vertex belongs.
 * \return The number of components found in the graph
 *
 * \see http://en.wikipedia.org/wiki/Connected_component_(graph_theory)
 *
 * \par Example
 *
 * \code
 * #include <cusp/csr_matrix.h>
 * #include <cusp/print.h>
 * #include <cusp/gallery/grid.h>
 *
 * //include connected components header file
 * #include <cusp/graph/connected_components.h>
 *
 * #include <thrust/fill.h>
 * #include <thrust/replace.h>
 *
 * #include <iostream>
 *
 * int main()
 * {
 *    // Build a 2D grid on the device
 *    cusp::csr_matrix<int,float,cusp::device_memory> G;
 *    cusp::gallery::grid2d(G, 4, 4);
 *
 *    // X is used to fill invalid edges in the graph
 *    const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *    // Disconnect vertex 0
 *    thrust::fill(G.column_indices.begin() + G.row_offsets[0],
 *                 G.column_indices.begin() + G.row_offsets[1],
 *                 X);
 *    thrust::replace(G.column_indices.begin(), G.column_indices.end(), 0, X);
 *
 *    cusp::array1d<int,cusp::device_memory> components(G.num_rows);
 *
 *    // Compute connected components on the device
 *    size_t numparts = cusp::graph::connected_components(G, components);
 *
 *    // Print the number of components and the per vertex membership
 *    std::cout << "Found " << numparts << " components in the graph." <<
 *    std::endl;
 *    cusp::print(components);
 *
 *    return 0;
 * }
 * \endcode
 */
template<typename MatrixType,
         typename ArrayType>
size_t connected_components(const MatrixType& G,
                                  ArrayType& components);
/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/connected_components.inl>

