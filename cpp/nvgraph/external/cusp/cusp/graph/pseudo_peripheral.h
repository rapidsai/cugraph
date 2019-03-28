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

/*! \file pseudo_peripheral.h
 *  \brief Pseduo peripheral vertex of a graph
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

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
          typename MatrixType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                         const MatrixType& G);

template<typename MatrixType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const MatrixType& G);

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                         const MatrixType& G,
                               ArrayType& levels);
/*! \endcond */

/**
 * \brief Compute the pseudo-peripheral vertex of a graph
 *
 * \tparam MatrixType Type of input matrix
 * \tparam ArrayType Type of components array
 *
 * \param G A symmetric matrix that represents the graph
 * \param levels Array containing the level set of all vertices from the
 * computed pseudo-peripheral vertex.
 *
 * \return The computed pseudo-peripheral vertex
 *
 * \par Overview
 * Finds a pseduo-peripheral vertex in a graph. A peripheral vertex
 * is the vertex which achieves the diameter of the graph, i.e. achieves the
 * maximum separation distance.
 *
 * \see http://en.wikipedia.org/wiki/Distance_(graph_theory)
 *
 * \par Example
 *
 * \code
 * #include <cusp/csr_matrix.h>
 * #include <cusp/print.h>
 * #include <cusp/gallery/grid.h>
 *
 * //include pseudo_peripheral header file
 * #include <cusp/graph/pseudo_peripheral.h>
 *
 * #include <iostream>
 *
 * int main()
 * {
 *    // Build a 2D grid on the device
 *    cusp::csr_matrix<int,float,cusp::device_memory> G;
 *    cusp::gallery::grid2d(G, 4, 4);
 *
 *    cusp::array1d<int,cusp::device_memory> levels(G.num_rows);
 *
 *    // Compute pseudo peripheral vertex on the device
 *    int pseudo_vertex = cusp::graph::pseudo_peripheral_vertex(G, levels);
 *
 *    // Print the pseudo-peripheral vertex and the level set
 *    std::cout << "Computed pseudo-peripheral vertex " << pseudo_vertex
 *    << " in the graph." << std::endl;
 *
 *    cusp::print(levels);
 *
 *    return 0;
 * }
 * \endcode
 *
 */
template<typename MatrixType,
         typename ArrayType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const MatrixType& G,
                               ArrayType& levels);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/pseudo_peripheral.inl>

