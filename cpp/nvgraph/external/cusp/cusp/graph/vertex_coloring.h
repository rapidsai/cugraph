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

/*! \file vertex_coloring.h
 *  \brief Breadth-first traversal of a graph
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
 *  \brief Algorithms for processing graphs represented in CSR and COO formats
 *  \ingroup algorithms
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
size_t vertex_coloring(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                       const MatrixType& G,
                             ArrayType& colors);
/*! \endcond */

/**
 * \brief Performs a vertex coloring a graph.
 *
 * \tparam MatrixType Type of input matrix
 * \tparam ArrayType Type of colors array
 *
 * \param G A symmetric matrix that represents the graph
 * \param colors Contains to the color associated with each vertex
 * computed during the coloring routine
 *
 *  \see http://en.wikipedia.org/wiki/Graph_coloring
 *
 *  \par Example
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/print.h>
 *  #include <cusp/gallery/grid.h>
 *
 *  //include coloring header file
 *  #include <cusp/graph/vertex_coloring.h>
 *
 *  int main()
 *  {
 *     // Build a 2D grid on the device
 *     cusp::csr_matrix<int,float,cusp::device_memory> G;
 *     cusp::gallery::grid2d(G, 4, 4);
 *
 *     cusp::array1d<int,cusp::device_memory> colors(G.num_rows);
 *
 *     // Execute vertex coloring on the device
 *     cusp::graph::vertex_coloring(G, colors);
 *
 *     // Print the vertex colors
 *     cusp::print(colors);
 *
 *     return 0;
 *  }
 *  \endcode
 */
template<typename MatrixType,
         typename ArrayType>
size_t vertex_coloring(const MatrixType& G,
                             ArrayType& colors);
/*! \}
 */

} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/vertex_coloring.inl>

