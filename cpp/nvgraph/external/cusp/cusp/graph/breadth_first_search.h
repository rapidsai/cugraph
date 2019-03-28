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

/*! \file breadth_first_search.h
 *  \brief Breadth-first traversal of a graph
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
 *  \brief Algorithms for processing graphs represented in CSR and COO formats
 *  \ingroup algorithms
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
void breadth_first_search(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                          const MatrixType& G,
                          const typename MatrixType::index_type src,
                                ArrayType& labels,
                          const bool mark_levels = true);
/*! \endcond */

/**
 * \brief Performs a Breadth-first traversal of a graph starting from a given source vertex.
 *
 * \tparam MatrixType Type of input matrix
 * \tparam ArrayType Type of labels array
 *
 * \param G A matrix that represents the graph (symmetric or unsymmetric)
 * \param src The source vertex to begin the BFS traversal
 * \param labels If mark_levels is \c false then labels will contain the
 * level set of all the vertices starting from the source vertex otherwise
 * labels will contain the immediate ancestor of each vertex forming a ancestor
 * \param mark_levels Boolean value indicating whether to return level sets, \c false, or
 * predecessor, \c true, markers
 * tree.
 *
 *  \see http://en.wikipedia.org/wiki/Breadth-first_search
 *
 *  \par Example
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/print.h>
 *  #include <cusp/gallery/grid.h>
 *
 *  //include bfs header file
 *  #include <cusp/graph/breadth_first_search.h>
 *
 *  int main()
 *  {
 *     // Build a 2D grid on the device
 *     cusp::csr_matrix<int,float,cusp::device_memory> G;
 *     cusp::gallery::grid2d(G, 4, 4);
 *
 *     cusp::array1d<int,cusp::device_memory> labels(G.num_rows);
 *
 *     // Execute a BFS traversal on the device
 *     cusp::graph::breadth_first_search(G, 0, labels);
 *
 *     // Print the level set constructed from the source vertex
 *     cusp::print(labels);
 *
 *     return 0;
 *  }
 *  \endcode
 */
template<typename MatrixType,
         typename ArrayType>
void breadth_first_search(const MatrixType& G,
                          const typename MatrixType::index_type src,
                                ArrayType& labels,
                          const bool mark_levels = true);
/*! \}
 */

} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/breadth_first_search.inl>

