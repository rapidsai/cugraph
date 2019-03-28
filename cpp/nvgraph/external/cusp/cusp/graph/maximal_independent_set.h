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

/*! \file maximal_independent_set.h
 *  \brief Maximal independent set of a graph
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
size_t maximal_independent_set(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                               const MatrixType& G,
                                     ArrayType& stencil,
                               const size_t k = 1);
/*! \endcond */

/**
 * \brief Compute maximal independent set of a graph
 *
 * \tparam MatrixType Type of input matrix
 * \tparam ArrayType Type of components array
 *
 * \param G symmetric matrix that represents a graph
 * \param stencil array to hold the MIS(k)
 * \param k radius of independence
 *
 * \return The number of MIS vertices computed for G.
 *
 * \par Overview
 *
 * Computes a maximal independent set (MIS) a graph. The MIS is a set of
 * vertices such that (1) no two vertices
 * are adjacent and (2) it is not possible to add another vertex to thes
 * set without violating the first property.  The MIS(k) is a generalization
 * of the MIS with the property that no two vertices in the set are joined
 * by a path of \p k edges or less.  The standard MIS is therefore a MIS(1).
 *
 * The MIS(k) is represented by an array of {0,1} values.  Specifically,
 * <tt>stencil[i]</tt> is 1 if vertex \p i is a member of the MIS(k) and
 * 0 otherwise.
 *
 *
 * \see http://en.wikipedia.org/wiki/Maximal_independent_set
 *
 * \par Example
 *
 * \code
 * #include <cusp/csr_matrix.h>
 * #include <cusp/print.h>
 * #include <cusp/gallery/grid.h>
 *
 * //include MIS header file
 * #include <cusp/graph/maximal_independent_set.h>
 *
 * #include <iostream>
 *
 * int main()
 * {
 *    // Build a 2D grid on the device
 *    cusp::csr_matrix<int,float,cusp::device_memory> G;
 *    cusp::gallery::grid2d(G, 4, 4);
 *
 *    cusp::array1d<int,cusp::device_memory> stencil(G.num_rows);
 *
 *    // Compute MIS on the device
 *    size_t num_mis = cusp::graph::maximal_independent_set(G, stencil);
 *
 *    // Print the number of MIS vertices and membership stencil
 *    std::cout << "Computed " << num_mis << " MIS(1) vertices in the graph." <<
 *    std::endl;
 *    cusp::print(stencil);
 *
 *    return 0;
 * }
 * \endcode
 */
template <typename MatrixType,
          typename ArrayType>
size_t maximal_independent_set(const MatrixType& G,
                                     ArrayType& stencil,
                               const size_t k = 1);
/*! \}
 */

} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/maximal_independent_set.inl>

