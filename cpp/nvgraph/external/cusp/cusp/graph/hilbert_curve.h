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

/*! \file hilbert_curve.h
 *  \brief Cluster points using a Hilbert space filling curve
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
          typename Array2dType,
          typename ArrayType>
void hilbert_curve(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                   const Array2dType& coord,
                   const size_t num_parts,
                         ArrayType& parts);
/*! \endcond */

/**
 * \brief Partition a graph using Hilbert curve
 *
 * \param coord Set of points in 2 or 3-D space
 * \param num_parts Number of partitions to construct
 * \param parts Partition assigned to each point
 *
 * \tparam Array2dType Type of input coordinates array
 * \tparam ArrayType Type of output partition indicator array, parts
 *
 * \par Overview
 * Uses a Hilbert space filling curve to partition
 * a set of points in 2 or 3 dimensional space.
 *
 * \see http://en.wikipedia.org/wiki/Hilbert_curve
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/csr_matrix.h>
 * #include <cusp/print.h>
 * #include <cusp/gallery/grid.h>
 *
 * //include Hilbert curve header file
 * #include <cusp/graph/hilbert_curve.h>
 *
 * #include <iostream>
 *
 * int main()
 * {
 *    // Build a 2D grid on the device
 *    cusp::csr_matrix<int,float,cusp::device_memory> G;
 *    cusp::gallery::grid2d(G, 4, 4);
 *
 *    // Array that indicates partition each vertex belongs
 *    cusp::array1d<int,cusp::device_memory> parts(G.num_rows);
 *
 *    // Partition the graph into 2 parts
 *    size_t num_parts = 2;
 *
 *    // Allocate array of coordinates in 2D
 *    cusp::array2d<float,cusp::device_memory> coords(G.num_rows, 2);
 *
 *    // Generate random coordinates
 *    cusp::copy(cusp::random_array<float>(coords.num_entries, rand()), coords.values);
 *
 *    // Compute the hilbert space filling curve partitioning the points
 *    cusp::graph::hilbert_curve(coords, num_parts, parts);
 *
 *    // Print the number of components and the per vertex membership
 *    cusp::print(parts);
 *
 *    return 0;
 * }
 * \endcode
 */
template <class Array2dType,
          class ArrayType>
void hilbert_curve(const Array2dType& coord,
                   const size_t num_parts,
                         ArrayType& parts);
/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/hilbert_curve.inl>

