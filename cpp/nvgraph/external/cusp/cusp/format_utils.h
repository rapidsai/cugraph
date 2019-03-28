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

/*! \file format_utils.h
 *  \brief Various matrix utility functions
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

#include <cstddef>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  \{
 */

/* \cond */
template <typename DerivedPolicy,
          typename OffsetArray,
          typename IndexArray>
void offsets_to_indices(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const OffsetArray& offsets,
                              IndexArray& indices);
/* \endcond */

/**
 * \brief Expand CSR row offsets to COO row indices
 *
 * \tparam OffsetType Type of input row offsets
 * \tparam IndexArray Type of input row indices
 *
 * \param offsets The input row offsets
 * \param indices The output row indices
 *
 * \par Example
 * \code
 * // include cusp array1d header file
 * #include <cusp/array1d.h>
 * #include <cusp/format_utils.h>
 *
 * int main()
 * {
 *   // Allocate a array of size 3
 *   cusp::array1d<int,cusp::host_memory> a(3);
 *
 *   // Set the elements to [0, 2, 5]
 *   a[0] = 0; a[1] = 2; a[2] = 5;
 *
 *   // Allocate a seceond array equal to the total number
 *   // of expanded entries from vector a (last entry in the vector)
 *   cusp::array1d<int,cusp::host_memory> b(a.back());
 *
 *   // compute the expanded entries
 *   cusp::offsets_to_indices(a, b);
 *
 *   // print [0, 0, 1, 1, 1]
 *   cusp::print(b);
 * }
 * \endcode
 */
template <typename OffsetArray,
          typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets,
                              IndexArray& indices);

/* \cond */
template <typename DerivedPolicy,
          typename IndexArray,
          typename OffsetArray>
void indices_to_offsets(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const IndexArray& indices,
                              OffsetArray& offsets);
/* \endcond */

/**
 * \brief Compress COO row indices to CSR row offsets
 *
 * \tparam IndexArray Type of input row indices
 * \tparam OffsetType Type of input row offsets
 *
 * \param indices The input row indices
 * \param offsets The output row offsets
 *
 * \par Example
 * \code
 * // include cusp array1d header file
 * #include <cusp/array1d.h>
 * #include <cusp/format_utils.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *   // Allocate a array of size 5
 *   cusp::array1d<int,cusp::host_memory> a(5);
 *
 *   // Set the elements to [0, 0, 1, 1, 1]
 *   a[0] = 0; a[1] = 0; a[2] = 1; a[3] = 1; a[4] = 1;
 *
 *   // Allocate a seceond array equal to the last entry in
 *   // a (number of rows) + 2 (zero entry + last entry)
 *   cusp::array1d<int,cusp::host_memory> b(a.back() + 2);
 *
 *   // compute the expanded entries
 *   cusp::indices_to_offsets(a, b);
 *
 *   // print [0, 2, 5]
 *   cusp::print(b);
 * }
 * \endcode
 */
template <typename IndexArray,
          typename OffsetArray>
void indices_to_offsets(const IndexArray& indices,
                              OffsetArray& offsets);

/* \cond */
template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
void extract_diagonal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      const MatrixType& A,
                            ArrayType& output);
/* \endcond */

/**
 * \brief Extract the main diagonal of a matrix
 *
 * \tparam MatrixType Type of input matrix
 * \tparam ArrayType Type of input diagonal array
 *
 * \param A The input matrix
 * \param output On return contains the main diagonal of A with zeros inserted
 * for missing entries.
 *
 * \par Example
 * \code
 * // include cusp array1d header file
 * #include <cusp/array1d.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 * #incldue <cusp/gallery/poisson.h>
 *
 * #include <cusp/format_utils.h>
 *
 * int main()
 * {
 *   // initialize 5x5 poisson matrix
 *   cusp::coo_matrix<int,float,cusp::host_memory> A;
 *   cusp::gallery::poisson5pt(A, 5, 5);
 *
 *   // allocate array to hold diagonal entries
 *   cusp::array1d<float, cusp::host_memory> diagonal(A.num_rows);
 *
 *   // extract diagonal of A
 *   cusp::extract_diagonal(A, diagonal);
 *
 *   // print diagonal entries
 *   cusp::print(diagonal);
 * }
 * \endcode
 */
template <typename MatrixType,
          typename ArrayType>
void extract_diagonal(const MatrixType& A,
                            ArrayType& output);

/* \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
size_t count_diagonals(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices );
/* \endcond */

/**
 * \brief Count the number of occupied diagonals in the input matrix
 *
 * \tparam ArrayType1 Type of input row indices
 * \tparam ArrayType2 Type of input column indices
 *
 * \param num_rows Number of rows.
 * \param num_cols Number of columns.
 * \param row_indices row indices of input matrix
 * \param column_indices column indices of input matrix
 * \return number of occupied diagonals
 *
 * \par Example
 * \code
 * #include <cusp/coo_matrix.h>
 * #incldue <cusp/gallery/poisson.h>
 *
 * #include <cusp/format_utils.h>
 *
 * #include <iostream>
 *
 * int main()
 * {
 *   // initialize 5x5 poisson matrix
 *   cusp::coo_matrix<int,float,cusp::host_memory> A;
 *   cusp::gallery::poisson5pt(A, 5, 5);
 *
 *   // count the number of occupied diagonals of A
 *   std::cout << count_diagonals(A.num_rows,
 *                                A.num_cols,
 *                                A.row_indices,
 *                                A.column_indices) << std::endl;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2>
size_t count_diagonals(const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices);

/* \cond */
template <typename DerivedPolicy,
          typename ArrayType>
size_t compute_max_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   const ArrayType& row_offsets);
/* \endcond */

/**
 * \brief Compute the maximum row length of a matrix
 *
 * \tparam ArrayType Type of input row offsets
 *
 * \param row_offsets row offsets of input matrix
 * \return maximum row length
 *
 * \par Example
 * \code
 * #include <cusp/csr_matrix.h>
 * #incldue <cusp/gallery/poisson.h>
 *
 * #include <cusp/format_utils.h>
 *
 * #include <iostream>
 *
 * int main()
 * {
 *   // initialize 5x5 poisson matrix
 *   cusp::csr_matrix<int,float,cusp::host_memory> A;
 *   cusp::gallery::poisson5pt(A, 5, 5);
 *
 *   // count the number of occupied diagonals of A
 *   std::cout << compute_max_entries_per_row(A.row_offsets) << std::endl;
 * }
 * \endcode
 */
template <typename ArrayType>
size_t compute_max_entries_per_row(const ArrayType& row_offsets);

/* \cond */
template <typename DerivedPolicy,
          typename ArrayType>
size_t compute_optimal_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096);
/* \endcond */

/**
 * \brief Compute the optimal number of entries per row of HYB matrix
 *
 * \tparam ArrayType Type of input row offsets
 *
 * \param row_offsets row offsets of input matrix
 * \param relative_speed Estimated performance difference between ELL and COO
 * row storage schemes
 * \param breakeven_threshold Threshold value separating ELL and COO row
 * classification
 * \return optimal number of columns to store in ELL matrix
 *
 * \par Example
 * \code
 * #include <cusp/csr_matrix.h>
 * #incldue <cusp/gallery/poisson.h>
 *
 * #include <cusp/format_utils.h>
 *
 * #include <iostream>
 *
 * int main()
 * {
 *   // initialize 5x5 poisson matrix
 *   cusp::csr_matrix<int,float,cusp::host_memory> A;
 *   cusp::gallery::poisson5pt(A, 5, 5);
 *
 *   // count the number of occupied diagonals of A
 *   std::cout << compute_optimal_entries_per_row(A.row_offsets) << std::endl;
 * }
 * \endcode
 */
template <typename ArrayType>
size_t compute_optimal_entries_per_row(const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096);
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/format_utils.inl>

