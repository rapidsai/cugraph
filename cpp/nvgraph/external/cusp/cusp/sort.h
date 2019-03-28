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

/*! \file sort.h
 *  \brief Specialized sorting routines
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  \{
 */

/* \cond */
template <typename DerivedPolicy,
          typename ArrayType>
void counting_sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   ArrayType& keys,
                   typename ArrayType::value_type min,
                   typename ArrayType::value_type max);
/* \endcond */

/**
 * \brief Use counting sort to order an array
 *
 * \tparam ArrayType Type of input array
 *
 * \param keys input of keys to sort
 * \param min minimum key in input array
 * \param max maximum key in input array
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p
 *  counting_sort.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/print.h>
 *  #include <cusp/sort.h>
 *
 *  #include <thrust/transform.h>
 *
 *  struct mod
 *  {
 *      int operator()(int i)
 *      {
 *          // transform i to range [0,5)
 *          return (i >= 0 ? i : -i) % 5;
 *      }
 *  };
 *
 *  int main(void)
 *  {
 *      // create random array with 10 elements
 *      cusp::random_array<int> rand(10);
 *      // initialize v to random array
 *      cusp::array1d<int,cusp::host_memory> v(rand);
 *      // transform entries to interval [0,5)
 *      thrust::transform(v.begin(), v.end(), v.begin(), mod());
 *      // print array
 *      cusp::print(v);
 *      // sort
 *      cusp::counting_sort(v, 0, 5);
 *      // print the sorted array
 *      cusp::print(v);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename ArrayType>
void counting_sort(ArrayType& keys,
                   typename ArrayType::value_type min,
                   typename ArrayType::value_type max);

/* \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
void counting_sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          ArrayType1& keys,
                          ArrayType2& vals,
                          typename ArrayType1::value_type min,
                          typename ArrayType1::value_type max);
/* \endcond */

/**
 * \brief Use counting sort to order an array
 * and permute an array of values
 *
 * \tparam ArrayType1 Type of keys array
 * \tparam ArrayType2 Type of values array
 *
 * \param keys input array of keys to sort
 * \param vals input array of values to permute
 * \param min minimum key in keys array
 * \param max maximum key in keys array
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p
 *  counting_sort_by_key.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/print.h>
 *  #include <cusp/sort.h>
 *
 *  #include <thrust/transform.h>
 *
 *  struct mod
 *  {
 *      int operator()(int i)
 *      {
 *          // transform i to range [0,5)
 *          return (i >= 0 ? i : -i) % 5;
 *      }
 *  };
 *
 *  int main(void)
 *  {
 *      // create random array with 10 elements
 *      cusp::random_array<int> rand1(10,0);
 *      cusp::random_array<int> rand2(10,3);
 *      // initialize v to random array
 *      cusp::array1d<int,cusp::host_memory> v1(rand1);
 *      cusp::array1d<int,cusp::host_memory> v2(rand2);
 *      // transform entries to interval [0,5)
 *      thrust::transform(v1.begin(), v1.end(), v1.begin(), mod());
 *      // print array
 *      cusp::print(v1);
 *      cusp::print(v2);
 *      // sort
 *      cusp::counting_sort_by_key(v1, v2, 0, 5);
 *      // print the sorted array
 *      cusp::print(v1);
 *      cusp::print(v2);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename ArrayType1,
          typename ArrayType2>
void counting_sort_by_key(ArrayType1& keys,
                          ArrayType2& vals,
                          typename ArrayType1::value_type min,
                          typename ArrayType1::value_type max);


/* \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ArrayType1& row_indices,
                 ArrayType2& column_indices,
                 ArrayType3& values,
                 typename ArrayType1::value_type min_row = 0,
                 typename ArrayType1::value_type max_row = 0);
/* \endcond */

/**
 * \brief Sort matrix indices by row
 *
 * \tparam ArrayType1 Type of input matrix row indices
 * \tparam ArrayType2 Type of input matrix column indices
 * \tparam ArrayType3 Type of input matrix values
 *
 * \param row_indices input matrix row indices
 * \param column_indices input matrix column indices
 * \param values input matrix values
 * \param min_row minimum row index
 * \param max_row maximum row index
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p sort_by_row.
 *
 *  \code
 *  #include <cusp/coo_matrix.h>
 *  #include <cusp/print.h>
 *  #include <cusp/sort.h>
 *
 *  int main(void)
 *  {
 *      // allocate storage for (4,3) matrix with 6 nonzeros
 *      cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *      // initialize matrix entries on host
 *      A.row_indices[0] = 3; A.column_indices[0] = 0; A.values[0] = 10;
 *      A.row_indices[1] = 3; A.column_indices[1] = 2; A.values[1] = 20;
 *      A.row_indices[2] = 2; A.column_indices[2] = 0; A.values[2] = 30;
 *      A.row_indices[3] = 2; A.column_indices[3] = 2; A.values[3] = 40;
 *      A.row_indices[4] = 0; A.column_indices[4] = 1; A.values[4] = 50;
 *      A.row_indices[5] = 0; A.column_indices[5] = 2; A.values[5] = 60;
 *
 *      // sort A by row
 *      cusp::sort_by_row(A.row_indices, A.column_indices, A.values);
 *
 *      // print the sorted matrix
 *      cusp::print(A);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row(ArrayType1& row_indices,
                 ArrayType2& column_indices,
                 ArrayType3& values,
                 typename ArrayType1::value_type min_row = 0,
                 typename ArrayType1::value_type max_row = 0);

/* \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row_and_column(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ArrayType1& row_indices,
                            ArrayType2& column_indices,
                            ArrayType3& values,
                            typename ArrayType1::value_type min_row = 0,
                            typename ArrayType1::value_type max_row = 0,
                            typename ArrayType2::value_type min_col = 0,
                            typename ArrayType2::value_type max_col = 0);
/* \endcond */

/**
 * \brief Sort matrix indices by row and column
 *
 * \tparam ArrayType1 Type of input matrix row indices
 * \tparam ArrayType2 Type of input matrix column indices
 * \tparam ArrayType3 Type of input matrix values
 *
 * \param row_indices input matrix row indices
 * \param column_indices input matrix column indices
 * \param values input matrix values
 * \param min_row minimum row index
 * \param max_row maximum row index
 * \param min_col minimum column index
 * \param max_col maximum column index
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p
 *  sort_by_row_and_column.
 *
 *  \code
 *  #include <cusp/coo_matrix.h>
 *  #include <cusp/print.h>
 *  #include <cusp/sort.h>
 *
 *  int main(void)
 *  {
 *      // allocate storage for (4,3) matrix with 6 nonzeros
 *      cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *      // initialize matrix entries on host
 *      A.row_indices[0] = 3; A.column_indices[0] = 2; A.values[0] = 10;
 *      A.row_indices[1] = 3; A.column_indices[1] = 0; A.values[1] = 20;
 *      A.row_indices[2] = 2; A.column_indices[2] = 0; A.values[2] = 30;
 *      A.row_indices[3] = 2; A.column_indices[3] = 2; A.values[3] = 40;
 *      A.row_indices[4] = 0; A.column_indices[4] = 2; A.values[4] = 50;
 *      A.row_indices[5] = 0; A.column_indices[5] = 1; A.values[5] = 60;
 *
 *      // sort A by row
 *      cusp::sort_by_row_and_column(A.row_indices, A.column_indices, A.values);
 *
 *      // print the sorted matrix
 *      cusp::print(A);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row_and_column(ArrayType1& row_indices,
                            ArrayType2& column_indices,
                            ArrayType3& values,
                            typename ArrayType1::value_type min_row = 0,
                            typename ArrayType1::value_type max_row = 0,
                            typename ArrayType2::value_type min_col = 0,
                            typename ArrayType2::value_type max_col = 0);
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/sort.inl>

