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

/*! \file verify.h
 *  \brief Validate matrix format
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  \{
 */

/**
 * \brief Validate format of a given matrix
 *
 * \tparam MatrixType matrix container
 *
 * \param matrix A matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \return \p true if format is valid otherwise \p false
 */
template <typename MatrixType>
bool is_valid_matrix(const MatrixType& matrix);

/**
 * \brief Validate format of a given matrix
 *
 * \tparam MatrixType matrix container
 * \tparam OutputStream stream type
 *
 * \param matrix A matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param ostream Stream to which the validate stream should print
 * \return \p true if format is valid otherwise \p false
 */
template <typename MatrixType,
          typename OutputStream>
bool is_valid_matrix(const MatrixType& matrix,
                           OutputStream& ostream);

/**
 * \brief Validate format of a given matrix and exit if invalid
 *
 * \tparam MatrixType matrix container
 *
 * \param matrix A matrix container (e.g. \p csr_matrix or \p coo_matrix)
 */
template <typename MatrixType>
void assert_is_valid_matrix(const MatrixType& matrix);

/**
 * \brief Assert dimensions of two arrays are equal. Throw invalid
 * input exception if dimensions do not match.
 *
 * \tparam Array1 First array container type
 * \tparam Array2 Second array container type
 *
 * \param array1 First array used in comparison
 * \param array2 Second array used in comparison
 */
template <typename Array1,
          typename Array2>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2);

/**
 * \brief Assert dimensions of two arrays are equal. Throw invalid
 * input exception if dimensions do not match.
 *
 * \tparam Array1 First array container type
 * \tparam Array2 Second array container type
 * \tparam Array3 Third array container type
 *
 * \param array1 First array used in comparison
 * \param array2 Second array used in comparison
 * \param array3 Third array used in comparison
 */
template <typename Array1,
          typename Array2,
          typename Array3>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3);

/**
 * \brief Assert dimensions of two arrays are equal. Throw invalid
 * input exception if dimensions do not match.
 *
 * \tparam Array1 First array container type
 * \tparam Array2 Second array container type
 * \tparam Array3 Third array container type
 * \tparam Array4 Fourth array container type
 *
 * \param array1 First array used in comparison
 * \param array2 Second array used in comparison
 * \param array3 Third array used in comparison
 * \param array4 Fourth array used in comparison
 */
template <typename Array1,
          typename Array2,
          typename Array3,
          typename Array4>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3,
                            const Array4& array4);

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/verify.inl>
