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

/*! \file matrix_market.h
 *  \brief MatrixMarket file I/O
 */

#pragma once

#include <cusp/detail/config.h>

#include <string>

namespace cusp
{
namespace io
{

/**
 *  \addtogroup io Input/Output
 *  \brief Provides routines for read/write different file types
 *  \ingroup utilities
 *  \{
 */

/**
 * \brief Read a MatrixMarket file
 *
 * \tparam Matrix matrix container
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param filename file name of the MatrixMarket file
 *
 * \par Overview
 * \note any contents of \p mtx will be overwritten
 *
 * \par Example
 * \code
 * #include <cusp/io/matrix_market.h>
 * #include <cusp/coo_matrix.h>
 *
 * int main(void)
 * {
 *     // read matrix stored in A.mtx into a coo_matrix
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *     cusp::io::read_matrix_market_file(A, "A.mtx");
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see \p write_matrix_market_file
 * \see \p write_matrix_market_stream
 */
template <typename Matrix>
void read_matrix_market_file(Matrix& mtx, const std::string& filename);

/**
 * \brief Read MatrixMarket data from a stream.
 *
 * \tparam Matrix matrix container
 * \tparam Stream stream type
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param input stream from which to read the MatrixMarket contents
 *
 * \par Overview
 * \note any contents of \p mtx will be overwritten
 *
 * \par Example
 * \code
 * #include <cusp/io/matrix_market.h>
 * #include <cusp/coo_matrix.h>
 *
 * int main(void)
 * {
 *     // read matrix stored in A.mtx into a coo_matrix
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *     cusp::io::read_matrix_market_stream(A, std::cin);
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see \p write_matrix_market_file
 * \see \p write_matrix_market_stream
 */
template <typename Matrix, typename Stream>
void read_matrix_market_stream(Matrix& mtx, Stream& input);


/**
 * \brief Write a MatrixMarket file
 *
 * \tparam Matrix matrix container
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param filename file name of the MatrixMarket file
 *
 * \par Overview
 * \note if the file already exists it will be overwritten
 *
 * \par Example
 * \code
 * #include <cusp/io/matrix_market.h>
 * #include <cusp/array2d.h>
 *
 * int main(void)
 * {
 *     // create a simple example
 *     cusp::array2d<float, cusp::host_memory> A(3,4);
 *     A(0,0) = 10;  A(0,1) =  0;  A(0,2) = 20;  A(0,3) =  0;
 *     A(1,0) =  0;  A(1,1) = 30;  A(1,2) =  0;  A(1,3) = 40;
 *     A(2,0) = 50;  A(2,1) = 60;  A(2,2) = 70;  A(2,3) = 80;
 *
 *     // save A into MatrixMarket file
 *     cusp::io::write_matrix_market_file(A, "A.mtx");
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see \p read_matrix_market_file
 * \see \p read_matrix_market_stream
 */
template <typename Matrix>
void write_matrix_market_file(const Matrix& mtx, const std::string& filename);

/**
 * \brief Write MatrixMarket data to a stream.
 *
 * \tparam Matrix matrix container
 * \tparam Stream stream type
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param output stream to which the MatrixMarket contents will be written
 *
 * \par Example
 * \code
 * #include <cusp/io/matrix_market.h>
 * #include <cusp/array2d.h>
 *
 * int main(void)
 * {
 *     // create a simple example
 *     cusp::array2d<float, cusp::host_memory> A(3,4);
 *     A(0,0) = 10;  A(0,1) =  0;  A(0,2) = 20;  A(0,3) =  0;
 *     A(1,0) =  0;  A(1,1) = 30;  A(1,2) =  0;  A(1,3) = 40;
 *     A(2,0) = 50;  A(2,1) = 60;  A(2,2) = 70;  A(2,3) = 80;
 *
 *     // save A into MatrixMarket file
 *     cusp::io::write_matrix_market_stream(A, std::cout);
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see read_matrix_market_file
 * \see read_matrix_market_stream
 */
template <typename Matrix, typename Stream>
void write_matrix_market_stream(const Matrix& mtx, Stream& output);

/*! \}
 */

} //end namespace io
} //end namespace cusp

#include <cusp/io/detail/matrix_market.inl>

