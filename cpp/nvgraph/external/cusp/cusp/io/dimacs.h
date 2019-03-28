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

/*! \file dimacs.h
 *  \brief Dimacs file I/O
 */

#pragma once

#include <cusp/detail/config.h>
#include <thrust/tuple.h>

#include <string>

namespace cusp
{
namespace io
{

/*! \addtogroup io Input/Output
 *  \ingroup utilities
 *  \{
 */

/**
 * \brief Read a Dimacs file
 *
 * \tparam Matrix matrix container
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param filename file name of the Dimacs file
 *
 * \par Overview
 * \note any contents of \p mtx will be overwritten
 *
 * \par Example
 * \code
 * #include <cusp/io/dimacs.h>
 * #include <cusp/coo_matrix.h>
 *
 * int main(void)
 * {
 *     // read matrix stored in A.mtx into a coo_matrix
 *     thrust::tuple<int,int> nodes;
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *     nodes = cusp::io::read_dimacs_file(A, "A.dimacs");
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see \p write_dimacs_file
 * \see \p write_dimacs_stream
 */
template <typename Matrix>
thrust::tuple<typename Matrix::index_type, typename Matrix::index_type>
read_dimacs_file(Matrix& mtx, const std::string& filename);

/**
 * \brief Read Dimacs data from a stream.
 *
 * \tparam Matrix matrix container
 * \tparam Stream stream type
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param input stream from which to read the Dimacs contents
 *
 * \par Overview
 * \note any contents of \p mtx will be overwritten
 *
 * \par Example
 * \code
 * #include <cusp/io/dimacs.h>
 * #include <cusp/coo_matrix.h>
 *
 * int main(void)
 * {
 *     // read matrix stored in A.mtx into a coo_matrix
 *     thrust::tuple<int,int> nodes;
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *     nodes = cusp::io::read_dimacs_stream(A, std::cin);
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see \p write_dimacs_file
 * \see \p write_dimacs_stream
 */
template <typename Matrix, typename Stream>
thrust::tuple<typename Matrix::index_type, typename Matrix::index_type>
read_dimacs_stream(Matrix& mtx, Stream& input);


/**
 * \brief Write a Dimacs file
 *
 * \tparam Matrix matrix container
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param t a tuple of indices specifying the source and sink vertices
 * \param filename file name of the Dimacs file
 *
 * \par Overview
 * \note if the file already exists it will be overwritten
 *
 * \par Example
 * \code
 * #include <cusp/io/dimacs.h>
 * #include <cusp/array2d.h>
 *
 * int main(void)
 * {
 *     // create a simple example
 *     cusp::array2d<float, cusp::host_memory> A(4,4);
 *     A(0,0) = 10;  A(0,1) =  0;  A(0,2) = 20;  A(0,3) =  0;
 *     A(1,0) =  0;  A(1,1) = 30;  A(1,2) =  0;  A(1,3) = 40;
 *     A(2,0) = 50;  A(2,1) = 60;  A(2,2) = 70;  A(2,3) = 80;
 *     A(3,0) =  0;  A(3,1) =  0;  A(3,2) =  0;  A(3,3) =  0;
 *
 *     // save A into Dimacs file
 *     thrust::tuple<int,int> nodes(0,3);
 *     cusp::io::write_dimacs_file(A, nodes, "A.dimacs");
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see \p read_dimacs_file
 * \see \p read_dimacs_stream
 */
template <typename Matrix>
void write_dimacs_file(const Matrix& mtx,
                       const thrust::tuple<typename Matrix::index_type,typename Matrix::index_type>& t,
                       const std::string& filename);

/**
 * \brief Write Dimacs data to a stream.
 *
 * \tparam Matrix matrix container
 * \tparam Stream stream type
 *
 * \param mtx a matrix container (e.g. \p csr_matrix or \p coo_matrix)
 * \param t a tuple of indices specifying the source and sink vertices
 * \param output stream to which the Dimacs contents will be written
 *
 * \par Example
 * \code
 * #include <cusp/io/dimacs.h>
 * #include <cusp/array2d.h>
 *
 * int main(void)
 * {
 *     // create a simple example
 *     cusp::array2d<float, cusp::host_memory> A(4,4);
 *     A(0,0) = 10;  A(0,1) =  0;  A(0,2) = 20;  A(0,3) =  0;
 *     A(1,0) =  0;  A(1,1) = 30;  A(1,2) =  0;  A(1,3) = 40;
 *     A(2,0) = 50;  A(2,1) = 60;  A(2,2) = 70;  A(2,3) = 80;
 *     A(3,0) =  0;  A(3,1) =  0;  A(3,2) =  0;  A(3,3) =  0;
 *
 *     // save A into Dimacs file
 *     thrust::tuple<int,int> nodes(0,3);
 *     cusp::io::write_dimacs_stream(A, nodes, std::cout);
 *
 *     return 0;
 * }
 * \endcode
 *
 * \see read_dimacs_file
 * \see read_dimacs_stream
 */
template <typename Matrix, typename Stream>
void write_dimacs_stream(const Matrix& mtx,
                         const thrust::tuple<typename Matrix::index_type,typename Matrix::index_type>& t,
                         Stream& output);

/*! \}
 */

} //end namespace io
} //end namespace cusp

#include <cusp/io/detail/dimacs.inl>

