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

/*! \file print.h
 *  \brief Print textual representation of an object
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/**
 *  \addtogroup io Input/Output
 *  \ingroup utilities
 *  \brief Provides load and store operations for sparse matrices
 *  \{
 */

/**
 * \brief print a textual representation of an object
 *
 * \tparam Printable printable type
 *
 * \param p matrix, array, or other printable object
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p print.
 *
 *  \code
 *  #include <cusp/array2d.h>
 *  #include <cusp/print.h>
 *
 *  int main(void)
 *  {
 *      // initialize a 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,3);
 *      A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
 *      A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;
 *
 *      // print A
 *      cusp::print(A);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename Printable>
void print(const Printable& p);

/**
 * \brief Print a textual representation of an object on a given stream
 *
 * \tparam Printable printable type
 * \tparam Stream output stream type
 *
 * \param p matrix, array, or other printable object
 * \param s stream on which to write the output
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p print.
 *
 *  \code
 *  #include <cusp/array2d.h>
 *  #include <cusp/print.h>
 *
 *  #include <sstream>
 *
 *  int main(void)
 *  {
 *      // initialize a 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,3);
 *      A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
 *      A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;
 *
 *      std::ostringstream oss;
 *
 *      // print A to stream
 *      cusp::print(A, oss);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename Printable,
          typename Stream>
void print(const Printable& p,
                 Stream& s);

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/print.inl>

