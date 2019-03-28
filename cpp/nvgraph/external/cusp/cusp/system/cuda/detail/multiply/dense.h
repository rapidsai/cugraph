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


#include <cusp/convert.h>
#include <cusp/multiply.h>

#include <cusp/detail/type_traits.h>

#include <cusp/detail/execution_policy.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              MatrixType& A,
              ArrayType1& x,
              ArrayType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::array2d_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename cusp::detail::as_array2d_type<MatrixType,cusp::host_memory>::type Array2d;
    typedef typename ArrayType1::value_type ValueType1;
    typedef typename ArrayType2::value_type ValueType2;

    Array2d A_(A);
    cusp::array1d<ValueType1,cusp::host_memory> x_(x);
    cusp::array1d<ValueType2,cusp::host_memory> y_(y.size());

    cusp::multiply(A_, x_, y_, initialize, combine, reduce);

    std::cout << "copying to output" << std::endl;
    cusp::copy(y_, y);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              MatrixType1& A,
              MatrixType2& B,
              MatrixType3& C,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::array2d_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    typedef typename cusp::detail::as_array2d_type<MatrixType1,cusp::host_memory>::type Array2dMatrix1;
    typedef typename cusp::detail::as_array2d_type<MatrixType2,cusp::host_memory>::type Array2dMatrix2;
    typedef typename cusp::detail::as_array2d_type<MatrixType3,cusp::host_memory>::type Array2dMatrix3;

    Array2dMatrix1 A_(A);
    Array2dMatrix2 B_(B);
    Array2dMatrix3 C_(A.num_rows, B.num_cols);

    cusp::multiply(A_, B_, C_, initialize, combine, reduce);

    cusp::convert(C_, C);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp
