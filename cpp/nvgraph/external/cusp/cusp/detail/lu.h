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

#pragma once

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/complex.h>
#include <cusp/linear_operator.h>

#include <cmath>

namespace cusp
{
namespace detail
{

template <typename IndexType, typename ValueType, typename MemorySpace, typename Orientation>
int lu_factor(cusp::array2d<ValueType,MemorySpace,Orientation>& A,
              cusp::array1d<IndexType,MemorySpace>& pivot)
{
    typedef typename cusp::norm_type<ValueType>::type NormType;

    const int n = A.num_rows;

    // For each row and column, k = 0, ..., n-1,
    for (int k = 0; k < n; k++)
    {
        // find the pivot row
        pivot[k] = k;
        NormType max = cusp::abs(A(k,k));

        for (int j = k + 1; j < n; j++)
        {
            if (max < cusp::abs(A(j,k)))
            {
                max = cusp::abs(A(j,k));
                pivot[k] = j;
            }
        }

        // and if the pivot row differs from the current row, then
        // interchange the two rows.
        if (pivot[k] != k)
            for (int j = 0; j < n; j++)
                std::swap(A(k,j), A(pivot[k],j));

        // and if the matrix is singular, return error
        if (A(k,k) == ValueType(0))
            return -1;

        // otherwise find the lower triangular matrix elements for column k.
        for (int i = k + 1; i < n; i++)
            A(i,k) /= A(k,k);

        // update remaining matrix
        for (int i = k + 1; i < n; i++)
            for (int j = k + 1; j < n; j++)
                A(i,j) -= A(i,k) * A(k,j);
    }

    return 0;
}



template <typename IndexType, typename ValueType, typename MemorySpace, typename Orientation>
int lu_solve(const cusp::array2d<ValueType,MemorySpace,Orientation>& A,
             const cusp::array1d<IndexType,MemorySpace>& pivot,
             const cusp::array1d<ValueType,MemorySpace>& b,
             cusp::array1d<ValueType,MemorySpace>& x)
{
    const int n = A.num_rows;

    // copy rhs to x
    x = b;

    // Solve the linear equation Lx = b for x, where L is a lower
    // triangular matrix with an implied 1 along the diagonal.
    for (int k = 0; k < n; k++)
    {
        if (pivot[k] != k)
            std::swap(x[k],x[pivot[k]]);

        for (int i = 0; i < k; i++)
            x[k] -= A(k,i) * x[i];
    }

    // Solve the linear equation Ux = y, where y is the solution
    // obtained above of Lx = b and U is an upper triangular matrix.
    for (int k = n - 1; k >= 0; k--)
    {
        for (int i = k + 1; i < n; i++)
            x[k] -= A(k,i) * x[i];

        if (A(k,k) == ValueType(0))
            return -1;

        x[k] /= A(k,k);
    }

    return 0;
}


template <typename ValueType, typename MemorySpace>
class lu_solver : public cusp::linear_operator<ValueType,MemorySpace>
{
private:
    typedef cusp::linear_operator<ValueType,MemorySpace> Parent;

    cusp::array2d<ValueType,cusp::host_memory> lu;
    cusp::array1d<int,cusp::host_memory>       pivot;

public:
    lu_solver()
        : linear_operator<ValueType,MemorySpace>()
    { }

    template <typename ValueType2, typename MemorySpace2>
    lu_solver(const lu_solver<ValueType2,MemorySpace2>& M)
        : lu(M.lu), pivot(M.pivot), Parent(M.num_rows, M.num_cols, M.num_entries)
    { }

    template <typename MatrixType>
    lu_solver(const MatrixType& A)
        : Parent(A.num_rows, A.num_cols, A.num_entries)
    {
        // TODO assert A is square
        lu = A;
        pivot.resize(A.num_rows);
        lu_factor(lu,pivot);
    }

    // TODO handle host and device
    template <typename VectorType1, typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        lu_solve(lu, pivot, x, y);
    }
};

} // end namespace detail
} // end namespace cusp

