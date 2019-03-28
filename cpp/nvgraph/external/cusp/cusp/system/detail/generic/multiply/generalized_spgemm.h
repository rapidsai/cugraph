/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <cusp/detail/config.h>
#include <cusp/detail/array2d_format_utils.h>
#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>
#include <cusp/detail/utils.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/functional.h>

#include <thrust/reduce.h>

#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename MatrixType1,
         typename MatrixType2,
         typename ArrayType,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
struct sparse_inner_functor
{
    typedef typename MatrixType1::index_type IndexType;
    typedef typename MatrixType1::value_type ValueType;

    typename MatrixType1::row_indices_array_type::const_iterator      A_row_indices;
    typename MatrixType1::column_indices_array_type::const_iterator   A_column_indices;
    typename MatrixType1::values_array_type::const_iterator           A_values;
    typename MatrixType2::row_indices_array_type::const_iterator      B_row_indices;
    typename MatrixType2::column_indices_array_type::const_iterator   B_column_indices;
    typename MatrixType2::values_array_type::const_iterator           B_values;
    typename ArrayType::const_iterator                                A_row_offsets;
    typename ArrayType::const_iterator                                B_row_offsets;
    typename ArrayType::const_iterator                                permutation;

    UnaryFunction initialize;
    BinaryFunction1 combine;
    BinaryFunction2 reduce;

    sparse_inner_functor(const MatrixType1& A,
                         const MatrixType2& B,
                         const ArrayType& A_row_offsets,
                         const ArrayType& B_row_offsets,
                         const ArrayType& permutation,
                         UnaryFunction   initialize,
                         BinaryFunction1 combine,
                         BinaryFunction2 reduce)
        : A_row_indices(A.row_indices.begin()),
          A_column_indices(A.column_indices.begin()),
          A_values(A.values.begin()),
          B_row_indices(B.row_indices.begin()),
          B_column_indices(B.column_indices.begin()),
          B_values(B.values.begin()),
          A_row_offsets(A_row_offsets.begin()),
          B_row_offsets(B_row_offsets.begin()),
          permutation(permutation.begin()),
          initialize(initialize),
          combine(combine),
          reduce(reduce)
    {}

    template <typename Tuple>
    __host__ __device__
    ValueType operator()(const Tuple& t) const
    {
        IndexType row = thrust::get<0>(t);
        IndexType col = thrust::get<1>(t);
        ValueType sum = initialize(thrust::get<2>(t));

        int A_pos = A_row_offsets[row];
        int A_end = A_row_offsets[row + 1];
        int B_pos = B_row_offsets[col];
        int B_end = B_row_offsets[col + 1];

        //while not finished with either A[row,:] or B[:,col]
        while(A_pos < A_end && B_pos < B_end) {
            IndexType perm = permutation[B_pos];
            IndexType A_j  = A_column_indices[A_pos];
            IndexType B_j  = B_row_indices[perm];

            if(A_j == B_j) {
                sum = reduce(sum, combine(A_values[A_pos], B_values[perm]));
                A_pos++;
                B_pos++;
            } else if (A_j < B_j) {
                A_pos++;
            } else {
                //B_j < A_j
                B_pos++;
            }
        }

        return sum;
    }
};

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void generalized_spgemm(thrust::execution_policy<DerivedPolicy> &exec,
                        const LinearOperator&  A,
                        const MatrixOrVector1& B,
                        MatrixOrVector2& C,
                        UnaryFunction   initialize,
                        BinaryFunction1 combine,
                        BinaryFunction2 reduce,
                        cusp::coo_format,
                        cusp::coo_format,
                        cusp::coo_format)
{
    typedef typename LinearOperator::index_type IndexType;

    typedef cusp::detail::temporary_array<IndexType, DerivedPolicy> ArrayType;
    typedef sparse_inner_functor<LinearOperator, MatrixOrVector1, ArrayType, UnaryFunction, BinaryFunction1, BinaryFunction2> InnerOp;

    if(C.num_entries == 0)
        return;

    ArrayType B_row_offsets(exec, B.num_cols + 1);
    ArrayType permutation(exec, B.num_entries);
    thrust::sequence(exec, permutation.begin(), permutation.end());

    {
        ArrayType indices(exec, B.column_indices);
        thrust::sort_by_key(exec, indices.begin(), indices.end(), permutation.begin());
        cusp::indices_to_offsets(exec, indices, B_row_offsets);
    }

    ArrayType A_row_offsets(exec, A.num_rows + 1);

    cusp::indices_to_offsets(exec, A.row_indices, A_row_offsets);
    InnerOp incomp_op(A, B, A_row_offsets, B_row_offsets, permutation, initialize, combine, reduce);

    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin(), C.values.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin(), C.values.begin())) + C.num_entries,
                      C.values.begin(),
                      incomp_op);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spgemm(thrust::execution_policy<DerivedPolicy>& exec,
                        const MatrixType1& A,
                        const MatrixType2& B,
                        MatrixType3& C,
                        UnaryFunction   initialize,
                        BinaryFunction1 combine,
                        BinaryFunction2 reduce,
                        cusp::sparse_format,
                        cusp::sparse_format,
                        cusp::sparse_format)
{
    // other formats use COO * COO
    typedef typename MatrixType1::const_coo_view_type             CooMatrix1;
    typedef typename MatrixType2::const_coo_view_type             CooMatrix2;
    typedef typename cusp::detail::as_coo_type<MatrixType3>::type CooMatrix3;

    if(C.num_entries == 0)
        return;

    CooMatrix1 A_(A);
    CooMatrix2 B_(B);
    CooMatrix3 C_(C);

    cusp::generalized_spgemm(exec, A_, B_, C_, initialize, combine, reduce);

    cusp::convert(exec, C_, C);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

