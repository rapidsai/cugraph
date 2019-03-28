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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/multiply.h>
#include <cusp/graph/maximal_independent_set.h>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void mis_to_aggregates(thrust::execution_policy<DerivedPolicy>& exec,
                       const MatrixType& C,
                       const ArrayType1& mis,
                             ArrayType2& aggregates)
{
    typedef typename MatrixType::index_type                              IndexType;
    typedef cusp::detail::temporary_array<IndexType, DerivedPolicy>      ArrayType;

    typedef typename ArrayType1::const_iterator                          ConstArrayIterator;

    typedef thrust::tuple<IndexType,IndexType>                           Tuple;
    typedef typename thrust::counting_iterator<IndexType>                CountingIterator;

    typedef typename MatrixType::row_indices_array_type::const_view      RowView;
    typedef typename MatrixType::column_indices_array_type::const_view   ColumnView;
    typedef typename cusp::constant_array<Tuple>                         ValueView;
    typedef typename cusp::coo_matrix_view<RowView,ColumnView,ValueView> CooView;

    typedef thrust::tuple<ConstArrayIterator,CountingIterator>           IteratorTuple1;
    typedef typename thrust::zip_iterator<IteratorTuple1>                ZipIterator1;
    typedef typename cusp::array1d_view<ZipIterator1>                    ArrayViewType1;

    typedef typename ArrayType::iterator                                 ArrayIterator;
    typedef thrust::tuple<ArrayIterator,ArrayIterator>                   IteratorTuple2;
    typedef typename thrust::zip_iterator<IteratorTuple2>                ZipIterator2;
    typedef typename cusp::array1d_view<ZipIterator2>                    ArrayViewType2;

    const size_t N = C.num_rows;
    const size_t M = C.num_entries;
    cusp::constant_array<Tuple> values(M, Tuple(1,1));
    CooView A(N, N, M,
              make_array1d_view(C.row_indices),
              make_array1d_view(C.column_indices),
              values);

    // current (ring,index)
    ArrayType mis1(exec, N);
    ArrayType mis2(exec, N);
    ArrayType mis_enum(exec, N);

    ArrayType idx1(exec, N);
    ArrayType idx2(exec, N);

    CountingIterator count_begin(0);
    ZipIterator1 x_iter(thrust::make_tuple(mis.begin(),  count_begin));
    ZipIterator2 y_iter(thrust::make_tuple(mis1.begin(), idx1.begin()));
    ZipIterator2 z_iter(thrust::make_tuple(mis2.begin(), idx2.begin()));

    ArrayViewType1 x(x_iter, x_iter + N);
    ArrayViewType2 y(y_iter, y_iter + N);
    ArrayViewType2 z(z_iter, z_iter + N);

    // (2,i) mis (0,i) non-mis
    // find the largest (mis[j],j) 1-ring neighbor for each node
    cusp::generalized_spmv(exec, A, x, x, y, thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

    // boost mis0 values so they win in second round
    thrust::transform(exec, mis.begin(), mis.end(), mis1.begin(), mis1.begin(), thrust::plus<IndexType>());

    // find the largest (mis[j],j) 2-ring neighbor for each node
    cusp::generalized_spmv(exec, A, y, y, z, thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

    // enumerate the MIS nodes
    thrust::exclusive_scan(exec, mis.begin(), mis.end(), mis_enum.begin());

    thrust::gather(exec, idx2.begin(), idx2.end(), mis_enum.begin(), aggregates.begin());
} // mis_to_aggregates()

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void mis_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                   const MatrixType& C,
                         ArrayType1& aggregates,
                         ArrayType2& mis,
                         cusp::coo_format)
{
    typedef typename MatrixType::index_type                         IndexType;
    typedef cusp::detail::temporary_array<IndexType, DerivedPolicy> ArrayType;

    // compute MIS(2)
    cusp::graph::maximal_independent_set(exec, C, mis, 2);

    // compute aggregates from MIS(2)
    mis_to_aggregates(exec, C, mis, aggregates);

    // locate singletons
    IndexType num_aggregates = *thrust::max_element(exec, aggregates.begin(), aggregates.end()) + 1;
    ArrayType sorted_aggregates(exec, aggregates);
    ArrayType aggregate_counts(exec, num_aggregates);
    ArrayType reduced_aggregates(exec, num_aggregates);

    // compute sizes of the aggregates
    thrust::sort(exec, sorted_aggregates.begin(), sorted_aggregates.end());
    thrust::reduce_by_key(exec,
                          sorted_aggregates.begin(), sorted_aggregates.end(),
                          thrust::constant_iterator<IndexType>(1),
                          thrust::make_discard_iterator(),
                          aggregate_counts.begin());

    // count the number of aggregates consisting of a single node
    IndexType num_singletons = thrust::count(exec, aggregate_counts.begin(), aggregate_counts.end(), IndexType(1));

    // mark singletons with -1 for filtering, the total number of aggregates is now (num_aggregates - num_singletons)
    if ( num_singletons > 0 ) {
        ArrayType aggregate_ids(exec, num_aggregates);
        cusp::detail::temporary_array<bool, DerivedPolicy> isone(exec, num_aggregates);

        // [2, 2, 1, 2, 2, 1] -> [1, 1, 0, 1, 1, 0]
        thrust::transform(exec,
                          aggregate_counts.begin(), aggregate_counts.end(),
                          thrust::constant_iterator<IndexType>(1), isone.begin(),
                          thrust::equal_to<IndexType>());
        // [1, 1, 0, 1, 1, 0] -> [0, 1, 2, 2, 3, 3]
        thrust::exclusive_scan(exec,
                               thrust::make_transform_iterator(isone.begin(), thrust::logical_not<bool>()),
                               thrust::make_transform_iterator(isone.end()  , thrust::logical_not<bool>()),
                               aggregate_ids.begin());
        // [0, 1, 2, 2, 3, 3] -> [0, 1, -1, 2, 3, -1]
        thrust::scatter_if(exec,
                           thrust::constant_iterator<IndexType>(-1),
                           thrust::constant_iterator<IndexType>(-1) + num_aggregates,
                           thrust::counting_iterator<IndexType>(0),
                           isone.begin(),
                           aggregate_ids.begin());

        thrust::gather(exec, aggregates.begin(), aggregates.end(), aggregate_ids.begin(), aggregates.begin());
    }
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void mis_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                   const MatrixType& C,
                         ArrayType1& aggregates,
                         ArrayType2& mis,
                         cusp::known_format)
{
    typedef typename MatrixType::const_coo_view_type MatrixViewType;

    MatrixViewType C_(C);

    cusp::precond::aggregation::mis_aggregate(exec, C_ , aggregates, mis);
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void mis_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                   const MatrixType& C,
                         ArrayType1& aggregates,
                         ArrayType2& mis)
{
    typedef typename MatrixType::format Format;

    Format format;

    mis_aggregate(thrust::detail::derived_cast(exec), C, aggregates, mis, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

