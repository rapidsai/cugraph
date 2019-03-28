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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>

#include <cusp/functional.h>
#include <cusp/complex.h>
#include <cusp/iterator/join_iterator.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{

template <typename, typename>           class array1d;
template <typename, typename, typename> class array2d;
template <typename, typename, typename> class dia_matrix;
template <typename, typename, typename> class coo_matrix;
template <typename, typename, typename> class csr_matrix;
template <typename, typename, typename> class ell_matrix;
template <typename, typename, typename> class hyb_matrix;

template <typename> class array1d_view;
template <typename, typename, typename, typename, typename, typename> class coo_matrix_view;

namespace detail
{

template <typename ,typename ,typename> struct logical_to_other_physical_functor;

template<typename MatrixType, typename CompareTag>
struct is_matrix_type
  : thrust::detail::integral_constant<bool,thrust::detail::is_same<typename MatrixType::format,CompareTag>::value>
{};

template<typename MatrixType> struct is_array2d : is_matrix_type<MatrixType,array2d_format> {};
template<typename MatrixType> struct is_coo     : is_matrix_type<MatrixType,coo_format> {};
template<typename MatrixType> struct is_csr     : is_matrix_type<MatrixType,csr_format> {};
template<typename MatrixType> struct is_dia     : is_matrix_type<MatrixType,dia_format> {};
template<typename MatrixType> struct is_ell     : is_matrix_type<MatrixType,ell_format> {};
template<typename MatrixType> struct is_hyb     : is_matrix_type<MatrixType,hyb_format> {};

template<typename IndexType, typename ValueType, typename MemorySpace, typename FormatTag> struct matrix_type {};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,array1d_format>
{
    typedef cusp::array1d<ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,array2d_format>
{
    typedef cusp::array2d<ValueType,MemorySpace,cusp::row_major> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,dia_format>
{
    typedef cusp::dia_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,coo_format>
{
    typedef cusp::coo_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,csr_format>
{
    typedef cusp::csr_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,ell_format>
{
    typedef cusp::ell_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,hyb_format>
{
    typedef cusp::hyb_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename MatrixType, typename Format = typename MatrixType::format>
struct get_index_type
{
    typedef typename MatrixType::index_type type;
};

template<typename MatrixType>
struct get_index_type<MatrixType,array1d_format>
{
    typedef int type;
};

template<typename MatrixType, typename MemorySpace, typename FormatTag>
struct as_matrix_type
{
    typedef typename get_index_type<MatrixType>::type IndexType;
    typedef typename MatrixType::value_type           ValueType;

    typedef typename matrix_type<IndexType,ValueType,MemorySpace,FormatTag>::type type;
};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_array2d_type : as_matrix_type<MatrixType,MemorySpace,array2d_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_dia_type : as_matrix_type<MatrixType,MemorySpace,dia_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_coo_type : as_matrix_type<MatrixType,MemorySpace,coo_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_csr_type : as_matrix_type<MatrixType,MemorySpace,csr_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_ell_type : as_matrix_type<MatrixType,MemorySpace,ell_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_hyb_type : as_matrix_type<MatrixType,MemorySpace,hyb_format> {};

template<typename MatrixType,typename FormatTag = typename MatrixType::format>
struct coo_view_type{};

template<typename MatrixType>
struct coo_view_type<MatrixType,csr_format>
{
    typedef typename MatrixType::index_type                                                              IndexType;
    typedef typename MatrixType::value_type                                                              ValueType;
    typedef typename MatrixType::memory_space                                                            MemorySpace;

    typedef typename thrust::detail::remove_const<IndexType>::type                                       TempType;
    typedef cusp::array1d<TempType,MemorySpace>                                                          Array1;
    typedef cusp::array1d_view<typename MatrixType::column_indices_array_type::iterator>                 Array2;
    typedef cusp::array1d_view<typename MatrixType::values_array_type::iterator>                         Array3;

    typedef cusp::coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>                  view;
};

template<typename MatrixType>
struct coo_view_type<MatrixType,dia_format>
{
    typedef typename MatrixType::index_type                                                              IndexType;
    typedef typename MatrixType::value_type                                                              ValueType;
    typedef typename MatrixType::memory_space                                                            MemorySpace;

    typedef typename thrust::detail::remove_const<IndexType>::type                                       TempType;
    typedef typename thrust::counting_iterator<TempType>                                                 CountingIterator;
    typedef typename thrust::transform_iterator<cusp::divide_value<TempType>, CountingIterator>          RowIndexIterator;
    typedef typename cusp::array1d<TempType,MemorySpace>::iterator                                       IndexIterator;

    typedef typename MatrixType::diagonal_offsets_array_type::iterator                                   OffsetsIterator;
    typedef typename thrust::transform_iterator<modulus_value<TempType>, CountingIterator>               ModulusIterator;
    typedef typename thrust::permutation_iterator<OffsetsIterator,ModulusIterator>                       OffsetsPermIterator;
    typedef typename thrust::tuple<OffsetsPermIterator, RowIndexIterator>                                IteratorTuple;
    typedef typename thrust::zip_iterator<IteratorTuple>                                                 ZipIterator;
    typedef typename thrust::transform_iterator<sum_pair_functor<IndexType>, ZipIterator>               ColumnIndexIterator;

    typedef typename MatrixType::values_array_type::values_array_type::iterator                          ValueIterator;
    typedef logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>            PermFunctor;
    typedef typename thrust::transform_iterator<PermFunctor, CountingIterator>                           PermIndexIterator;
    typedef typename thrust::permutation_iterator<ValueIterator, PermIndexIterator>                      PermValueIterator;

    typedef thrust::permutation_iterator<RowIndexIterator, IndexIterator>                                RowPermIterator;
    typedef thrust::permutation_iterator<ColumnIndexIterator, IndexIterator>                             ColumnPermIterator;
    typedef thrust::permutation_iterator<PermValueIterator, IndexIterator>                               ValuePermIterator;

    typedef cusp::array1d_view<RowPermIterator>                                                          Array1;
    typedef cusp::array1d_view<ColumnPermIterator>                                                       Array2;
    typedef cusp::array1d_view<ValuePermIterator>                                                        Array3;

    typedef cusp::coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>                  view;
};

template<typename MatrixType>
struct coo_view_type<MatrixType,ell_format>
{
    typedef typename MatrixType::index_type                                                              IndexType;
    typedef typename MatrixType::value_type                                                              ValueType;
    typedef typename MatrixType::memory_space                                                            MemorySpace;

    typedef typename thrust::detail::remove_const<IndexType>::type                                       TempType;
    typedef typename thrust::counting_iterator<TempType>                                                 CountingIterator;
    typedef thrust::transform_iterator<cusp::divide_value<TempType>, CountingIterator>                   RowIndexIterator;
    typedef typename MatrixType::column_indices_array_type::values_array_type::iterator                  ColumnIndexIterator;
    typedef typename MatrixType::values_array_type::values_array_type::iterator                          ValueIterator;

    typedef cusp::detail::logical_to_other_physical_functor<TempType, cusp::row_major, cusp::column_major> PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, CountingIterator>                                    PermIndexIterator;
    typedef thrust::permutation_iterator<ColumnIndexIterator, PermIndexIterator>                         PermColumnIndexIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                               PermValueIterator;

    typedef typename cusp::array1d<TempType,MemorySpace>::iterator                                       IndexIterator;
    typedef thrust::permutation_iterator<RowIndexIterator, IndexIterator>                                RowPermIterator;
    typedef thrust::permutation_iterator<PermColumnIndexIterator, IndexIterator>                         ColumnPermIterator;
    typedef thrust::permutation_iterator<PermValueIterator, IndexIterator>                               ValuePermIterator;

    typedef cusp::array1d_view<RowPermIterator>                                                          Array1;
    typedef cusp::array1d_view<ColumnPermIterator>                                                       Array2;
    typedef cusp::array1d_view<ValuePermIterator>                                                        Array3;

    typedef cusp::coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>                  view;
};

template<typename MatrixType>
struct coo_view_type<MatrixType,hyb_format>
{
    typedef typename MatrixType::index_type                                                              IndexType;
    typedef typename MatrixType::value_type                                                              ValueType;
    typedef typename MatrixType::memory_space                                                            MemorySpace;
    typedef typename thrust::detail::remove_const<IndexType>::type                                       TempType;

    typedef typename MatrixType::ell_matrix_type                                                         ell_matrix_type;
    typedef typename MatrixType::coo_matrix_type                                                         coo_matrix_type;

    typedef coo_view_type<ell_matrix_type>                                                               ell_view_type;
    typedef typename coo_matrix_type::view                                                               coo_view;

    typedef typename ell_view_type::PermIndexIterator                                                    EllPermIndexIterator;
    typedef typename ell_view_type::RowIndexIterator                                                     EllRowIndexIterator;
    typedef typename ell_view_type::PermColumnIndexIterator                                              EllColumnIndexIterator;
    typedef typename ell_view_type::PermValueIterator                                                    EllValueIterator;

    typedef typename coo_view::row_indices_array_type::iterator                                          CooRowIndexIterator;
    typedef typename coo_view::column_indices_array_type::iterator                                       CooColumnIndexIterator;
    typedef typename coo_view::values_array_type::iterator                                               CooValueIterator;

    typedef typename cusp::array1d<TempType,MemorySpace>::iterator                                       IndexIterator;
    typedef cusp::join_iterator< thrust::tuple<EllRowIndexIterator,   CooRowIndexIterator,   IndexIterator> >  JoinRowIterator;
    typedef cusp::join_iterator< thrust::tuple<EllColumnIndexIterator,CooColumnIndexIterator,IndexIterator> >  JoinColumnIterator;
    typedef cusp::join_iterator< thrust::tuple<EllValueIterator,      CooValueIterator,      IndexIterator> >  JoinValueIterator;

    typedef cusp::array1d_view<typename JoinRowIterator::iterator>                                       Array1;
    typedef cusp::array1d_view<typename JoinColumnIterator::iterator>                                    Array2;
    typedef cusp::array1d_view<typename JoinValueIterator::iterator>                                     Array3;

    typedef cusp::coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>                  view;
};

} // end detail
} // end cusp

