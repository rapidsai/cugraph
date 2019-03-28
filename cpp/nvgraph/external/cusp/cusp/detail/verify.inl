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

#include <cusp/detail/format.h>
#include <cusp/exception.h>

#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

#include <sstream>

namespace cusp
{
namespace detail
{

///////////////////////////////////
// Helper functions and functors //
///////////////////////////////////

template <typename IndexVector>
thrust::pair<typename IndexVector::value_type, typename IndexVector::value_type>
index_range(const IndexVector& indices)
{
//    // return a pair<> containing the min and max value in a range
//    thrust::pair<typename IndexVector::const_iterator, typename IndexVector::const_iterator> iter = thrust::minmax_element(indices.begin(), indices.end());
//    return thrust::make_pair(*iter.first, *iter.second);

    // WAR lack of const_iterator in array1d_view
    return thrust::make_pair
           (*thrust::min_element(indices.begin(), indices.end()),
            *thrust::max_element(indices.begin(), indices.end()));
}

template <typename IndexType>
struct is_ell_entry
{
    IndexType num_rows;
    IndexType pitch;
    IndexType invalid_index;

    is_ell_entry(IndexType num_rows, IndexType pitch, IndexType invalid_index)
        : num_rows(num_rows), pitch(pitch), invalid_index(invalid_index) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        IndexType n = thrust::get<0>(t);
        IndexType j = thrust::get<1>(t);
        return (n % pitch < num_rows) && (j != invalid_index);
    }
};

template <typename IndexType>
struct is_ell_entry_in_bounds
{
    IndexType num_rows;
    IndexType num_cols;
    IndexType pitch;
    IndexType invalid_index;

    is_ell_entry_in_bounds(IndexType num_rows, IndexType num_cols, IndexType pitch, IndexType invalid_index)
        : num_rows(num_rows), num_cols(num_cols), pitch(pitch), invalid_index(invalid_index) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        IndexType n = thrust::get<0>(t);
        IndexType j = thrust::get<1>(t);
        return (n % pitch < num_rows) && (j != invalid_index) && (j >= 0) && (j < num_cols);
    }
};


///////////////////////////////
// Matrix-Specific Functions //
///////////////////////////////

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A,
                     OutputStream& ostream,
                     cusp::coo_format)
{
    typedef typename MatrixType::index_type IndexType;

    // we could relax some of these conditions if necessary
    if (A.row_indices.size() != A.num_entries)
    {
        ostream << "size of row_indices (" << A.row_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.column_indices.size() != A.num_entries)
    {
        ostream << "size of column_indices (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.values.size() != A.num_entries)
    {
        ostream << "size of values (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.num_entries > 0)
    {
        // check that row indices are within [0, num_rows)
        thrust::pair<IndexType,IndexType> min_max_row = index_range(A.row_indices);
        if (min_max_row.first < 0)
        {
            ostream << "row indices should be non-negative";
            return false;
        }
        if (static_cast<size_t>(min_max_row.second) >= A.num_rows)
        {
            ostream << "row indices should be less than num_row (" << A.num_rows << ")";
            return false;
        }

        // check that row_indices is a non-decreasing sequence
        if (!thrust::is_sorted(A.row_indices.begin(), A.row_indices.end()))
        {
            ostream << "row indices should form a non-decreasing sequence";
            return false;
        }

        // check that column indices are within [0, num_cols)
        thrust::pair<IndexType,IndexType> min_max_col = index_range(A.column_indices);
        if (min_max_col.first < 0)
        {
            ostream << "column indices should be non-negative";
            return false;
        }
        if (static_cast<size_t>(min_max_col.second) >= A.num_cols)
        {
            ostream << "column indices should be less than num_cols (" << A.num_cols << ")";
            return false;
        }
    }

    return true;
}


template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A,
                     OutputStream& ostream,
                     cusp::csr_format)
{
    typedef typename MatrixType::index_type IndexType;

    // we could relax some of these conditions if necessary

    if (A.row_offsets.size() != A.num_rows + 1)
    {
        ostream << "size of row_offsets (" << A.row_offsets.size() << ") "
                << "should be equal to num_rows + 1 (" << (A.num_rows + 1) << ")";
        return false;
    }

    if (A.row_offsets.front() != IndexType(0))
    {
        ostream << "first value in row_offsets (" << A.row_offsets.front() << ") "
                << "should be equal to 0";
        return false;
    }

    // TODO is this overly strict?
    if (static_cast<size_t>(A.row_offsets.back()) != A.num_entries)
    {
        ostream << "last value in row_offsets (" << A.row_offsets.back() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.column_indices.size() != A.num_entries)
    {
        ostream << "size of column_indices (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.values.size() != A.num_entries)
    {
        ostream << "size of values (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

    // check that row_offsets is a non-decreasing sequence
    if (!thrust::is_sorted(A.row_offsets.begin(), A.row_offsets.end()))
    {
        ostream << "row offsets should form a non-decreasing sequence";
        return false;
    }

    if (A.num_entries > 0)
    {
        // check that column indices are within [0, num_cols)
        thrust::pair<IndexType,IndexType> min_max = index_range(A.column_indices);

        if (min_max.first < 0)
        {
            ostream << "column indices should be non-negative";
            return false;
        }
        if (static_cast<size_t>(min_max.second) >= A.num_cols)
        {
            ostream << "column indices should be less than num_cols (" << A.num_cols << ")";
            return false;
        }
    }

    return true;
}


template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A,
                     OutputStream& ostream,
                     cusp::dia_format)
{
    if (A.num_rows > A.values.num_rows)
    {
        ostream << "number of rows in values array (" << A.values.num_rows << ") ";
        ostream << "should be >= num_rows (" << A.num_rows << ")";
        return false;
    }

    if (A.num_rows > A.values.num_rows)
    {
        ostream << "number of rows in values array (" << A.values.num_rows << ") ";
        ostream << "should be >= num_rows (" << A.num_rows << ")";
        return false;
    }

    return true;
}

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A,
                     OutputStream& ostream,
                     cusp::ell_format)
{
    typedef typename MatrixType::index_type IndexType;

    const IndexType invalid_index = MatrixType::invalid_index;

    if (A.column_indices.num_rows != A.values.num_rows ||
            A.column_indices.num_cols != A.values.num_cols)
    {
        ostream << "shape of column_indices array (" << A.column_indices.num_rows << "," << A.column_indices.num_cols << ") ";
        ostream << "should agree with the values array (" << A.values.num_rows << "," << A.values.num_cols << ")";
        return false;
    }

    if (A.num_rows > A.values.num_rows)
    {
        ostream << "number of rows in values array (" << A.values.num_rows << ") ";
        ostream << "should be >= num_rows (" << A.num_rows << ")";
        return false;
    }

    // count true number of entries in ell structure
    size_t true_num_entries =
        thrust::count_if(thrust::make_zip_iterator
                         (
                             thrust::make_tuple(thrust::counting_iterator<IndexType>(0),
                                     A.column_indices.values.begin())
                         ),
                         thrust::make_zip_iterator
                         (
                             thrust::make_tuple(thrust::counting_iterator<IndexType>(0),
                                     A.column_indices.values.begin())
                         ) + A.column_indices.values.size(),
                         is_ell_entry<IndexType>(A.num_rows, A.column_indices.pitch, invalid_index));

    if (A.num_entries != true_num_entries)
    {
        ostream << "number of valid column indices (" << true_num_entries << ") ";
        ostream << "should be == num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.num_entries > 0)
    {
        // check that column indices are in [0, num_cols)
        size_t num_entries_in_bounds =
            thrust::count_if(thrust::make_zip_iterator
                             (
                                 thrust::make_tuple(thrust::counting_iterator<IndexType>(0),
                                         A.column_indices.values.begin())
                             ),
                             thrust::make_zip_iterator
                             (
                                 thrust::make_tuple(thrust::counting_iterator<IndexType>(0),
                                         A.column_indices.values.begin())
                             ) + A.column_indices.values.size(),
                             is_ell_entry_in_bounds<IndexType>(A.num_rows, A.num_cols, A.column_indices.pitch, invalid_index));
        if (num_entries_in_bounds != true_num_entries)
        {
            ostream << "matrix contains (" << (true_num_entries - num_entries_in_bounds) << ") out-of-bounds column indices";
            return false;
        }
    }

    return true;
}

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A,
                     OutputStream& ostream,
                     cusp::hyb_format)
{
    // make sure redundant shapes values agree
    if (A.num_rows != A.ell.num_rows || A.num_rows != A.coo.num_rows ||
            A.num_cols != A.ell.num_cols || A.num_cols != A.coo.num_cols)
    {
        ostream << "matrix shape (" << A.num_rows << "," << A.num_cols << ") ";
        ostream << "should be equal to shape of ELL part (" << A.ell.num_rows << "," << A.ell.num_cols << ") and ";
        ostream << "COO part (" << A.coo.num_rows << "," << A.coo.num_cols << ")";
        return false;
    }

    // check that num_entries = A.ell.num_entries + A.coo.num_entries
    if (A.num_entries != A.ell.num_entries + A.coo.num_entries)
    {
        ostream << "num_entries (" << A.num_entries << ") ";
        ostream << "should be equal to sum of ELL num_entries (" << A.ell.num_entries << ") and ";
        ostream << "COO num_entries (" << A.coo.num_entries << ")";
        return false;
    }

    return cusp::is_valid_matrix(A.ell, ostream) && cusp::is_valid_matrix(A.coo, ostream);
}


template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A,
                     OutputStream& ostream,
                     cusp::array2d_format)
{
    if (A.num_rows * A.num_cols != A.num_entries)
    {
        ostream << "product of matrix dimensions (" << A.num_rows << "," << A.num_cols << ") ";
        ostream << "should equal num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.num_entries != A.values.size())
    {
        ostream << "num_entries (" << A.num_entries << ") ";
        ostream << "should agree with size of values array (" << A.values.size() << ")";
        return false;
    }

    // TODO check .pitch

    return true;
}

} // end namespace detail


//////////////////
// Entry points //
//////////////////

template <typename MatrixType>
bool is_valid_matrix(const MatrixType& A)
{
    std::ostringstream oss;
    return cusp::is_valid_matrix(A, oss);
}

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A, OutputStream& ostream)
{
    // dispatch on matrix format
    return detail::is_valid_matrix(A, ostream, typename MatrixType::format());
}

template <typename MatrixType>
void assert_is_valid_matrix(const MatrixType& A)
{
    std::ostringstream oss;
    bool is_valid = cusp::is_valid_matrix(A, oss);

    if (!is_valid)
        throw cusp::format_exception(oss.str());
}

template <typename Array1, typename Array2>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2)
{
    if(array1.size() != array2.size())
        throw cusp::invalid_input_exception("array dimensions do not match");
}

template <typename Array1, typename Array2, typename Array3>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3)
{
    assert_same_dimensions(array1, array2);
    assert_same_dimensions(array2, array3);
}

template <typename Array1, typename Array2, typename Array3, typename Array4>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3,
                            const Array4& array4)
{
    assert_same_dimensions(array1, array2);
    assert_same_dimensions(array2, array3);
    assert_same_dimensions(array3, array4);
}

} // end namespace cusp

