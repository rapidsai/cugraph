#include <unittest/unittest.h>

#include <cusp/dia_matrix.h>
#include <cusp/multiply.h>

template <typename MemorySpace>
void TestDiaMatrixView(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::dia_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView1d;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView1d;
    typedef typename cusp::array2d_view<ValueView1d,cusp::column_major>   ValueView2d;
    typedef typename cusp::dia_matrix_view<IndexView1d,ValueView2d>       View;

    Matrix M(3, 2, 6, 4);

    View V(3, 2, 6,
           cusp::make_array1d_view(M.diagonal_offsets),
           cusp::make_array2d_view(M.values));

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.diagonal_offsets.begin(), M.diagonal_offsets.begin());
    ASSERT_EQUAL_QUIET(V.diagonal_offsets.end(),   M.diagonal_offsets.end());
    ASSERT_EQUAL_QUIET(V.values.num_rows,          M.values.num_rows);
    ASSERT_EQUAL_QUIET(V.values.num_cols,          M.values.num_cols);
    ASSERT_EQUAL_QUIET(V.values.num_entries,       M.values.num_entries);
    ASSERT_EQUAL_QUIET(V.values.pitch,             M.values.pitch);
    ASSERT_EQUAL_QUIET(V.values.values.begin(),    M.values.values.begin());
    ASSERT_EQUAL_QUIET(V.values.values.end(),      M.values.values.end());

    View W(M);

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    2);
    ASSERT_EQUAL(W.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.diagonal_offsets.begin(), M.diagonal_offsets.begin());
    ASSERT_EQUAL_QUIET(V.diagonal_offsets.end(),   M.diagonal_offsets.end());
    ASSERT_EQUAL_QUIET(V.values.num_rows,          M.values.num_rows);
    ASSERT_EQUAL_QUIET(V.values.num_cols,          M.values.num_cols);
    ASSERT_EQUAL_QUIET(V.values.num_entries,       M.values.num_entries);
    ASSERT_EQUAL_QUIET(V.values.pitch,             M.values.pitch);
    ASSERT_EQUAL_QUIET(V.values.values.begin(),    M.values.values.begin());
    ASSERT_EQUAL_QUIET(V.values.values.end(),      M.values.values.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixView);


template <typename MemorySpace>
void TestDiaMatrixViewAssignment(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::dia_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView1d;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView1d;
    typedef typename cusp::array2d_view<ValueView1d,cusp::column_major>   ValueView2d;
    typedef typename cusp::dia_matrix_view<IndexView1d,ValueView2d>       View;

    Matrix M(3, 2, 6, 4);

    View V = M;

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.diagonal_offsets.begin(), M.diagonal_offsets.begin());
    ASSERT_EQUAL_QUIET(V.diagonal_offsets.end(),   M.diagonal_offsets.end());
    ASSERT_EQUAL_QUIET(V.values.num_rows,          M.values.num_rows);
    ASSERT_EQUAL_QUIET(V.values.num_cols,          M.values.num_cols);
    ASSERT_EQUAL_QUIET(V.values.num_entries,       M.values.num_entries);
    ASSERT_EQUAL_QUIET(V.values.pitch,             M.values.pitch);
    ASSERT_EQUAL_QUIET(V.values.values.begin(),    M.values.values.begin());
    ASSERT_EQUAL_QUIET(V.values.values.end(),      M.values.values.end());

    View W = V;

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    2);
    ASSERT_EQUAL(W.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.diagonal_offsets.begin(), M.diagonal_offsets.begin());
    ASSERT_EQUAL_QUIET(V.diagonal_offsets.end(),   M.diagonal_offsets.end());
    ASSERT_EQUAL_QUIET(V.values.num_rows,          M.values.num_rows);
    ASSERT_EQUAL_QUIET(V.values.num_cols,          M.values.num_cols);
    ASSERT_EQUAL_QUIET(V.values.num_entries,       M.values.num_entries);
    ASSERT_EQUAL_QUIET(V.values.pitch,             M.values.pitch);
    ASSERT_EQUAL_QUIET(V.values.values.begin(),    M.values.values.begin());
    ASSERT_EQUAL_QUIET(V.values.values.end(),      M.values.values.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixViewAssignment);


template <typename MemorySpace>
void TestMakeDiaMatrixView(void)
{
    typedef int                                                               IndexType;
    typedef float                                                             ValueType;
    typedef typename cusp::dia_matrix<IndexType,ValueType,MemorySpace>        Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator           IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator           ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                        IndexView1d;
    typedef typename cusp::array1d_view<ValueIterator>                        ValueView1d;
    typedef typename cusp::array2d_view<ValueView1d,cusp::column_major>       ValueView2d;
    typedef typename cusp::dia_matrix_view<IndexView1d,ValueView2d>           View;
    typedef typename cusp::array1d<IndexType,MemorySpace>::const_iterator     ConstIndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::const_iterator     ConstValueIterator;
    typedef typename cusp::array1d_view<ConstIndexIterator>                   ConstIndexView1d;
    typedef typename cusp::array1d_view<ConstValueIterator>                   ConstValueView1d;
    typedef typename cusp::array2d_view<ConstValueView1d,cusp::column_major>  ConstValueView2d;
    typedef typename cusp::dia_matrix_view<ConstIndexView1d,ConstValueView2d> ConstView;

    // construct view from parts
    {
        Matrix M(3, 2, 6, 4);

        View V =
            cusp::make_dia_matrix_view(3, 2, 6,
                                       cusp::make_array1d_view(M.diagonal_offsets),
                                       cusp::make_array2d_view(M.values));

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.diagonal_offsets[0] = 1;
        V.values(0,0) = 20;

        ASSERT_EQUAL_QUIET(V.diagonal_offsets, M.diagonal_offsets);
        ASSERT_EQUAL_QUIET(V.values,           M.values);
    }

    // construct view from matrix
    {
        Matrix M(3, 2, 6, 4);

        View V = cusp::make_dia_matrix_view(M);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.diagonal_offsets[0] = 1;
        V.values(0,0) = 20;

        ASSERT_EQUAL_QUIET(V.diagonal_offsets, M.diagonal_offsets);
        ASSERT_EQUAL_QUIET(V.values,           M.values);
    }

    // construct view from view
    {
        Matrix M(3, 2, 6, 4);

        View X = cusp::make_dia_matrix_view(M);
        View V = cusp::make_dia_matrix_view(X);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.diagonal_offsets[0] = 1;
        V.values(0,0) = 20;

        ASSERT_EQUAL_QUIET(V.diagonal_offsets, M.diagonal_offsets);
        ASSERT_EQUAL_QUIET(V.values,           M.values);
    }

    // construct view from const matrix
    {
        const Matrix M(3, 2, 6, 4);

        ConstView V = cusp::make_dia_matrix_view(M);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        ASSERT_EQUAL_QUIET(V.diagonal_offsets, M.diagonal_offsets);
        ASSERT_EQUAL_QUIET(V.values,           M.values);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestMakeDiaMatrixView);

template <typename MemorySpace>
void TestDiaToCooMatrixView(void)
{
    typedef int   IndexType;
    typedef float ValueType;

    typedef cusp::dia_matrix<IndexType,ValueType,MemorySpace> TestMatrix;
    typedef typename TestMatrix::coo_view_type View;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> A(3, 2, 6);
    A.row_indices[0] = 0;
    A.column_indices[0] = 0;
    A.values[0] = 1;
    A.row_indices[1] = 0;
    A.column_indices[1] = 1;
    A.values[1] = 2;
    A.row_indices[2] = 1;
    A.column_indices[2] = 0;
    A.values[2] = 3;
    A.row_indices[3] = 1;
    A.column_indices[3] = 1;
    A.values[3] = 4;
    A.row_indices[4] = 2;
    A.column_indices[4] = 0;
    A.values[4] = 5;
    A.row_indices[5] = 2;
    A.column_indices[5] = 1;
    A.values[5] = 6;

    TestMatrix M(A);
    View V(M);

    V.values[0] = -1;

    ASSERT_EQUAL(M.values(0,2), -1);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaToCooMatrixView);

