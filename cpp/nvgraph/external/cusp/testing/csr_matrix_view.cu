#include <unittest/unittest.h>

#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

template <typename MemorySpace>
void TestCsrMatrixView(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::csr_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView;
    typedef typename cusp::csr_matrix_view<IndexView,IndexView,ValueView> View;

    Matrix M(3, 2, 6);

    View V(3, 2, 6,
           cusp::make_array1d_view(M.row_offsets.begin(),    M.row_offsets.end()),
           cusp::make_array1d_view(M.column_indices.begin(), M.column_indices.end()),
           cusp::make_array1d_view(M.values.begin(),         M.values.end()));

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.row_offsets.begin(),    M.row_offsets.begin());
    ASSERT_EQUAL_QUIET(V.row_offsets.end(),      M.row_offsets.end());
    ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());

    View W(M);

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    2);
    ASSERT_EQUAL(W.num_entries, 6);

    ASSERT_EQUAL_QUIET(W.row_offsets.begin(),    M.row_offsets.begin());
    ASSERT_EQUAL_QUIET(W.row_offsets.end(),      M.row_offsets.end());
    ASSERT_EQUAL_QUIET(W.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(W.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(W.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(W.values.end(),           M.values.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrMatrixView);


template <typename MemorySpace>
void TestCsrMatrixViewAssignment(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::csr_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView;
    typedef typename cusp::csr_matrix_view<IndexView,IndexView,ValueView> View;

    Matrix M(3, 2, 6);

    View V = M;

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.row_offsets.begin(),    M.row_offsets.begin());
    ASSERT_EQUAL_QUIET(V.row_offsets.end(),      M.row_offsets.end());
    ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());

    View W = V;

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    2);
    ASSERT_EQUAL(W.num_entries, 6);

    ASSERT_EQUAL_QUIET(W.row_offsets.begin(),    M.row_offsets.begin());
    ASSERT_EQUAL_QUIET(W.row_offsets.end(),      M.row_offsets.end());
    ASSERT_EQUAL_QUIET(W.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(W.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(W.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(W.values.end(),           M.values.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrMatrixViewAssignment);


template <typename MemorySpace>
void TestMakeCsrMatrixView(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::csr_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView;
    typedef typename cusp::csr_matrix_view<IndexView,IndexView,ValueView> View;

    // construct view from parts
    {
        Matrix M(3, 2, 6);

        View V =
            cusp::make_csr_matrix_view(3, 2, 6,
                                       cusp::make_array1d_view(M.row_offsets),
                                       cusp::make_array1d_view(M.column_indices),
                                       cusp::make_array1d_view(M.values));

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.row_offsets[0] = 0;
        V.column_indices[0] = 1;
        V.values[0] = 2;

        ASSERT_EQUAL_QUIET(V.row_offsets.begin(),    M.row_offsets.begin());
        ASSERT_EQUAL_QUIET(V.row_offsets.end(),      M.row_offsets.end());
        ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());
    }

    // construct view from matrix
    {
        Matrix M(3, 2, 6);

        View V = cusp::make_csr_matrix_view(M);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.row_offsets[0] = 0;
        V.column_indices[0] = 1;
        V.values[0] = 2;

        ASSERT_EQUAL_QUIET(V.row_offsets.begin(),    M.row_offsets.begin());
        ASSERT_EQUAL_QUIET(V.row_offsets.end(),      M.row_offsets.end());
        ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());
    }

    // construct view from view
    {
        Matrix M(3, 2, 6);

        View X = cusp::make_csr_matrix_view(M);
        View V = cusp::make_csr_matrix_view(X);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.row_offsets[0] = 0;
        V.column_indices[0] = 1;
        V.values[0] = 2;

        ASSERT_EQUAL_QUIET(V.row_offsets.begin(),    M.row_offsets.begin());
        ASSERT_EQUAL_QUIET(V.row_offsets.end(),      M.row_offsets.end());
        ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());
    }

    // construct view from const matrix
    {
        const Matrix M(3, 2, 6);

        ASSERT_EQUAL(cusp::make_csr_matrix_view(M).num_rows,    3);
        ASSERT_EQUAL(cusp::make_csr_matrix_view(M).num_cols,    2);
        ASSERT_EQUAL(cusp::make_csr_matrix_view(M).num_entries, 6);

        ASSERT_EQUAL_QUIET(cusp::make_csr_matrix_view(M).row_offsets.begin(),    M.row_offsets.begin());
        ASSERT_EQUAL_QUIET(cusp::make_csr_matrix_view(M).row_offsets.end(),      M.row_offsets.end());
        ASSERT_EQUAL_QUIET(cusp::make_csr_matrix_view(M).column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(cusp::make_csr_matrix_view(M).column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(cusp::make_csr_matrix_view(M).values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(cusp::make_csr_matrix_view(M).values.end(),           M.values.end());
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestMakeCsrMatrixView);

template <typename MemorySpace>
void TestCsrToCooMatrixView(void)
{
    typedef int   IndexType;
    typedef float ValueType;

    typedef cusp::csr_matrix<IndexType,ValueType,MemorySpace> TestMatrix;
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

    V.column_indices[0] = -1;
    V.values[0] = -1;

    ASSERT_EQUAL(M.column_indices[0], -1);
    ASSERT_EQUAL(M.values[0],         -1);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrToCooMatrixView);

