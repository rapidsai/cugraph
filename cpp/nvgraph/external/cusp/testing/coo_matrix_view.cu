#include <unittest/unittest.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>

template <typename MemorySpace>
void TestCooMatrixView(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::coo_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView;
    typedef typename cusp::coo_matrix_view<IndexView,IndexView,ValueView> View;

    Matrix M(3, 2, 6);

    View V(3, 2, 6,
           cusp::make_array1d_view(M.row_indices.begin(),    M.row_indices.end()),
           cusp::make_array1d_view(M.column_indices.begin(), M.column_indices.end()),
           cusp::make_array1d_view(M.values.begin(),         M.values.end()));

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.row_indices.begin(),    M.row_indices.begin());
    ASSERT_EQUAL_QUIET(V.row_indices.end(),      M.row_indices.end());
    ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());

    View W(M);

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    2);
    ASSERT_EQUAL(W.num_entries, 6);

    ASSERT_EQUAL_QUIET(W.row_indices.begin(),    M.row_indices.begin());
    ASSERT_EQUAL_QUIET(W.row_indices.end(),      M.row_indices.end());
    ASSERT_EQUAL_QUIET(W.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(W.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(W.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(W.values.end(),           M.values.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixView);


template <typename MemorySpace>
void TestCooMatrixViewAssignment(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::coo_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView;
    typedef typename cusp::coo_matrix_view<IndexView,IndexView,ValueView> View;

    Matrix M(3, 2, 6);

    View V = M;

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.row_indices.begin(),    M.row_indices.begin());
    ASSERT_EQUAL_QUIET(V.row_indices.end(),      M.row_indices.end());
    ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());

    View W = V;

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    2);
    ASSERT_EQUAL(W.num_entries, 6);

    ASSERT_EQUAL_QUIET(W.row_indices.begin(),    M.row_indices.begin());
    ASSERT_EQUAL_QUIET(W.row_indices.end(),      M.row_indices.end());
    ASSERT_EQUAL_QUIET(W.column_indices.begin(), M.column_indices.begin());
    ASSERT_EQUAL_QUIET(W.column_indices.end(),   M.column_indices.end());
    ASSERT_EQUAL_QUIET(W.values.begin(),         M.values.begin());
    ASSERT_EQUAL_QUIET(W.values.end(),           M.values.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixViewAssignment);


template <typename MemorySpace>
void TestMakeCooMatrixView(void)
{
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::coo_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView;
    typedef typename cusp::coo_matrix_view<IndexView,IndexView,ValueView> View;

    // construct view from parts
    {
        Matrix M(3, 2, 6);

        View V =
            cusp::make_coo_matrix_view(3, 2, 6,
                                       cusp::make_array1d_view(M.row_indices),
                                       cusp::make_array1d_view(M.column_indices),
                                       cusp::make_array1d_view(M.values));

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.row_indices[0] = 0;
        V.column_indices[0] = 1;
        V.values[0] = 2;

        ASSERT_EQUAL_QUIET(V.row_indices.begin(),    M.row_indices.begin());
        ASSERT_EQUAL_QUIET(V.row_indices.end(),      M.row_indices.end());
        ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());
    }

    // construct view from matrix
    {
        Matrix M(3, 2, 6);

        View V = cusp::make_coo_matrix_view(M);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.row_indices[0] = 0;
        V.column_indices[0] = 1;
        V.values[0] = 2;

        ASSERT_EQUAL_QUIET(V.row_indices.begin(),    M.row_indices.begin());
        ASSERT_EQUAL_QUIET(V.row_indices.end(),      M.row_indices.end());
        ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());
    }

    // construct view from view
    {
        Matrix M(3, 2, 6);

        View X = cusp::make_coo_matrix_view(M);
        View V = cusp::make_coo_matrix_view(X);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 6);

        V.row_indices[0] = 0;
        V.column_indices[0] = 1;
        V.values[0] = 2;

        ASSERT_EQUAL_QUIET(V.row_indices.begin(),    M.row_indices.begin());
        ASSERT_EQUAL_QUIET(V.row_indices.end(),      M.row_indices.end());
        ASSERT_EQUAL_QUIET(V.column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(V.values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(V.values.end(),           M.values.end());
    }

    // construct view from const matrix
    {
        const Matrix M(3, 2, 6);

        ASSERT_EQUAL(cusp::make_coo_matrix_view(M).num_rows,    3);
        ASSERT_EQUAL(cusp::make_coo_matrix_view(M).num_cols,    2);
        ASSERT_EQUAL(cusp::make_coo_matrix_view(M).num_entries, 6);

        ASSERT_EQUAL_QUIET(cusp::make_coo_matrix_view(M).row_indices.begin(),    M.row_indices.begin());
        ASSERT_EQUAL_QUIET(cusp::make_coo_matrix_view(M).row_indices.end(),      M.row_indices.end());
        ASSERT_EQUAL_QUIET(cusp::make_coo_matrix_view(M).column_indices.begin(), M.column_indices.begin());
        ASSERT_EQUAL_QUIET(cusp::make_coo_matrix_view(M).column_indices.end(),   M.column_indices.end());
        ASSERT_EQUAL_QUIET(cusp::make_coo_matrix_view(M).values.begin(),         M.values.begin());
        ASSERT_EQUAL_QUIET(cusp::make_coo_matrix_view(M).values.end(),           M.values.end());
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestMakeCooMatrixView);

template <typename TestMatrix>
void TestToCooMatrixView(void)
{
    typedef typename TestMatrix::index_type   IndexType;
    typedef typename TestMatrix::value_type   ValueType;

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

    ASSERT_EQUAL(V.num_rows,              3);
    ASSERT_EQUAL(V.num_cols,              2);
    ASSERT_EQUAL(V.num_entries,           6);
    ASSERT_EQUAL(V.row_indices.size(),    6);
    ASSERT_EQUAL(V.column_indices.size(), 6);
    ASSERT_EQUAL(V.values.size(),         6);

    cusp::array1d<IndexType,cusp::host_memory> row_indices(V.row_indices);
    cusp::array1d<IndexType,cusp::host_memory> column_indices(V.column_indices);
    cusp::array1d<ValueType,cusp::host_memory> values(V.values);

    ASSERT_EQUAL(row_indices,    A.row_indices);
    ASSERT_EQUAL(column_indices, A.column_indices);
    ASSERT_EQUAL(values,         A.values);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestToCooMatrixView);

template <typename MemorySpace>
void TestCooToCooMatrixView(void)
{
    typedef int   IndexType;
    typedef float ValueType;

    typedef cusp::coo_matrix<IndexType,ValueType,MemorySpace> TestMatrix;
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

    V.row_indices[0] = -1;
    V.column_indices[0] = -1;
    V.values[0] = -1;

    ASSERT_EQUAL(M.row_indices[0],    -1);
    ASSERT_EQUAL(M.column_indices[0], -1);
    ASSERT_EQUAL(M.values[0],         -1);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooToCooMatrixView);

