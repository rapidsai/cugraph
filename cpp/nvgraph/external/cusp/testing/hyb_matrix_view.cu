#include <unittest/unittest.h>

#include <cusp/hyb_matrix.h>

template <typename MemorySpace>
void TestHybMatrixView(void)
{
    typedef int                                                              IndexType;
    typedef float                                                            ValueType;
    typedef typename cusp::hyb_matrix<IndexType,ValueType,MemorySpace>       Matrix;
    typedef typename cusp::hyb_matrix<IndexType,ValueType,MemorySpace>::view View;

    Matrix M(3, 2, 3, 2, 1);

    {
        View V(cusp::make_ell_matrix_view(M.ell),
               cusp::make_coo_matrix_view(M.coo));

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 5);
        ASSERT_EQUAL(V.ell.num_rows,    3);
        ASSERT_EQUAL(V.ell.num_cols,    2);
        ASSERT_EQUAL(V.ell.num_entries, 3);
        ASSERT_EQUAL(V.coo.num_rows,    3);
        ASSERT_EQUAL(V.coo.num_cols,    2);
        ASSERT_EQUAL(V.coo.num_entries, 2);
        ASSERT_EQUAL_QUIET(V.ell.column_indices.values.begin(), M.ell.column_indices.values.begin());
        ASSERT_EQUAL_QUIET(V.ell.values.values.begin(),         M.ell.values.values.begin());
        ASSERT_EQUAL_QUIET(V.coo.row_indices.begin(),           M.coo.row_indices.begin());
        ASSERT_EQUAL_QUIET(V.coo.column_indices.begin(),        M.coo.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.coo.values.begin(),                M.coo.values.begin());
    }

    {
        View V(M);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    2);
        ASSERT_EQUAL(V.num_entries, 5);
        ASSERT_EQUAL(V.ell.num_rows,    3);
        ASSERT_EQUAL(V.ell.num_cols,    2);
        ASSERT_EQUAL(V.ell.num_entries, 3);
        ASSERT_EQUAL(V.coo.num_rows,    3);
        ASSERT_EQUAL(V.coo.num_cols,    2);
        ASSERT_EQUAL(V.coo.num_entries, 2);
        ASSERT_EQUAL_QUIET(V.ell.column_indices.values.begin(), M.ell.column_indices.values.begin());
        ASSERT_EQUAL_QUIET(V.ell.values.values.begin(),         M.ell.values.values.begin());
        ASSERT_EQUAL_QUIET(V.coo.row_indices.begin(),           M.coo.row_indices.begin());
        ASSERT_EQUAL_QUIET(V.coo.column_indices.begin(),        M.coo.column_indices.begin());
        ASSERT_EQUAL_QUIET(V.coo.values.begin(),                M.coo.values.begin());
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestHybMatrixView);

template <typename MemorySpace>
void TestHybToCooMatrixView(void)
{
    typedef int   IndexType;
    typedef float ValueType;

    typedef cusp::hyb_matrix<IndexType,ValueType,MemorySpace> TestMatrix;
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

    ASSERT_EQUAL(M.ell.column_indices(0,0), -1);
    ASSERT_EQUAL(M.ell.values(0,0),         -1);
}
/* DECLARE_HOST_DEVICE_UNITTEST(TestHybToCooMatrixView); */

