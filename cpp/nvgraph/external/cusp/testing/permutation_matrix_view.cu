#include <unittest/unittest.h>

#include <cusp/permutation_matrix.h>
#include <cusp/multiply.h>

template <typename MemorySpace>
void TestPermutationMatrixView(void)
{
    typedef unsigned int                                                  IndexType;
    typedef typename cusp::permutation_matrix<IndexType,MemorySpace>      Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::permutation_matrix_view<IndexView>             View;

    Matrix M(3);

    View V(3, cusp::make_array1d_view(M.permutation.begin(), M.permutation.end()));

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    3);
    ASSERT_EQUAL(V.num_entries, 3);

    ASSERT_EQUAL_QUIET(V.permutation.begin(),    M.permutation.begin());
    ASSERT_EQUAL_QUIET(V.permutation.end(),      M.permutation.end());

    View W(M);

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    3);
    ASSERT_EQUAL(W.num_entries, 3);

    ASSERT_EQUAL_QUIET(W.permutation.begin(),    M.permutation.begin());
    ASSERT_EQUAL_QUIET(W.permutation.end(),      M.permutation.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestPermutationMatrixView);


template <typename MemorySpace>
void TestPermutationMatrixViewAssignment(void)
{
    typedef unsigned int                                                  IndexType;
    typedef typename cusp::permutation_matrix<IndexType,MemorySpace>      Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::permutation_matrix_view<IndexView>             View;

    Matrix M(3);

    View V = M;

    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    3);
    ASSERT_EQUAL(V.num_entries, 3);

    ASSERT_EQUAL_QUIET(V.permutation.begin(),    M.permutation.begin());
    ASSERT_EQUAL_QUIET(V.permutation.end(),      M.permutation.end());

    View W = V;

    ASSERT_EQUAL(W.num_rows,    3);
    ASSERT_EQUAL(W.num_cols,    3);
    ASSERT_EQUAL(W.num_entries, 3);

    ASSERT_EQUAL_QUIET(W.permutation.begin(),    M.permutation.begin());
    ASSERT_EQUAL_QUIET(W.permutation.end(),      M.permutation.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestPermutationMatrixViewAssignment);


template <typename MemorySpace>
void TestMakePermutationMatrixView(void)
{
    typedef unsigned int                                                  IndexType;
    typedef typename cusp::permutation_matrix<IndexType,MemorySpace>      Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::permutation_matrix_view<IndexView> View;

    // construct view from parts
    {
        Matrix M(3);

        View V = cusp::make_permutation_matrix_view(3, cusp::make_array1d_view(M.permutation));

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    3);
        ASSERT_EQUAL(V.num_entries, 3);

        V.permutation[0] = 0;
        V.permutation[1] = 1;
        V.permutation[2] = 2;

        ASSERT_EQUAL_QUIET(V.permutation.begin(),    M.permutation.begin());
        ASSERT_EQUAL_QUIET(V.permutation.end(),      M.permutation.end());
    }

    // construct view from matrix
    {
        Matrix M(3);

        View V = cusp::make_permutation_matrix_view(M);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    3);
        ASSERT_EQUAL(V.num_entries, 3);

        V.permutation[0] = 0;
        V.permutation[1] = 1;
        V.permutation[2] = 2;

        ASSERT_EQUAL_QUIET(V.permutation.begin(),    M.permutation.begin());
        ASSERT_EQUAL_QUIET(V.permutation.end(),      M.permutation.end());
    }

    // construct view from view
    {
        Matrix M(3);

        View X = cusp::make_permutation_matrix_view(M);
        View V = cusp::make_permutation_matrix_view(X);

        ASSERT_EQUAL(V.num_rows,    3);
        ASSERT_EQUAL(V.num_cols,    3);
        ASSERT_EQUAL(V.num_entries, 3);

        V.permutation[0] = 0;
        V.permutation[1] = 1;
        V.permutation[2] = 2;

        ASSERT_EQUAL_QUIET(V.permutation.begin(),    M.permutation.begin());
        ASSERT_EQUAL_QUIET(V.permutation.end(),      M.permutation.end());
    }

    // construct view from const matrix
    {
        const Matrix M(3);

        ASSERT_EQUAL(cusp::make_permutation_matrix_view(M).num_rows,    3);
        ASSERT_EQUAL(cusp::make_permutation_matrix_view(M).num_cols,    3);
        ASSERT_EQUAL(cusp::make_permutation_matrix_view(M).num_entries, 3);

        ASSERT_EQUAL_QUIET(cusp::make_permutation_matrix_view(M).permutation.begin(),    M.permutation.begin());
        ASSERT_EQUAL_QUIET(cusp::make_permutation_matrix_view(M).permutation.end(),      M.permutation.end());
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestMakePermutationMatrixView);

