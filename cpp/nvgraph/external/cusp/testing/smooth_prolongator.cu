#include <unittest/unittest.h>

#include <cusp/precond/aggregation/smooth_prolongator.h>
#include <cusp/precond/aggregation/detail/sa_view_traits.h>

#include <cusp/coo_matrix.h>

template <class MemorySpace>
void TestSmoothProlongator(void)
{
    typedef typename cusp::precond::aggregation::detail::select_sa_matrix_type<int,float,MemorySpace>::type SetupMatrixType;

    // simple example with diagonal S
    {
        cusp::coo_matrix<int,float,MemorySpace> _S(4,4,4);
        _S.row_indices[0] = 0;
        _S.column_indices[0] = 0;
        _S.values[0] = 1;
        _S.row_indices[1] = 1;
        _S.column_indices[1] = 1;
        _S.values[1] = 2;
        _S.row_indices[2] = 2;
        _S.column_indices[2] = 2;
        _S.values[2] = 3;
        _S.row_indices[3] = 3;
        _S.column_indices[3] = 3;
        _S.values[3] = 4;
        SetupMatrixType S(_S);

        cusp::coo_matrix<int,float,MemorySpace> _T(4,2,4);
        _T.row_indices[0] = 0;
        _T.column_indices[0] = 0;
        _T.values[0] = 0.5;
        _T.row_indices[1] = 1;
        _T.column_indices[1] = 0;
        _T.values[1] = 0.5;
        _T.row_indices[2] = 2;
        _T.column_indices[2] = 1;
        _T.values[2] = 0.5;
        _T.row_indices[3] = 3;
        _T.column_indices[3] = 1;
        _T.values[3] = 0.5;
        SetupMatrixType T(_T);

        SetupMatrixType _P;

        cusp::precond::aggregation::smooth_prolongator(S, T, _P, 2.0f, 4.0f);

        cusp::coo_matrix<int,float,MemorySpace> P(_P);

        ASSERT_EQUAL(P.num_rows,    4);
        ASSERT_EQUAL(P.num_cols,    2);
        ASSERT_EQUAL(P.num_entries, 4);

        ASSERT_EQUAL(P.row_indices[0], 0);
        ASSERT_EQUAL(P.column_indices[0], 0);
        ASSERT_EQUAL(P.row_indices[1], 1);
        ASSERT_EQUAL(P.column_indices[1], 0);
        ASSERT_EQUAL(P.row_indices[2], 2);
        ASSERT_EQUAL(P.column_indices[2], 1);
        ASSERT_EQUAL(P.row_indices[3], 3);
        ASSERT_EQUAL(P.column_indices[3], 1);

        ASSERT_EQUAL(P.values[0], -0.5);
        ASSERT_EQUAL(P.values[1], -0.5);
        ASSERT_EQUAL(P.values[2], -0.5);
        ASSERT_EQUAL(P.values[3], -0.5);
    }

    // 1D Poisson problem w/ 4 points and 2 aggregates
    {
        cusp::coo_matrix<int,float,MemorySpace> _S(4,4,10);
        _S.row_indices[0] = 0;
        _S.column_indices[0] = 0;
        _S.values[0] = 2;
        _S.row_indices[1] = 0;
        _S.column_indices[1] = 1;
        _S.values[1] =-1;
        _S.row_indices[2] = 1;
        _S.column_indices[2] = 0;
        _S.values[2] =-1;
        _S.row_indices[3] = 1;
        _S.column_indices[3] = 1;
        _S.values[3] = 2;
        _S.row_indices[4] = 1;
        _S.column_indices[4] = 2;
        _S.values[4] =-1;
        _S.row_indices[5] = 2;
        _S.column_indices[5] = 1;
        _S.values[5] =-1;
        _S.row_indices[6] = 2;
        _S.column_indices[6] = 2;
        _S.values[6] = 2;
        _S.row_indices[7] = 2;
        _S.column_indices[7] = 3;
        _S.values[7] =-1;
        _S.row_indices[8] = 3;
        _S.column_indices[8] = 2;
        _S.values[8] =-1;
        _S.row_indices[9] = 3;
        _S.column_indices[9] = 3;
        _S.values[9] = 2;
        SetupMatrixType S(_S);

        cusp::coo_matrix<int,float,MemorySpace> _T(4,2,4);
        _T.row_indices[0] = 0;
        _T.column_indices[0] = 0;
        _T.values[0] = 0.5;
        _T.row_indices[1] = 1;
        _T.column_indices[1] = 0;
        _T.values[1] = 0.5;
        _T.row_indices[2] = 2;
        _T.column_indices[2] = 1;
        _T.values[2] = 0.5;
        _T.row_indices[3] = 3;
        _T.column_indices[3] = 1;
        _T.values[3] = 0.5;
        SetupMatrixType T(_T);

        SetupMatrixType _P;

        cusp::precond::aggregation::smooth_prolongator(S, T, _P, 1.8090169943749472f, 4.0f/3.0f);

        cusp::coo_matrix<int,float,MemorySpace> P(_P);
        P.sort_by_row_and_column();

        ASSERT_EQUAL(P.num_rows,    4);
        ASSERT_EQUAL(P.num_cols,    2);
        ASSERT_EQUAL(P.num_entries, 6);

        ASSERT_EQUAL(P.row_indices[0], 0);
        ASSERT_EQUAL(P.column_indices[0], 0);
        ASSERT_EQUAL(P.row_indices[1], 1);
        ASSERT_EQUAL(P.column_indices[1], 0);
        ASSERT_EQUAL(P.row_indices[2], 1);
        ASSERT_EQUAL(P.column_indices[2], 1);
        ASSERT_EQUAL(P.row_indices[3], 2);
        ASSERT_EQUAL(P.column_indices[3], 0);
        ASSERT_EQUAL(P.row_indices[4], 2);
        ASSERT_EQUAL(P.column_indices[4], 1);
        ASSERT_EQUAL(P.row_indices[5], 3);
        ASSERT_EQUAL(P.column_indices[5], 1);

        ASSERT_ALMOST_EQUAL(P.values[0], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[1], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[2], 0.18426213f);
        ASSERT_ALMOST_EQUAL(P.values[3], 0.18426213f);
        ASSERT_ALMOST_EQUAL(P.values[4], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[5], 0.31573787f);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestSmoothProlongator);

