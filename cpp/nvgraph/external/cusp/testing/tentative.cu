#include <unittest/unittest.h>

#include <cusp/precond/aggregation/tentative.h>
#include <cusp/precond/aggregation/detail/sa_view_traits.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>

template <typename MemorySpace>
void TestFitCandidates(void)
{
    typedef typename cusp::precond::aggregation::detail::select_sa_matrix_type<int,float,MemorySpace>::type SetupMatrixType;

    // 2 aggregates with 2 nodes each
    {
        cusp::array1d<int,MemorySpace> aggregates(4);
        aggregates[0] = 0;
        aggregates[1] = 0;
        aggregates[2] = 1;
        aggregates[3] = 1;
        cusp::array1d<float,MemorySpace> B(4);
        B[0] = 0.0f;
        B[1] = 1.0f;
        B[2] = 3.0f;
        B[3] = 4.0f;

        SetupMatrixType Q;
        cusp::array1d<float,MemorySpace> R(2);

        cusp::precond::aggregation::fit_candidates(aggregates, B, Q, R);

        ASSERT_EQUAL(R[0], 1.0f);
        ASSERT_EQUAL(R[1], 5.0f);
        ASSERT_ALMOST_EQUAL(Q.values[0], 0.0f);
        ASSERT_ALMOST_EQUAL(Q.values[1], 1.0f);
        ASSERT_ALMOST_EQUAL(Q.values[2], 0.6f);
        ASSERT_ALMOST_EQUAL(Q.values[3], 0.8f);
    }

    // 4 aggregates with varying numbers of nodes
    {
        cusp::array1d<int,MemorySpace> aggregates(10);
        aggregates[0] = 1;
        aggregates[1] = 2;
        aggregates[2] = 0;
        aggregates[3] = 3;
        aggregates[4] = 0;
        aggregates[5] = 2;
        aggregates[6] = 1;
        aggregates[7] = 2;
        aggregates[8] = 1;
        aggregates[9] = 1;
        cusp::array1d<float,MemorySpace> B(10,1.0f);

        SetupMatrixType Q;
        cusp::array1d<float,MemorySpace> R(4);

        cusp::precond::aggregation::fit_candidates(aggregates, B, Q, R);

        ASSERT_ALMOST_EQUAL(R[0], 1.41421f);
        ASSERT_ALMOST_EQUAL(R[1], 2.00000f);
        ASSERT_ALMOST_EQUAL(R[2], 1.73205f);
        ASSERT_ALMOST_EQUAL(R[3], 1.00000f);

        ASSERT_ALMOST_EQUAL(Q.values[0], 0.500000f);
        ASSERT_ALMOST_EQUAL(Q.values[1], 0.577350f);
        ASSERT_ALMOST_EQUAL(Q.values[2], 0.707107f);
        ASSERT_ALMOST_EQUAL(Q.values[3], 1.000000f);
        ASSERT_ALMOST_EQUAL(Q.values[4], 0.707107f);
        ASSERT_ALMOST_EQUAL(Q.values[5], 0.577350f);
        ASSERT_ALMOST_EQUAL(Q.values[6], 0.500000f);
        ASSERT_ALMOST_EQUAL(Q.values[7], 0.577350f);
        ASSERT_ALMOST_EQUAL(Q.values[8], 0.500000f);
        ASSERT_ALMOST_EQUAL(Q.values[9], 0.500000f);
    }

    // TODO test case w/ unaggregated nodes (marked w/ -1)
}
DECLARE_HOST_DEVICE_UNITTEST(TestFitCandidates);

