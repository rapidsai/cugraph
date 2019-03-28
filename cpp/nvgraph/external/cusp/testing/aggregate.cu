#include <unittest/unittest.h>

#include <cusp/precond/aggregation/aggregate.h>
#include <cusp/precond/aggregation/detail/sa_view_traits.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>

template <class MemorySpace>
void TestStandardAggregate(void)
{
    // TODO make this test something, possibly disjoint things that must aggregate

    typedef typename cusp::precond::aggregation::detail::select_sa_matrix_type<int,float,MemorySpace>::type SetupMatrixType;

    SetupMatrixType A;
    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<int,MemorySpace> aggregates(A.num_rows);
    cusp::precond::aggregation::standard_aggregate(A, aggregates);
}
DECLARE_HOST_DEVICE_UNITTEST(TestStandardAggregate);

template <class MemorySpace>
void TestMISAggregate(void)
{
    // TODO make this test something, possibly disjoint things that must aggregate

    typedef typename cusp::precond::aggregation::detail::select_sa_matrix_type<int,float,MemorySpace>::type SetupMatrixType;

    SetupMatrixType A;
    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<int,MemorySpace> aggregates(A.num_rows);
    cusp::precond::aggregation::mis_aggregate(A, aggregates);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMISAggregate);

