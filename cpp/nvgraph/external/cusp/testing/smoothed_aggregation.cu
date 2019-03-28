#include <unittest/unittest.h>

#include <cusp/precond/aggregation/smoothed_aggregation.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>

#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>

template <typename SparseMatrix>
void TestSmoothedAggregation(void)
{
    typedef typename SparseMatrix::index_type   IndexType;
    typedef typename SparseMatrix::value_type   ValueType;
    typedef typename SparseMatrix::memory_space MemorySpace;

    // Create 2D Poisson problem
    SparseMatrix A;
    cusp::gallery::poisson5pt(A, 100, 100);

    // create smoothed aggregation solver
    cusp::precond::aggregation::smoothed_aggregation<IndexType,ValueType,MemorySpace> M(A);

    // test as standalone solver
    {
        cusp::array1d<ValueType,MemorySpace> b = unittest::random_samples<ValueType>(A.num_rows);
        cusp::array1d<ValueType,MemorySpace> x = unittest::random_samples<ValueType>(A.num_rows);

        // set stopping criteria (iteration_limit = 40, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 40, 1e-4);
        M.solve(b,x,monitor);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.8, true);
    }

    // test as preconditioner
    {
        cusp::array1d<ValueType,MemorySpace> b = unittest::random_samples<ValueType>(A.num_rows);
        cusp::array1d<ValueType,MemorySpace> x = unittest::random_samples<ValueType>(A.num_rows);

        // set stopping criteria (iteration_limit = 20, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 20, 1e-4);
        cusp::krylov::cg(A, x, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.5, true);
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSmoothedAggregation);

void TestSmoothedAggregationHostToDevice(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;

    // Create 2D Poisson problem
    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> A_h;
    cusp::gallery::poisson5pt(A_h, 100, 100);

    // create smoothed aggregation solver
    cusp::precond::aggregation::smoothed_aggregation<IndexType,ValueType,cusp::host_memory> M_h(A_h);
    cusp::precond::aggregation::smoothed_aggregation<IndexType,ValueType,cusp::device_memory> M_d(M_h);

    // test as standalone solver
    {
        cusp::array1d<ValueType,cusp::device_memory> b = unittest::random_samples<ValueType>(A_h.num_rows);
        cusp::array1d<ValueType,cusp::device_memory> x = unittest::random_samples<ValueType>(A_h.num_rows);

        // set stopping criteria (iteration_limit = 40, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 40, 1e-4);
        M_d.solve(b,x,monitor);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.8, true);
    }

    // test as preconditioner
    {
        cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> A_d(A_h);
        cusp::array1d<ValueType,cusp::device_memory> b = unittest::random_samples<ValueType>(A_d.num_rows);
        cusp::array1d<ValueType,cusp::device_memory> x = unittest::random_samples<ValueType>(A_d.num_rows);

        // set stopping criteria (iteration_limit = 20, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 20, 1e-4);
        cusp::krylov::cg(A_d, x, b, monitor, M_d);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.5, true);
    }
}
DECLARE_UNITTEST(TestSmoothedAggregationHostToDevice);

