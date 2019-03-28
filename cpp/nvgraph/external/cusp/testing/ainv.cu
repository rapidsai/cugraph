#include <unittest/unittest.h>

#include <cusp/precond/ainv.h>

#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>

inline
__host__ __device__
unsigned int hash32(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a <<  5);
    a = (a + 0xd3a2646c) ^ (a <<  9);
    a = (a + 0xfd7046c5) + (a <<  3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

struct hash_01
{
    __host__ __device__
    float operator()(const unsigned int& index) const
    {
        return (float)(hash32(index)) / ((float)0xffffffff);
    }
};

void TestAINVHeap(void)
{
    hash_01 rng;
    int seed = 100;

    for (int trial = 0; trial < 10000; trial ++) {
        cusp::precond::detail::ainv_matrix_row<int, float> row;

        int size =100;
        int i;
        for (i=0; i < size; i++)
            row.insert(i, rng(seed++));

        for (i=0; i < size; i++)
            row.add_to_value(i, rng(seed++) - .5);

        for (i=0; i < size/2; i++)
            row.remove_min();

        for (i=0; i < size; i++)
            row.insert(i+size, rng(seed++));

        ASSERT_EQUAL(row.validate_heap(), true);
        ASSERT_EQUAL(row.validate_backpointers(), true);

    }
}
DECLARE_UNITTEST(TestAINVHeap);


void TestAINVFactorization(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // Create 2D Poisson problem
    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A;
    cusp::gallery::poisson5pt(A, 10, 10);
    A.values[0] = 10;
    int N = A.num_rows;

    // factor exactly
    cusp::precond::scaled_bridson_ainv<ValueType,MemorySpace> M(A, 0, -1);



    cusp::array1d<ValueType,MemorySpace> x(N);
    cusp::array1d<ValueType,MemorySpace> b(N, 0);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      x.begin(),
                      hash_01());

    cusp::array1d<ValueType,MemorySpace> x_solve = x;

    // cg should converge in 1 iteration
    // because we're in single precision, this isn't exact, but 1e-5 tolerance should give enough leeway.
    cusp::monitor<ValueType> monitor(b, 1, 0, 1e-5);
    cusp::krylov::cg(A, x_solve, b, monitor, M);

    ASSERT_EQUAL(monitor.converged(), true);
}
DECLARE_UNITTEST(TestAINVFactorization);

void TestAINVSymmetry(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // Create 2D Poisson problem
    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A;
    cusp::gallery::poisson5pt(A, 100, 100);
    A.values[0] = 10;
    int N = A.num_rows;

    cusp::array1d<ValueType,MemorySpace> x(N);
    cusp::array1d<ValueType,MemorySpace> b(N, 0);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      x.begin(),
                      hash_01());

    ValueType nrm1, nrm2;

    // test symmetric version
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::bridson_ainv<ValueType,MemorySpace> M(A, .1);

        cusp::monitor<ValueType> monitor(b, 125, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        nrm1 = monitor.residual_norm();
        ASSERT_EQUAL(monitor.converged(), true);
    }

    // test non-symmetric version
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::nonsym_bridson_ainv<ValueType,MemorySpace> M(A, .1);

        cusp::monitor<ValueType> monitor(b, 125, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        nrm2 = monitor.residual_norm();
        ASSERT_EQUAL(monitor.converged(), true);
    }

    ASSERT_EQUAL(nrm1, nrm2);

    // assert they returned identical results
}
DECLARE_UNITTEST(TestAINVSymmetry);

void TestAINVConvergence(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // Create 2D Poisson problem
    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A;
    cusp::gallery::poisson5pt(A, 100, 100);
    A.values[0] = 10;
    int N = A.num_rows;

    cusp::array1d<ValueType,MemorySpace> x(N);
    cusp::array1d<ValueType,MemorySpace> b(N, 0);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      x.begin(),
                      hash_01());



    // test drop tolerance strategy
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::scaled_bridson_ainv<ValueType,MemorySpace> M(A, .1);

        cusp::monitor<ValueType> monitor(b, 125, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }

    // test sparsity strategy
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::scaled_bridson_ainv<ValueType,MemorySpace> M(A, 0, 10);

        cusp::monitor<ValueType> monitor(b, 70, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }

    // test both
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::scaled_bridson_ainv<ValueType,MemorySpace> M(A, .01, 4);

        cusp::monitor<ValueType> monitor(b, 120, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }
    // test lin dropping
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::scaled_bridson_ainv<ValueType,MemorySpace> M(A, 0, -1, true, 4);

        cusp::monitor<ValueType> monitor(b, 120, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }


    // test drop tolerance strategy
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::bridson_ainv<ValueType,MemorySpace> M(A, .1);

        cusp::monitor<ValueType> monitor(b, 125, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }

    // test sparsity strategy
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::bridson_ainv<ValueType,MemorySpace> M(A, 0, 10);

        cusp::monitor<ValueType> monitor(b, 70, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }

    // test both
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::bridson_ainv<ValueType,MemorySpace> M(A, .01, 4);

        cusp::monitor<ValueType> monitor(b, 120, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }

    // test lin dropping
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::bridson_ainv<ValueType,MemorySpace> M(A, 0, -1, true, 4);

        cusp::monitor<ValueType> monitor(b, 120, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }

}
DECLARE_UNITTEST(TestAINVConvergence);

