#include <unittest/unittest.h>

#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>
#include <cusp/gallery/poisson.h>
#include <cusp/gallery/random.h>

template <typename TestMatrix>
void TestGeneralizedSpMV()
{
    typedef typename TestMatrix::index_type   IndexType;
    typedef typename TestMatrix::value_type   ValueType;
    typedef typename TestMatrix::memory_space MemorySpace;

    {
        // initialize example matrix
        cusp::array2d<ValueType, cusp::host_memory> A(5,4);
        A(0,0) = 13;
        A(0,1) = 80;
        A(0,2) =  0;
        A(0,3) =  0;
        A(1,0) =  0;
        A(1,1) = 27;
        A(1,2) =  0;
        A(1,3) =  0;
        A(2,0) = 55;
        A(2,1) =  0;
        A(2,2) = 24;
        A(2,3) = 42;
        A(3,0) =  0;
        A(3,1) = 69;
        A(3,2) =  0;
        A(3,3) = 83;
        A(4,0) =  0;
        A(4,1) =  0;
        A(4,2) = 27;
        A(4,3) =  0;

        // convert to desired format
        TestMatrix test_matrix = A;

        // allocate vectors
        cusp::array1d<ValueType, MemorySpace> x(4);
        cusp::array1d<ValueType, MemorySpace> y(5);
        cusp::array1d<ValueType, MemorySpace> z(5,-1);

        // initialize input and output vectors
        x[0] = 1.0f;
        y[0] = 10.0f;
        x[1] = 2.0f;
        y[1] = 20.0f;
        x[2] = 3.0f;
        y[2] = 30.0f;
        x[3] = 4.0f;
        y[3] = 40.0f;
        y[4] = 50.0f;

        generalized_spmv(test_matrix, x, y, z, thrust::multiplies<ValueType>(), thrust::plus<ValueType>());

        ASSERT_EQUAL(z[0], 183.0f);
        ASSERT_EQUAL(z[1],  74.0f);
        ASSERT_EQUAL(z[2], 325.0f);
        ASSERT_EQUAL(z[3], 510.0f);
        ASSERT_EQUAL(z[4], 131.0f);
    }

    typedef typename cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> HostMatrix;
    //cusp::array1d<HostMatrix, cusp::host_memory> matrices;
    std::vector<HostMatrix> matrices;

    {
        HostMatrix M;
        cusp::gallery::poisson5pt(M,   5,   5);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::poisson5pt(M,  10,  10);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::poisson5pt(M, 117, 113);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::poisson5pt(M, 313, 444);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::poisson5pt(M, 876, 321);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::random(M,  21,  23,   5);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::random(M,  45,  37,  15);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::random(M, 129, 127,  40);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::random(M, 355, 378, 234);
        matrices.push_back(M);
    }
    {
        HostMatrix M;
        cusp::gallery::random(M, 512, 512, 276);
        matrices.push_back(M);
    }

    for(size_t i = 0; i < matrices.size(); i++)
    {
        TestMatrix M = matrices[i];

        // allocate vectors
        cusp::array1d<ValueType, MemorySpace> x = unittest::random_integers<bool>(M.num_cols);
        cusp::array1d<ValueType, MemorySpace> y(M.num_rows,ValueType(0));
        cusp::array1d<ValueType, MemorySpace> z = unittest::random_integers<char>(M.num_rows);

        cusp::generalized_spmv(M, x, y, z, thrust::multiplies<ValueType>(), thrust::plus<ValueType>());

        // compute reference
        cusp::array1d<ValueType, MemorySpace> reference(M.num_rows,ValueType(0));
        cusp::multiply(M, x, reference);

        ASSERT_EQUAL(z, reference);
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestGeneralizedSpMV);

template <typename LinearOperator,
         typename Vector1,
         typename Vector2,
         typename Vector3,
         typename BinaryFunction1,
         typename BinaryFunction2>
void generalized_spmv(my_system &system,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce)
{
    system.validate_dispatch();
    return;
}

void TestGeneralizedSpMVDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::array1d<float, cusp::device_memory> x;

    my_system sys(0);

    // call with explicit dispatching
    cusp::generalized_spmv(sys, A, x, x, x, thrust::multiplies<float>(), thrust::plus<float>());

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestGeneralizedSpMVDispatch);

