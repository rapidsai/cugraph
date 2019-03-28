#include <unittest/unittest.h>

#include <cusp/array2d.h>
#include <cusp/blas/blas.h>
#include <cusp/gallery/poisson.h>

#include <cusp/system/cuda/detail/cublas/blas.h>

template<typename ValueType>
void TestCublasAmax(void)
{
    typedef cusp::array1d<ValueType, cusp::device_memory>  Array;
    typedef typename Array::view                           View;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] = -5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::amax(cusp::cuda::par.with(handle),x), 0);

    ASSERT_EQUAL(cusp::blas::amax(cusp::cuda::par.with(handle),view_x), 0);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasAmax);

template<typename ValueType>
void TestCublasAsum(void)
{
    typedef cusp::array1d<ValueType, cusp::device_memory>  Array;
    typedef typename Array::view                           View;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::asum(cusp::cuda::par.with(handle),x), 20.0f);

    ASSERT_EQUAL(cusp::blas::asum(cusp::cuda::par.with(handle),view_x), 20.0f);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasAsum);

template<typename ValueType>
void TestCublasAxpy(void)
{
    typedef cusp::array1d<ValueType, cusp::device_memory>  Array;
    typedef typename Array::view                           View;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array x(4);
    Array y(4);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;

    cusp::blas::axpy(cusp::cuda::par.with(handle), x, y, 2.0f);

    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);

    View view_x(x);
    View view_y(y);

    cusp::blas::axpy(cusp::cuda::par.with(handle), view_x, view_y, 2.0f);

    ASSERT_EQUAL(y[0],  28.0);
    ASSERT_EQUAL(y[1],  18.0);
    ASSERT_EQUAL(y[2],  16.0);
    ASSERT_EQUAL(y[3],  -7.0);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasAxpy);

template<typename ValueType>
void TestCublasCopy(void)
{
    typedef cusp::array1d<ValueType, cusp::device_memory>  Array;
    typedef typename Array::view                           View;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array x(4);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    {
        Array y(4, -1);
        cusp::blas::copy(cusp::cuda::par.with(handle), x, y);
        ASSERT_EQUAL(x==y, true);
    }

    {
        Array y(4, -1);
        View view_x(x);
        View view_y(y);
        cusp::blas::copy(cusp::cuda::par.with(handle), view_x, view_y);
        ASSERT_EQUAL(x==y, true);
    }

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasCopy);

template<typename ValueType>
void TestCublasDot(void)
{
    typedef cusp::array1d<ValueType, cusp::device_memory>  Array;
    typedef typename Array::view                           View;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array x(6);
    Array y(6);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;
    x[4] =  0.0f;
    y[4] =  6.0f;
    x[5] =  4.0f;
    y[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::dot(cusp::cuda::par.with(handle), x, y), -21.0f);

    View view_x(x);
    View view_y(y);
    ASSERT_EQUAL(cusp::blas::dot(cusp::cuda::par.with(handle), view_x, view_y), -21.0f);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_REAL_UNITTEST(TestCublasDot);

template<typename ValueType>
void TestCublasNrm2(void)
{
    typedef cusp::array1d<ValueType, cusp::device_memory>  Array;
    typedef typename Array::view                           View;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(cusp::cuda::par.with(handle), x), 10.0f);
    ASSERT_EQUAL(cusp::blas::nrm2(cusp::cuda::par.with(handle), View(x)), 10.0f);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasNrm2);

template<typename ValueType>
void TestCublasScal(void)
{
    typedef cusp::array1d<ValueType, cusp::device_memory>      Array;
    typedef typename Array::view                               View;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas::scal(cusp::cuda::par.with(handle), x, 4.0f);

    ASSERT_EQUAL(x[0],  28.0);
    ASSERT_EQUAL(x[1],  20.0);
    ASSERT_EQUAL(x[2],  16.0);
    ASSERT_EQUAL(x[3], -12.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  16.0);

    View v(x);
    cusp::blas::scal(cusp::cuda::par.with(handle), v, 2.0f);

    ASSERT_EQUAL(x[0],  56.0);
    ASSERT_EQUAL(x[1],  40.0);
    ASSERT_EQUAL(x[2],  32.0);
    ASSERT_EQUAL(x[3], -24.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  32.0);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasScal);

template<typename ValueType>
void TestCublasGemv(void)
{
    typedef cusp::array2d<ValueType, cusp::device_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::device_memory> Array1d;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array2d A;
    Array1d x(9);
    Array1d y(9);

    cusp::gallery::poisson5pt(A, 3, 3);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;
    x[6] = -3.0f;
    x[7] =  0.0f;
    x[8] =  4.0f;

    cusp::blas::gemv(cusp::cuda::par.with(handle), A, x, y);

    ASSERT_EQUAL(y[0],  26.0);
    ASSERT_EQUAL(y[1],   9.0);
    ASSERT_EQUAL(y[2],   7.0);
    ASSERT_EQUAL(y[3], -16.0);
    ASSERT_EQUAL(y[4],  -6.0);
    ASSERT_EQUAL(y[5],   8.0);
    ASSERT_EQUAL(y[6],  -9.0);
    ASSERT_EQUAL(y[7],  -1.0);
    ASSERT_EQUAL(y[8],  12.0);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasGemv);

template<typename ValueType>
void TestCublasSymv(void)
{
    typedef cusp::array2d<ValueType, cusp::device_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::device_memory> Array1d;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array2d A;
    Array1d x(9);
    Array1d y(9);

    cusp::gallery::poisson5pt(A, 3, 3);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;
    x[6] = -3.0f;
    x[7] =  0.0f;
    x[8] =  4.0f;

    cusp::blas::symv(cusp::cuda::par.with(handle), A, x, y);

    ASSERT_EQUAL(y[0],  26.0);
    ASSERT_EQUAL(y[1],   9.0);
    ASSERT_EQUAL(y[2],   7.0);
    ASSERT_EQUAL(y[3], -16.0);
    ASSERT_EQUAL(y[4],  -6.0);
    ASSERT_EQUAL(y[5],   8.0);
    ASSERT_EQUAL(y[6],  -9.0);
    ASSERT_EQUAL(y[7],  -1.0);
    ASSERT_EQUAL(y[8],  12.0);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_REAL_UNITTEST(TestCublasSymv);

template<typename ValueType>
void TestCublasTrmv(void)
{
    typedef cusp::array2d<ValueType, cusp::device_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::device_memory> Array1d;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array2d A;
    Array1d x(9);
    Array1d expected(9);

    cusp::gallery::poisson5pt(A, 3, 3);

    // set lower diagonal entries to zero
    for(int j = 0; j < 9; j++)
      for(int i = j + 1; i < 9; i++)
          A(i,j) = ValueType(0);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;
    x[6] = -3.0f;
    x[7] =  0.0f;
    x[8] =  4.0f;

    cusp::blas::gemv(cusp::cuda::par.with(handle), A, x, expected);
    cusp::blas::trmv(cusp::cuda::par.with(handle), A, x);

    /* ASSERT_ALMOST_EQUAL(x, expected); */

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }

    KNOWN_FAILURE;
}
DECLARE_NUMERIC_UNITTEST(TestCublasTrmv);

template<typename ValueType>
void TestCublasTrsv(void)
{
    typedef cusp::array2d<ValueType, cusp::device_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::device_memory> Array1d;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array2d A;
    Array1d x(9);

    cusp::gallery::poisson5pt(A, 3, 3);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;
    x[6] = -3.0f;
    x[7] =  0.0f;
    x[8] =  4.0f;

    Array1d b(x);

    cusp::blas::trsv(cusp::cuda::par.with(handle), A, x);

    // check residual norm
    cusp::array1d<ValueType, cusp::device_memory> residual(x);
    cusp::blas::trmv(cusp::cuda::par.with(handle), A, residual);
    cusp::blas::axpby(residual, b, residual, -1.0f, 1.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(residual) < 1e-7, true);

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}
DECLARE_NUMERIC_UNITTEST(TestCublasTrsv);

template<typename ValueType, typename Orientation>
void TestCublasGemmOrientation(void)
{
    typedef cusp::array2d<ValueType, cusp::device_memory, Orientation> Array2dDev;
    typedef typename Array2dDev::rebind<cusp::host_memory>::type       Array2dHost;

    cublasHandle_t handle;

    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasCreate failed");
    }

    Array2dDev A(3, 4);
    Array2dDev B(4, 3);

    cusp::counting_array<ValueType> init_values(A.num_entries, 1);
    A.values = init_values;
    B.values = init_values;

    Array2dHost A_h(A);
    Array2dHost B_h(B);

    {
      Array2dDev C(A.num_rows, B.num_cols);
      cusp::blas::gemm(cusp::cuda::par.with(handle), A, B, C);

      Array2dHost C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A_h, B_h, C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }

    {
      Array2dDev C(A.T().num_rows, B.T().num_cols);
      cusp::blas::gemm(cusp::cuda::par.with(handle), A.T(), B.T(), C);

      Array2dHost C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A_h.T(), B_h.T(), C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }

    {
      Array2dDev C(A.T().num_rows, A.num_cols);
      cusp::blas::gemm(cusp::cuda::par.with(handle), A.T(), A, C);

      Array2dHost C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A_h.T(), A_h, C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }

    {
      Array2dDev C(A.num_rows, A.T().num_cols);
      cusp::blas::gemm(cusp::cuda::par.with(handle), A, A.T(), C);

      Array2dHost C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A_h, A_h.T(), C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }

    if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
    {
      throw cusp::runtime_exception("cublasDestroy failed");
    }
}

template<typename ValueType>
void TestCublasGemm(void)
{
    TestCublasGemmOrientation<ValueType,cusp::row_major>();
    TestCublasGemmOrientation<ValueType,cusp::column_major>();
}
DECLARE_REAL_UNITTEST(TestCublasGemm);

