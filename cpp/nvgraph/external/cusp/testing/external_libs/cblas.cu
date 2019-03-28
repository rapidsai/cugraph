#include <unittest/unittest.h>

#include <cusp/array2d.h>
#include <cusp/blas/blas.h>
#include <cusp/gallery/poisson.h>

#include <cusp/system/cpp/detail/cblas/blas.h>

template<typename ValueType>
void TestCblasAmax(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] = -5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::amax(cusp::cblas::par,x), 0);

    ASSERT_EQUAL(cusp::blas::amax(cusp::cblas::par,view_x), 0);
}
DECLARE_NUMERIC_UNITTEST(TestCblasAmax);

template<typename ValueType>
void TestCblasAsum(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::asum(cusp::cblas::par,x), 20.0f);

    ASSERT_EQUAL(cusp::blas::asum(cusp::cblas::par,view_x), 20.0f);
}
DECLARE_NUMERIC_UNITTEST(TestCblasAsum);

template<typename ValueType>
void TestCblasAxpy(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

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

    cusp::blas::axpy(cusp::cblas::par, x, y, 2.0f);

    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);

    View view_x(x);
    View view_y(y);

    cusp::blas::axpy(cusp::cblas::par, view_x, view_y, 2.0f);

    ASSERT_EQUAL(y[0],  28.0);
    ASSERT_EQUAL(y[1],  18.0);
    ASSERT_EQUAL(y[2],  16.0);
    ASSERT_EQUAL(y[3],  -7.0);
}
DECLARE_NUMERIC_UNITTEST(TestCblasAxpy);

template<typename ValueType>
void TestCblasCopy(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

    Array x(4);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    {
        Array y(4, -1);
        cusp::blas::copy(cusp::cblas::par, x, y);
        ASSERT_EQUAL(x == y, true);
    }

    {
        Array y(4, -1);
        View view_x(x);
        View view_y(y);
        cusp::blas::copy(cusp::cblas::par, view_x, view_y);
        ASSERT_EQUAL(view_x == view_y, true);
    }
}
DECLARE_NUMERIC_UNITTEST(TestCblasCopy);

template<typename ValueType>
void TestCblasDot(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

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

    ASSERT_EQUAL(cusp::blas::dot(cusp::cblas::par, x, y), -21.0f);

    View view_x(x);
    View view_y(y);
    ASSERT_EQUAL(cusp::blas::dot(cusp::cblas::par, view_x, view_y), -21.0f);
}
DECLARE_REAL_UNITTEST(TestCblasDot);

template<typename ValueType>
void TestCblasDotc(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

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

    ASSERT_EQUAL(cusp::blas::dotc(cusp::cblas::par, x, y), -21.0);

    View view_x(x);
    View view_y(y);
    ASSERT_EQUAL(cusp::blas::dotc(cusp::cblas::par, view_x, view_y), -21.0);
}
DECLARE_COMPLEX_UNITTEST(TestCblasDotc);

template<typename ValueType>
void TestCblasNrm2(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(cusp::cblas::par, x), 10.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(cusp::cblas::par, View(x)), 10.0f);
}
DECLARE_NUMERIC_UNITTEST(TestCblasNrm2);

template<typename ValueType>
void TestCblasScal(void)
{
    typedef cusp::array1d<ValueType, cusp::host_memory>       Array;
    typedef typename Array::view                              View;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas::scal(cusp::cblas::par, x, 4.0f);

    ASSERT_EQUAL(x[0],  28.0);
    ASSERT_EQUAL(x[1],  20.0);
    ASSERT_EQUAL(x[2],  16.0);
    ASSERT_EQUAL(x[3], -12.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  16.0);

    View v(x);
    cusp::blas::scal(cusp::cblas::par, v, 2.0f);

    ASSERT_EQUAL(x[0],  56.0);
    ASSERT_EQUAL(x[1],  40.0);
    ASSERT_EQUAL(x[2],  32.0);
    ASSERT_EQUAL(x[3], -24.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  32.0);
}
DECLARE_NUMERIC_UNITTEST(TestCblasScal);

template<typename ValueType>
void TestCblasGemv(void)
{
    typedef cusp::array2d<ValueType, cusp::host_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::host_memory> Array1d;

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

    cusp::blas::gemv(cusp::cblas::par, A, x, y);

    ASSERT_EQUAL(y[0],  26.0);
    ASSERT_EQUAL(y[1],   9.0);
    ASSERT_EQUAL(y[2],   7.0);
    ASSERT_EQUAL(y[3], -16.0);
    ASSERT_EQUAL(y[4],  -6.0);
    ASSERT_EQUAL(y[5],   8.0);
    ASSERT_EQUAL(y[6],  -9.0);
    ASSERT_EQUAL(y[7],  -1.0);
    ASSERT_EQUAL(y[8],  12.0);
}
DECLARE_NUMERIC_UNITTEST(TestCblasGemv);

template<typename ValueType>
void TestCblasSymv(void)
{
    typedef cusp::array2d<ValueType, cusp::host_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::host_memory> Array1d;

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

    cusp::blas::symv(cusp::cblas::par, A, x, y);

    ASSERT_EQUAL(y[0],  26.0);
    ASSERT_EQUAL(y[1],   9.0);
    ASSERT_EQUAL(y[2],   7.0);
    ASSERT_EQUAL(y[3], -16.0);
    ASSERT_EQUAL(y[4],  -6.0);
    ASSERT_EQUAL(y[5],   8.0);
    ASSERT_EQUAL(y[6],  -9.0);
    ASSERT_EQUAL(y[7],  -1.0);
    ASSERT_EQUAL(y[8],  12.0);
}
DECLARE_REAL_UNITTEST(TestCblasSymv);

template<typename ValueType>
void TestCblasTrmv(void)
{
    typedef cusp::array2d<ValueType, cusp::host_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::host_memory> Array1d;

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

    cusp::blas::gemv(cusp::cblas::par, A, x, expected);
    cusp::blas::trmv(cusp::cblas::par, A, x);

    ASSERT_ALMOST_EQUAL(x, expected);
}
DECLARE_NUMERIC_UNITTEST(TestCblasTrmv);

template<typename ValueType>
void TestCblasTrsv(void)
{
    typedef cusp::array2d<ValueType, cusp::host_memory> Array2d;
    typedef cusp::array1d<ValueType, cusp::host_memory> Array1d;

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

    cusp::blas::trsv(cusp::cblas::par, A, x);

    // check residual norm
    cusp::array1d<ValueType, cusp::host_memory> residual(x);
    cusp::blas::trmv(cusp::cblas::par, A, residual);
    cusp::blas::axpby(residual, b, residual, -1.0f, 1.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(residual) < 1e-7, true);
}
DECLARE_NUMERIC_UNITTEST(TestCblasTrsv);

template<typename ValueType, typename Orientation>
void TestCblasGemmOrientation(void)
{
    typedef cusp::array2d<ValueType, cusp::host_memory, Orientation> Array2d;

    Array2d A(3, 4);
    Array2d B(4, 3);

    cusp::counting_array<ValueType> init_values(A.num_entries, 1);
    A.values = init_values;
    B.values = init_values;

    {
      Array2d C(A.num_rows, B.num_cols);
      cusp::blas::gemm(cusp::cblas::par, A, B, C);

      Array2d C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A, B, C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }

    {
      Array2d C(A.T().num_rows, B.T().num_cols);
      cusp::blas::gemm(cusp::cblas::par, A.T(), B.T(), C);

      Array2d C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A.T(), B.T(), C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }

    {
      Array2d C(A.T().num_rows, A.num_cols);
      cusp::blas::gemm(cusp::cblas::par, A.T(), A, C);

      Array2d C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A.T(), A, C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }

    {
      Array2d C(A.num_rows, A.T().num_cols);
      cusp::blas::gemm(cusp::cblas::par, A, A.T(), C);

      Array2d C_h(C.num_rows, C.num_cols);
      cusp::blas::gemm(A, A.T(), C_h);
      ASSERT_EQUAL(C_h.values, C.values);
    }
}

template<typename ValueType>
void TestCblasGemm(void)
{
    TestCblasGemmOrientation<ValueType,cusp::row_major>();
    TestCblasGemmOrientation<ValueType,cusp::column_major>();
}
DECLARE_REAL_UNITTEST(TestCblasGemm);

