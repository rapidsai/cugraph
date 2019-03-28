#include <unittest/unittest.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/complex.h>
#include <cusp/blas/blas.h>

template <class MemorySpace>
void TestAmax(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  0.0f;
    x[1] = -5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  7.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::amax(x), 4);

    ASSERT_EQUAL(cusp::blas::amax(view_x), 4);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAmax)

template <class MemorySpace>
void TestComplexAmax(void)
{
    typedef cusp::complex<float> ValueType;
    typedef typename cusp::array1d<ValueType, MemorySpace>       Array;
    typedef typename cusp::array1d<ValueType, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] = ValueType( 7.0,  1.0);
    x[1] = ValueType(-5.0,  0.0);
    x[2] = ValueType( 4.0, -3.0);
    x[3] = ValueType(-3.0,  4.0);
    x[4] = ValueType( 0.0, -5.0);
    x[5] = ValueType( 1.0,  7.0);

    ASSERT_EQUAL(cusp::blas::amax(x), 0);

    ASSERT_EQUAL(cusp::blas::amax(view_x), 0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexAmax)

template <class MemorySpace>
void TestAxpy(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

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

    cusp::blas::axpy(x, y, 2.0f);

    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);

    View view_x(x);
    View view_y(y);

    cusp::blas::axpy(view_x, view_y, 2.0f);

    ASSERT_EQUAL(y[0],  28.0);
    ASSERT_EQUAL(y[1],  18.0);
    ASSERT_EQUAL(y[2],  16.0);
    ASSERT_EQUAL(y[3],  -7.0);

    // test size checking
    Array w(3);
    ASSERT_THROWS(cusp::blas::axpy(x, w, 1.0f), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpy)


template <class MemorySpace>
void TestAxpby(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    Array y(4);
    Array z(4,0);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;


    cusp::blas::axpby(x, y, z, 2.0f, 1.0f);

    ASSERT_EQUAL(z[0],  14.0);
    ASSERT_EQUAL(z[1],   8.0);
    ASSERT_EQUAL(z[2],   8.0);
    ASSERT_EQUAL(z[3],  -1.0);

    z[0] = 0.0f;
    z[1] = 0.0f;
    z[2] = 0.0f;
    z[3] = 0.0f;

    View view_x(x);
    View view_y(y);
    View view_z(z);

    cusp::blas::axpby(view_x, view_y, view_z, 2.0f, 1.0f);

    ASSERT_EQUAL(z[0],  14.0);
    ASSERT_EQUAL(z[1],   8.0);
    ASSERT_EQUAL(z[2],   8.0);
    ASSERT_EQUAL(z[3],  -1.0);

    // test size checking
    Array w(3);
    ASSERT_THROWS(cusp::blas::axpby(x, y, w, 2.0f, 1.0f), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpby)


template <class MemorySpace>
void TestAxpbypcz(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    Array y(4);
    Array z(4);
    Array w(4,0);

    x[0] =  7.0f;
    y[0] =  0.0f;
    z[0] =  1.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    z[1] =  0.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    z[2] =  3.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;
    z[3] = -2.0f;


    cusp::blas::axpbypcz(x, y, z, w, 2.0f, 1.0f, 3.0f);

    ASSERT_EQUAL(w[0],  17.0);
    ASSERT_EQUAL(w[1],   8.0);
    ASSERT_EQUAL(w[2],  17.0);
    ASSERT_EQUAL(w[3],  -7.0);

    w[0] = 0.0f;
    w[1] = 0.0f;
    w[2] = 0.0f;
    w[3] = 0.0f;

    View view_x(x);
    View view_y(y);
    View view_z(z);
    View view_w(w);

    cusp::blas::axpbypcz(view_x, view_y, view_z, view_w, 2.0f, 1.0f, 3.0f);

    ASSERT_EQUAL(w[0],  17.0);
    ASSERT_EQUAL(w[1],   8.0);
    ASSERT_EQUAL(w[2],  17.0);
    ASSERT_EQUAL(w[3],  -7.0);

    // test size checking
    Array output(3);
    ASSERT_THROWS(cusp::blas::axpbypcz(x, y, z, output, 2.0f, 1.0f, 3.0f), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpbypcz)


template <class MemorySpace>
void TestXmy(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    Array y(4);
    Array z(4,0);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;


    cusp::blas::xmy(x, y, z);

    ASSERT_EQUAL(z[0],   0.0f);
    ASSERT_EQUAL(z[1], -10.0f);
    ASSERT_EQUAL(z[2],   0.0f);
    ASSERT_EQUAL(z[3], -15.0f);

    z[0] = 0.0f;
    z[1] = 0.0f;
    z[2] = 0.0f;
    z[3] = 0.0f;

    View view_x(x);
    View view_y(y);
    View view_z(z);

    cusp::blas::xmy(view_x, view_y, view_z);

    ASSERT_EQUAL(z[0],   0.0f);
    ASSERT_EQUAL(z[1], -10.0f);
    ASSERT_EQUAL(z[2],   0.0f);
    ASSERT_EQUAL(z[3], -15.0f);

    // test size checking
    Array output(3);
    ASSERT_THROWS(cusp::blas::xmy(x, y, output), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestXmy)


template <class MemorySpace>
void TestCopy(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    {
        Array y(4, -1);
        cusp::blas::copy(x, y);
        ASSERT_EQUAL(x, y);
    }

    {
        Array y(4, -1);
        View view_y(y);
        cusp::blas::copy(view_x, view_y);
        ASSERT_EQUAL(x, y);
    }

    // test size checking
    cusp::array1d<float, MemorySpace> w(3);
    ASSERT_THROWS(cusp::blas::copy(w, x), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCopy)


template <class MemorySpace>
void TestDot(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

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

    ASSERT_EQUAL(cusp::blas::dot(x, y), -21.0f);

    ASSERT_EQUAL(cusp::blas::dot(View(x), View(y)), -21.0f);

    // test size checking
    cusp::array1d<float, MemorySpace> w(3);
    ASSERT_THROWS(cusp::blas::dot(x, w), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDot)


template <class MemorySpace>
void TestDotc(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);
    Array y(6);

    x[0] = cusp::complex<float>( 7.0f, 0.0f);
    y[0] = cusp::complex<float>( 0.0f, 0.0f);

    x[1] = cusp::complex<float>( 5.0f, 0.0f);
    y[1] = cusp::complex<float>(-2.0f, 0.0f);

    x[2] = cusp::complex<float>( 4.0f, 0.0f);
    y[2] = cusp::complex<float>( 0.0f, 0.0f);

    x[3] = cusp::complex<float>(-3.0f, 0.0f);
    y[3] = cusp::complex<float>( 5.0f, 0.0f);

    x[4] = cusp::complex<float>( 0.0f, 0.0f);
    y[4] = cusp::complex<float>( 6.0f, 0.0f);

    x[5] = cusp::complex<float>( 4.0f, 0.0f);
    y[5] = cusp::complex<float>( 1.0f, 0.0f);

    ASSERT_EQUAL(cusp::blas::dotc(x, y), -21.0f);

    ASSERT_EQUAL(cusp::blas::dotc(View(x), View(y)), -21.0f);

    // test size checking
    Array w(3);
    ASSERT_THROWS(cusp::blas::dotc(x, w), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDotc)


template <class MemorySpace>
void TestFill(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    cusp::blas::fill(x, 2.0f);

    ASSERT_EQUAL(x[0], 2.0);
    ASSERT_EQUAL(x[1], 2.0);
    ASSERT_EQUAL(x[2], 2.0);
    ASSERT_EQUAL(x[3], 2.0);

    cusp::blas::fill(view_x, 1.0f);

    ASSERT_EQUAL(x[0], 1.0);
    ASSERT_EQUAL(x[1], 1.0);
    ASSERT_EQUAL(x[2], 1.0);
    ASSERT_EQUAL(x[3], 1.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestFill)


template <class MemorySpace>
void TestNrm1(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm1(x), 20.0f);

    ASSERT_EQUAL(cusp::blas::nrm1(view_x), 20.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrm1)

template <class MemorySpace>
void TestComplexNrm1(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm1(x), 20.0f);

    ASSERT_EQUAL(cusp::blas::nrm1(view_x), 20.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexNrm1)


template <class MemorySpace>
void TestNrm2(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(x), 10.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(view_x), 10.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrm2)

template <class MemorySpace>
void TestComplexNrm2(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(x), 10.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(view_x), 10.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexNrm2)


template <class MemorySpace>
void TestNrmmax(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  0.0f;
    x[1] = -5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  7.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrmmax(x), 7.0f);

    ASSERT_EQUAL(cusp::blas::nrmmax(view_x), 7.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrmmax)

template <class MemorySpace>
void TestComplexNrmmax(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] = -5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrmmax(x), 7.0f);

    ASSERT_EQUAL(cusp::blas::nrmmax(view_x), 7.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexNrmmax)


template <class MemorySpace>
void TestScal(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array1d;
    typedef typename cusp::array1d<float, MemorySpace>::view View1d;

    typedef typename cusp::array2d<float, MemorySpace>       Array2d;
    typedef typename cusp::array2d<float, MemorySpace>::view View2d;

    Array1d x(6);
    View1d view_x(x);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas::scal(x, 4.0f);

    ASSERT_EQUAL(x[0],  28.0);
    ASSERT_EQUAL(x[1],  20.0);
    ASSERT_EQUAL(x[2],  16.0);
    ASSERT_EQUAL(x[3], -12.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  16.0);

    cusp::blas::scal(view_x, 2.0f);

    ASSERT_EQUAL(x[0],  56.0);
    ASSERT_EQUAL(x[1],  40.0);
    ASSERT_EQUAL(x[2],  32.0);
    ASSERT_EQUAL(x[3], -24.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  32.0);

    Array2d X(6,1);
    View2d view_X(X);

    X(0,0) =  7.0f;
    X(1,0) =  5.0f;
    X(2,0) =  4.0f;
    X(3,0) = -3.0f;
    X(4,0) =  0.0f;
    X(5,0) =  4.0f;

    cusp::blas::scal(X.column(0), 4.0f);

    ASSERT_EQUAL(X.column(0)[0],  28.0);
    ASSERT_EQUAL(X.column(0)[1],  20.0);
    ASSERT_EQUAL(X.column(0)[2],  16.0);
    ASSERT_EQUAL(X.column(0)[3], -12.0);
    ASSERT_EQUAL(X.column(0)[4],   0.0);
    ASSERT_EQUAL(X.column(0)[5],  16.0);

    cusp::blas::scal(view_X.column(0), 2.0f);

    ASSERT_EQUAL(X.column(0)[0],  56.0);
    ASSERT_EQUAL(X.column(0)[1],  40.0);
    ASSERT_EQUAL(X.column(0)[2],  32.0);
    ASSERT_EQUAL(X.column(0)[3], -24.0);
    ASSERT_EQUAL(X.column(0)[4],   0.0);
    ASSERT_EQUAL(X.column(0)[5],  32.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestScal)

template <class MemorySpace>
void TestGemv(void)
{
    typedef cusp::array2d<float, MemorySpace> Array2d;
    typedef cusp::array1d<float, MemorySpace> Array1d;

    Array2d A(6,6);
    Array1d x(6);

    ASSERT_THROWS(cusp::blas::gemv(A, x, x), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestGemv)

template <class MemorySpace>
void TestGer(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array2d;
    typedef typename cusp::array1d<float, MemorySpace> Array1d;

    Array2d A(6,6);
    Array1d x(6);

    ASSERT_THROWS(cusp::blas::ger(x, x, A), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestGer)

template <class MemorySpace>
void TestSymv(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array2d;
    typedef typename cusp::array1d<float, MemorySpace> Array1d;

    Array2d A(6,6);
    Array1d x(6);

    ASSERT_THROWS(cusp::blas::symv(A, x, x), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestSymv)

template <class MemorySpace>
void TestSyr(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array2d;
    typedef typename cusp::array1d<float, MemorySpace> Array1d;

    Array2d A(6,6);
    Array1d x(6);

    ASSERT_THROWS(cusp::blas::syr(x, A), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestSyr)

template <class MemorySpace>
void TestTrmv(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array2d;
    typedef typename cusp::array1d<float, MemorySpace> Array1d;

    Array2d A(6,6);
    Array1d x(6);

    ASSERT_THROWS(cusp::blas::trmv(A, x), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestTrmv)

template <class MemorySpace>
void TestTrsv(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array2d;
    typedef typename cusp::array1d<float, MemorySpace> Array1d;

    Array2d A(6,6);
    Array1d x(6);

    ASSERT_THROWS(cusp::blas::trsv(A, x), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestTrsv)

template <class MemorySpace>
void TestGemm(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array;

    Array A(6,6,1);
    Array B(6,6,0);

    cusp::blas::gemm(A, A, B);

    Array C(6,6,6);
    ASSERT_EQUAL(B.values, C.values);
}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template<>
void TestGemm<cusp::system::cuda::detail::par_t>(void)
{
    typedef typename cusp::array2d<float, cusp::device_memory> Array;

    Array A(6,6);

    ASSERT_THROWS(cusp::blas::gemm(A, A, A), cusp::not_implemented_exception);
}
#endif
DECLARE_HOST_DEVICE_UNITTEST(TestGemm)

template <class MemorySpace>
void TestSymm(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array;

    Array A(6,6);

    ASSERT_THROWS(cusp::blas::symm(A, A, A), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestSymm)

template <class MemorySpace>
void TestSyrk(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array;

    Array A(6,6);

    ASSERT_THROWS(cusp::blas::syrk(A, A), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestSyrk)

template <class MemorySpace>
void TestSyr2k(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array;

    Array A(6,6);

    ASSERT_THROWS(cusp::blas::syr2k(A, A, A), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestSyr2k)

template <class MemorySpace>
void TestTrmm(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array;

    Array A(6,6);

    ASSERT_THROWS(cusp::blas::trmm(A, A), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestTrmm)

template <class MemorySpace>
void TestTrsm(void)
{
    typedef typename cusp::array2d<float, MemorySpace> Array;

    Array A(6,6);

    ASSERT_THROWS(cusp::blas::trsm(A, A), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestTrsm)

template <typename Array>
int amax(my_system& system, const Array& x)
{
    system.validate_dispatch();
    return 0;
}

template <typename Array>
typename cusp::norm_type<typename Array::value_type>::type
asum(my_system& system, const Array& x)
{
    system.validate_dispatch();
    return 0;
}

template <typename Array1, typename Array2, typename ScalarType>
void axpy(my_system& system, const Array1& x, Array2& y, const ScalarType alpha)
{
    system.validate_dispatch();
    return;
}

template <typename Array1, typename Array2, typename Array3,
          typename ScalarType1, typename ScalarType2>
void axpby(my_system& system, const Array1& x, const Array2& y, Array3& output,
           ScalarType1 alpha, ScalarType2 beta)
{
    system.validate_dispatch();
    return;
}

template <typename Array1, typename Array2, typename Array3, typename Array4,
          typename ScalarType1, typename ScalarType2, typename ScalarType3>
void axpbypcz(my_system& system, const Array1& x, const Array2& y, const Array3& z, Array4& output,
              ScalarType1 alpha, ScalarType2 beta, ScalarType3 gamma)
{
    system.validate_dispatch();
    return;
}

template <typename Array1, typename Array2, typename Array3>
void xmy(my_system& system, const Array1& x, const Array2& y, Array3& output)
{
    system.validate_dispatch();
    return;
}

template <typename Array1, typename Array2>
void copy(my_system& system, const Array1& x, Array2& y)
{
    system.validate_dispatch();
    return;
}

template <typename Array1, typename Array2>
typename Array1::value_type
dot(my_system& system, const Array1& x, const Array2& y)
{
    system.validate_dispatch();
    return 0;
}

template <typename Array1, typename Array2>
typename Array1::value_type
dotc(my_system& system, const Array1& x, const Array2& y)
{
    system.validate_dispatch();
    return 0;
}

template <typename Array, typename ScalarType>
void fill(my_system& system, Array& array, const ScalarType alpha)
{
    system.validate_dispatch();
    return;
}

template <typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm1(my_system& system, const Array& array)
{
    system.validate_dispatch();
    return 0;
}

template <typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm2(my_system& system, const Array& array)
{
    system.validate_dispatch();
    return 0;
}

template <typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrmmax(my_system& system, const Array& array)
{
    system.validate_dispatch();
    return 0;
}

template <typename Array, typename ScalarType>
void scal(my_system& system, Array& x, const ScalarType alpha)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d1, typename Array1d1, typename Array1d2>
void gemv(my_system& system, const Array2d1& A, const Array1d1& x, Array1d2& y, float alpha = 1.0, float beta = 0.0)
{
    system.validate_dispatch();
    return;
}

template<typename Array1d1, typename Array1d2, typename Array2d1>
void ger(my_system& system, const Array1d1& x, const Array1d2& y, Array2d1& A, float alpha = 1.0)
{
    system.validate_dispatch();
    return;
}

template <typename Array2d1, typename Array1d1, typename Array1d2>
void symv(my_system& system, const Array2d1& A, const Array1d1& x, Array1d2& y, float alpha = 1.0, float beta = 0.0)
{
    system.validate_dispatch();
    return;
}

template <typename Array1d, typename Array2d>
void syr(my_system& system, const Array1d& x, Array2d& A, float alpha = 1.0)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d, typename Array1d>
void trmv(my_system& system, const Array2d& A, Array1d& x)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d, typename Array1d>
void trsv(my_system& system, const Array2d& A, Array1d& x)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d1, typename Array2d2, typename Array2d3>
void gemm(my_system& system, const Array2d1& A, const Array2d2& B, Array2d3& C, float alpha = 1.0, float beta = 0.0)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d1, typename Array2d2, typename Array2d3>
void symm(my_system& system, const Array2d1& A, const Array2d2& B, Array2d3& C, float alpha = 1.0, float beta = 0.0)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d1, typename Array2d2>
void syrk(my_system& system, const Array2d1& A, Array2d2& B, float alpha = 1.0, float beta = 0.0)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d1, typename Array2d2, typename Array2d3>
void syr2k(my_system& system, const Array2d1& A, const Array2d2& B, Array2d3& C, float alpha = 1.0, float beta = 0.0)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d1, typename Array2d2>
void trmm(my_system& system, const Array2d1& A, Array2d2& B, float alpha = 1.0)
{
    system.validate_dispatch();
    return;
}

template<typename Array2d1, typename Array2d2>
void trsm(my_system& system, const Array2d1& A, Array2d2& B, float alpha = 1.0)
{
    system.validate_dispatch();
    return;
}

void TestBlasDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::device_memory> A;
    cusp::array1d<float, cusp::device_memory> x;

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::amax(sys, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::asum(sys, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::axpy(sys, x, x, 1);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::axpby(sys, x, x, x, 1, 1);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::axpbypcz(sys, x, x, x, x, 1, 1, 1);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::xmy(sys, x, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::copy(sys, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::dot(sys, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::dotc(sys, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::fill(sys, x, 1);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::nrm1(sys, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::nrm2(sys, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::nrmmax(sys, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::scal(sys, x, 0);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::gemv(sys, A, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::ger(sys, x, x, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::symv(sys, A, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::syr(sys, x, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::trmv(sys, A, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::trsv(sys, A, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::gemm(sys, A, A, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::symm(sys, A, A, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::syrk(sys, A, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::syr2k(sys, A, A, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::trmm(sys, A, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::blas::trsm(sys, A, A);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }
}
DECLARE_UNITTEST(TestBlasDispatch);

