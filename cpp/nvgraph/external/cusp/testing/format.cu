#include <unittest/unittest.h>

#include <cusp/detail/format.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

typedef cusp::array1d<float, cusp::host_memory> A1D;
typedef cusp::array2d<float, cusp::host_memory> A2D;
typedef cusp::coo_matrix<int, float, cusp::host_memory> COO;
typedef cusp::csr_matrix<int, float, cusp::host_memory> CSR;
typedef cusp::dia_matrix<int, float, cusp::host_memory> DIA;
typedef cusp::ell_matrix<int, float, cusp::host_memory> ELL;
typedef cusp::hyb_matrix<int, float, cusp::host_memory> HYB;

void TestMatrixFormatArray1d(void)
{
    typedef A1D::format format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::array1d_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::sparse_format>::value), false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::dense_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::known_format>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatArray1d);

void TestMatrixFormatArray2d(void)
{
    typedef A2D::format format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::array2d_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::sparse_format>::value),false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::dense_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::known_format>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatArray2d);

void TestMatrixFormatCooMatrix(void)
{
    typedef COO::format format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::coo_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::sparse_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::dense_format>::value), false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::known_format>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatCooMatrix);

void TestMatrixFormatCsrMatrix(void)
{
    typedef CSR::format format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::csr_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::sparse_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::dense_format>::value), false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::known_format>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatCsrMatrix);

void TestMatrixFormatDiaMatrix(void)
{
    typedef DIA::format format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::dia_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::sparse_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::dense_format>::value), false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::known_format>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatDiaMatrix);

void TestMatrixFormatEllMatrix(void)
{
    typedef ELL::format format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::ell_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::sparse_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::dense_format>::value), false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::known_format>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatEllMatrix);

void TestMatrixFormatHybMatrix(void)
{
    typedef HYB::format format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::hyb_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::sparse_format>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::dense_format>::value), false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::known_format>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatHybMatrix);

