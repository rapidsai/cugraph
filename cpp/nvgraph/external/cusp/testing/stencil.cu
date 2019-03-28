#include <unittest/unittest.h>

#include <cusp/gallery/stencil.h>

void TestGenerateMatrixFromStencil1d(void)
{
    typedef int   IndexType;
    typedef float ValueType;
    typedef thrust::tuple<IndexType>              StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;

    stencil.push_back(StencilPoint(StencilIndex(-1), 1));
    stencil.push_back(StencilPoint(StencilIndex( 0), 2));
    stencil.push_back(StencilPoint(StencilIndex( 2), 3));

    cusp::dia_matrix<int, float, cusp::host_memory> matrix;
    cusp::gallery::generate_matrix_from_stencil(matrix,
            stencil,
            StencilIndex(4));

    // convert result to array2d
    cusp::array2d<float, cusp::host_memory> R(matrix);

    cusp::array2d<float, cusp::host_memory> E(4,4);

    E(0,0) = 2;
    E(0,1) = 0;
    E(0,2) = 3;
    E(0,3) = 0;
    E(1,0) = 1;
    E(1,1) = 2;
    E(1,2) = 0;
    E(1,3) = 3;
    E(2,0) = 0;
    E(2,1) = 1;
    E(2,2) = 2;
    E(2,3) = 0;
    E(3,0) = 0;
    E(3,1) = 0;
    E(3,2) = 1;
    E(3,3) = 2;

    ASSERT_EQUAL_QUIET(R, E);
}
DECLARE_UNITTEST(TestGenerateMatrixFromStencil1d);

void TestGenerateMatrixFromStencil2d(void)
{
    typedef int   IndexType;
    typedef float ValueType;
    typedef thrust::tuple<IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;

    stencil.push_back(StencilPoint(StencilIndex(-1, -1), 1));
    stencil.push_back(StencilPoint(StencilIndex(-1,  0), 2));
    stencil.push_back(StencilPoint(StencilIndex( 0,  0), 3));
    stencil.push_back(StencilPoint(StencilIndex( 1,  0), 4));
    stencil.push_back(StencilPoint(StencilIndex( 0,  2), 5));

    // grid is 2x3
    // [45]
    // [23]
    // [01]

    cusp::dia_matrix<int, float, cusp::host_memory> matrix;
    cusp::gallery::generate_matrix_from_stencil(matrix,
            stencil,
            StencilIndex(2,3));

    // convert result to array2d
    cusp::array2d<float, cusp::host_memory> R(matrix);

    cusp::array2d<float, cusp::host_memory> E(6,6);

    E(0,0) = 3;
    E(0,1) = 4;
    E(0,2) = 0;
    E(0,3) = 0;
    E(0,4) = 5;
    E(0,5) = 0;
    E(1,0) = 2;
    E(1,1) = 3;
    E(1,2) = 0;
    E(1,3) = 0;
    E(1,4) = 0;
    E(1,5) = 5;
    E(2,0) = 0;
    E(2,1) = 0;
    E(2,2) = 3;
    E(2,3) = 4;
    E(2,4) = 0;
    E(2,5) = 0;
    E(3,0) = 1;
    E(3,1) = 0;
    E(3,2) = 2;
    E(3,3) = 3;
    E(3,4) = 0;
    E(3,5) = 0;
    E(4,0) = 0;
    E(4,1) = 0;
    E(4,2) = 0;
    E(4,3) = 0;
    E(4,4) = 3;
    E(4,5) = 4;
    E(5,0) = 0;
    E(5,1) = 0;
    E(5,2) = 1;
    E(5,3) = 0;
    E(5,4) = 2;
    E(5,5) = 3;

    ASSERT_EQUAL_QUIET(R, E);
}
DECLARE_UNITTEST(TestGenerateMatrixFromStencil2d);

