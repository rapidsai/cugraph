#include <unittest/unittest.h>

#include <cusp/graph/vertex_coloring.h>

#include <cusp/csr_matrix.h>

template <typename MatrixType, typename ArrayType>
size_t vertex_coloring(my_system& system, const MatrixType& G, ArrayType& colors)
{
    system.validate_dispatch();
    return 0;
}

void TestVertexColoringDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::array1d<int, cusp::device_memory> colors;

    my_system sys(0);

    // call with explicit dispatching
    cusp::graph::vertex_coloring(sys, A, colors);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestVertexColoringDispatch);

