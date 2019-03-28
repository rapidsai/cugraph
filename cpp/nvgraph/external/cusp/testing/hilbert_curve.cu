#include <unittest/unittest.h>

#include <cusp/graph/hilbert_curve.h>

#include <cusp/array2d.h>

template <typename Array2dType, typename ArrayType>
void hilbert_curve(my_system& system, const Array2dType& coord, const size_t num_parts, ArrayType& parts)
{
    system.validate_dispatch();
    return;
}

void TestHilbertCurveDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::device_memory> coords;
    cusp::array1d<int, cusp::device_memory> parts;

    my_system sys(0);

    // call with explicit dispatching
    cusp::graph::hilbert_curve(sys, coords, 0, parts);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestHilbertCurveDispatch);

