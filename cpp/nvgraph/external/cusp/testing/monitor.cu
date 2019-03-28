#include <unittest/unittest.h>

#include <cusp/monitor.h>

template <typename MemorySpace>
void TestMonitorSimple(void)
{
    cusp::array1d<float,MemorySpace> b(2);
    b[0] = 10;
    b[1] =  0;

    cusp::array1d<float,MemorySpace> r(2);
    r[0] = 10;
    r[1] =  0;

    cusp::monitor<float> monitor(b, 5, 0.5, 1.0);

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 0);
    ASSERT_EQUAL(monitor.iteration_limit(), 5);
    ASSERT_EQUAL(monitor.relative_tolerance(), 0.5);
    ASSERT_EQUAL(monitor.absolute_tolerance(), 1.0);
    ASSERT_EQUAL(monitor.tolerance(),          6.0);

    ++monitor;

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 1);
    ASSERT_EQUAL(monitor.residual_norm(), 10.0);

    r[0] = 2;

    ASSERT_EQUAL(monitor.finished(r), true);
    ASSERT_EQUAL(monitor.iteration_count(), 1);
    ASSERT_EQUAL(monitor.residual_norm(), 2.0);

    r[0] = 7;

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 1);
    ASSERT_EQUAL(monitor.residual_norm(), 7.0);

    ++monitor;

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 2);
    ASSERT_EQUAL(monitor.residual_norm(), 7.0);

    ++monitor;
    ++monitor;

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 4);
    ASSERT_EQUAL(monitor.residual_norm(), 7.0);

    ++monitor;

    ASSERT_EQUAL(monitor.finished(r), true);
    ASSERT_EQUAL(monitor.iteration_count(), 5);
    ASSERT_EQUAL(monitor.residual_norm(), 7.0);

    monitor.reset(r);

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 0);
    ASSERT_EQUAL(monitor.residual_norm(), 7.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMonitorSimple);

