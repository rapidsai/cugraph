#include <unittest/unittest.h>

#include <cusp/array1d.h>

#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename MemorySpace>
void TestArray1d(void)
{
    cusp::array1d<int, MemorySpace> a(4);

    ASSERT_EQUAL(a.size(), 4);

    a[0] = 0;
    a[1] = 1;
    a[2] = 2;
    a[3] = 3;

    a.push_back(4);

    ASSERT_EQUAL(a.size(), 5);

    ASSERT_EQUAL(a[0], 0);
    ASSERT_EQUAL(a[1], 1);
    ASSERT_EQUAL(a[2], 2);
    ASSERT_EQUAL(a[3], 3);
    ASSERT_EQUAL(a[4], 4);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1d)


template <typename MemorySpace>
void TestArray1dConstructor(void)
{
    cusp::array1d<int, MemorySpace> a(2);
    a[0] = 0;
    a[1] = 1;

    cusp::array1d<int, cusp::host_memory> h(a);
    ASSERT_EQUAL(h.size(), 2);
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);

    cusp::array1d<int, cusp::device_memory> d(a);
    ASSERT_EQUAL(d.size(), 2);
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);

    const cusp::array1d<int, cusp::host_memory> ch(2, 10);
    ASSERT_EQUAL(ch.size(), 2);
    ASSERT_EQUAL(ch[0], 10);
    ASSERT_EQUAL(ch[1], 10);

    const cusp::array1d<int, cusp::device_memory> cd(ch);
    ASSERT_EQUAL(cd.size(), 2);
    ASSERT_EQUAL(cd[0], 10);
    ASSERT_EQUAL(cd[1], 10);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dConstructor)


template <typename MemorySpace>
void TestArray1dInteroperability(void)
{
    typedef typename cusp::array1d<int, MemorySpace> Array;
    typedef typename std::vector<int>                Vector;
    typedef typename thrust::host_vector<int>        HostVector;
    typedef typename thrust::device_vector<int>      DeviceVector;

    // construct from std::vector
    {
        Vector v(2,10);

        Array a(v);
        ASSERT_EQUAL(a.size(), 2);
        ASSERT_EQUAL(a[0], 10);
        ASSERT_EQUAL(a[1], 10);

        Array b = v;
        ASSERT_EQUAL(b.size(), 2);
        ASSERT_EQUAL(b[0], 10);
        ASSERT_EQUAL(b[1], 10);

        Array c(v.begin(), v.end());
        ASSERT_EQUAL(c.size(), 2);
        ASSERT_EQUAL(c[0], 10);
        ASSERT_EQUAL(c[1], 10);
    }

    // construct from thrust::host_vector
    {
        HostVector v(2,10);

        Array a(v);
        ASSERT_EQUAL(a.size(), 2);
        ASSERT_EQUAL(a[0], 10);
        ASSERT_EQUAL(a[1], 10);

        Array b = v;
        ASSERT_EQUAL(b.size(), 2);
        ASSERT_EQUAL(b[0], 10);
        ASSERT_EQUAL(b[1], 10);

        Array c(v.begin(), v.end());
        ASSERT_EQUAL(c.size(), 2);
        ASSERT_EQUAL(c[0], 10);
        ASSERT_EQUAL(c[1], 10);
    }

    // construct from thrust::device_vector
    {
        DeviceVector v(2,10);

        Array a(v);
        ASSERT_EQUAL(a.size(), 2);
        ASSERT_EQUAL(a[0], 10);
        ASSERT_EQUAL(a[1], 10);

        Array b = v;
        ASSERT_EQUAL(b.size(), 2);
        ASSERT_EQUAL(b[0], 10);
        ASSERT_EQUAL(b[1], 10);

        Array c(v.begin(), v.end());
        ASSERT_EQUAL(c.size(), 2);
        ASSERT_EQUAL(c[0], 10);
        ASSERT_EQUAL(c[1], 10);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dInteroperability)


template <typename MemorySpace>
void TestArray1dAssignment(void)
{
    cusp::array1d<int, MemorySpace> a(2);
    a[0] = 0;
    a[1] = 1;

    cusp::array1d<int, cusp::host_memory> h = a;
    ASSERT_EQUAL(h.size(), 2);
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);

    cusp::array1d<int, cusp::device_memory> d = a;
    ASSERT_EQUAL(d.size(), 2);
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);

    const cusp::array1d<int, cusp::host_memory> ch(2, 10);
    a = ch;
    ASSERT_EQUAL(a.size(), 2);
    ASSERT_EQUAL(a[0], 10);
    ASSERT_EQUAL(a[1], 10);

    const cusp::array1d<int, cusp::device_memory> cd(2, 20);
    a = cd;
    ASSERT_EQUAL(a.size(), 2);
    ASSERT_EQUAL(a[0], 20);
    ASSERT_EQUAL(a[1], 20);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dAssignment)


template <typename MemorySpace>
void TestArray1dEquality(void)
{
    cusp::array1d<int, MemorySpace> A(2);
    A[0] = 10;
    A[1] = 20;

    cusp::array1d<int, cusp::host_memory>   h(A.begin(), A.end());
    cusp::array1d<int, cusp::device_memory> d(A.begin(), A.end());
    std::vector<int>                        v(2);
    v[0] = 10;
    v[1] = 20;

    ASSERT_EQUAL_QUIET(A, h);
    ASSERT_EQUAL_QUIET(A, d);
    ASSERT_EQUAL_QUIET(A, v);

    h.push_back(30);
    d.push_back(30);
    v.push_back(30);

    ASSERT_EQUAL_QUIET(A != h, true);
    ASSERT_EQUAL_QUIET(A != d, true);
    ASSERT_EQUAL_QUIET(A != v, true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dEquality)

