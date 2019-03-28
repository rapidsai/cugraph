#include <unittest/unittest.h>
#include <cusp/iterator/join_iterator.h>
#include <cusp/iterator/strided_iterator.h>

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

template <class Vector>
void TestJoinIterator(void)
{
    typedef typename Vector::value_type T;
    typedef thrust::counting_iterator<T> CountingIterator;
    typedef thrust::constant_iterator<T> ConstantIterator;
    typedef typename cusp::join_iterator< thrust::tuple<CountingIterator,ConstantIterator,CountingIterator> >::iterator JoinIterator;

    // construct join_iterator
    JoinIterator iter = cusp::make_join_iterator(5, 5, CountingIterator(0), ConstantIterator(9), CountingIterator(0));

    ASSERT_EQUAL(iter[0], 0);
    ASSERT_EQUAL(iter[1], 1);
    ASSERT_EQUAL(iter[4], 4);
    ASSERT_EQUAL(iter[5], 9);
    ASSERT_EQUAL(iter[8], 9);
    ASSERT_EQUAL(iter[9], 9);
}
DECLARE_VECTOR_UNITTEST(TestJoinIterator);

template <class Vector>
void TestStridedIterator(void)
{
    typedef typename Vector::value_type T;
    typedef thrust::counting_iterator<T> CountingIterator;
    typedef cusp::strided_iterator<CountingIterator> StridedIterator;

    // construct strided_iterator
    StridedIterator iter(CountingIterator(0), CountingIterator(20), 5);

    ASSERT_EQUAL(thrust::distance(iter.begin(), iter.end()), 4);
    ASSERT_EQUAL(iter[0],  0);
    ASSERT_EQUAL(iter[1],  5);
    ASSERT_EQUAL(iter[2], 10);
    ASSERT_EQUAL(iter[3], 15);
    ASSERT_EQUAL(iter[4], 20);
}
DECLARE_VECTOR_UNITTEST(TestStridedIterator);

