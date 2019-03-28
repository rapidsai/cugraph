#include <unittest/unittest.h>

#include <cusp/sort.h>

template <class Array>
void InitializeSimpleKeySortTest(Array& unsorted_keys, Array& sorted_keys)
{
    unsorted_keys.resize(7);
    unsorted_keys[0] = 1;
    unsorted_keys[1] = 3;
    unsorted_keys[2] = 6;
    unsorted_keys[3] = 5;
    unsorted_keys[4] = 2;
    unsorted_keys[5] = 0;
    unsorted_keys[6] = 4;

    sorted_keys.resize(7);
    sorted_keys[0] = 0;
    sorted_keys[1] = 1;
    sorted_keys[2] = 2;
    sorted_keys[3] = 3;
    sorted_keys[4] = 4;
    sorted_keys[5] = 5;
    sorted_keys[6] = 6;
}

template <typename ArrayType>
void TestCountingSort(void)
{
    typedef typename ArrayType::template rebind<cusp::host_memory>::type HostArray;

    HostArray unsorted_keys;
    HostArray sorted_keys;

    InitializeSimpleKeySortTest(unsorted_keys, sorted_keys);

    ArrayType keys(unsorted_keys);
    ArrayType skeys(sorted_keys);

    cusp::counting_sort(keys, 0, 6);

    ASSERT_EQUAL(keys, skeys);
}
DECLARE_VECTOR_UNITTEST(TestCountingSort);

template <class Array>
void InitializeSimpleKeyValueSortTest(Array& unsorted_keys, Array& unsorted_values,
                                      Array& sorted_keys,   Array& sorted_values)
{
    unsorted_keys.resize(7);
    unsorted_values.resize(7);
    unsorted_keys[0] = 1;
    unsorted_values[0] = 0;
    unsorted_keys[1] = 3;
    unsorted_values[1] = 1;
    unsorted_keys[2] = 6;
    unsorted_values[2] = 2;
    unsorted_keys[3] = 5;
    unsorted_values[3] = 3;
    unsorted_keys[4] = 2;
    unsorted_values[4] = 4;
    unsorted_keys[5] = 0;
    unsorted_values[5] = 5;
    unsorted_keys[6] = 4;
    unsorted_values[6] = 6;

    sorted_keys.resize(7);
    sorted_values.resize(7);
    sorted_keys[0] = 0;
    sorted_values[1] = 0;
    sorted_keys[1] = 1;
    sorted_values[3] = 1;
    sorted_keys[2] = 2;
    sorted_values[6] = 2;
    sorted_keys[3] = 3;
    sorted_values[5] = 3;
    sorted_keys[4] = 4;
    sorted_values[2] = 4;
    sorted_keys[5] = 5;
    sorted_values[0] = 5;
    sorted_keys[6] = 6;
    sorted_values[4] = 6;
}

template <typename ArrayType>
void TestCountingSortByKey(void)
{
    typedef typename ArrayType::template rebind<cusp::host_memory>::type HostArray;

    HostArray unsorted_keys;
    HostArray unsorted_vals;
    HostArray sorted_keys;
    HostArray sorted_vals;

    InitializeSimpleKeyValueSortTest(unsorted_keys, unsorted_vals, sorted_keys, sorted_vals);

    ArrayType keys(unsorted_keys);
    ArrayType skeys(sorted_keys);
    ArrayType vals(unsorted_vals);
    ArrayType svals(sorted_vals);

    cusp::counting_sort_by_key(keys, vals, 0, 6);

    ASSERT_EQUAL(keys, skeys);
    ASSERT_EQUAL(vals, svals);
}
DECLARE_VECTOR_UNITTEST(TestCountingSortByKey);

