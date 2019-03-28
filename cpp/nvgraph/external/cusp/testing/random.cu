#include <unittest/unittest.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>

#include <limits>

// #include <cusp/print.h>

template <typename T>
struct TestRandomIntegersDistribution
{
    void operator()(void)
    {
        size_t n = 123456;
        cusp::random_array<T> random(n);

        cusp::array2d<size_t, cusp::host_memory> counts(2 * sizeof(T), 16, 0);

        for (size_t i = 0; i < n; i++)
        {
            unsigned long long raw = random[i] - std::numeric_limits<T>::min();
            for (size_t nibble = 0; nibble < 2 * sizeof(T); nibble++)
            {
                counts(nibble, (raw >> (4 * nibble)) % 16)++;
            }
        }

        // std::cout << "min " << *thrust::min_element(counts.values.begin(), counts.values.end()) << std::endl;
        // std::cout << "max " << *thrust::max_element(counts.values.begin(), counts.values.end()) << std::endl;
        // cusp::print_matrix(counts);

        size_t expected = n / 16;
        size_t min_bin = *thrust::min_element(counts.values.begin(), counts.values.end());
        size_t max_bin = *thrust::max_element(counts.values.begin(), counts.values.end());

        ASSERT_GEQUAL(min_bin, (size_t) (0.95 * expected));
        ASSERT_LEQUAL(max_bin, (size_t) (1.05 * expected));
    }
};
SimpleUnitTest<TestRandomIntegersDistribution, IntegralTypes> TestRandomIntegersDistributionInstance;


template <typename T>
struct TestRandomRealsDistribution
{
    void operator()(void)
    {
        size_t n = 123456;
        cusp::random_array<T> random(n);

        cusp::array1d<size_t, cusp::host_memory> buckets(32, 0);

        for (size_t i = 0; i < n; i++)
        {
            const T val = random[i];
            ASSERT_EQUAL(T(0) <= val, true);
            ASSERT_EQUAL(val < T(1), true);

            buckets[ size_t(val * T(buckets.size())) ]++;
        }

        // std::cout << "min " << *thrust::min_element(buckets.begin(), buckets.end()) << std::endl;
        // std::cout << "max " << *thrust::max_element(buckets.begin(), buckets.end()) << std::endl;
        // cusp::print_matrix(buckets);

        size_t expected = n / buckets.size();
        size_t min_bin = *thrust::min_element(buckets.begin(), buckets.end());
        size_t max_bin = *thrust::max_element(buckets.begin(), buckets.end());

        ASSERT_GEQUAL(min_bin, (size_t) (0.95 * expected));
        ASSERT_LEQUAL(max_bin, (size_t) (1.05 * expected));
    }
};
SimpleUnitTest<TestRandomRealsDistribution, FloatingPointTypes> TestRandomRealsDistributionInstance;


template <typename T>
struct TestRandomIntegers
{
    void operator()(void)
    {
        size_t n = 123456;
        cusp::random_array<T> random(n);

        cusp::array1d<T, cusp::host_memory>   h(random);
        cusp::array1d<T, cusp::device_memory> d(random);

        ASSERT_EQUAL(h, d);
    }
};
SimpleUnitTest<TestRandomIntegers, IntegralTypes> TestRandomIntegersInstance;


template <typename T>
struct TestRandomReals
{
    void operator()(void)
    {
        size_t n = 123456;
        cusp::random_array<T> random(n);

        cusp::array1d<T, cusp::host_memory>   h(random);
        cusp::array1d<T, cusp::device_memory> d(random);

        ASSERT_ALMOST_EQUAL(h, d);
    }
};
SimpleUnitTest<TestRandomReals, unittest::type_list<float, double, cusp::complex<float>, cusp::complex<double> > >  TestRandomRealsInstance;

