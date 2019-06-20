/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include <algorithm>    //std::accumulate
#include <cmath>        //std::sqrt

namespace xlib {

template<class iterator_t>
float average(iterator_t start, iterator_t end) noexcept {
    static_assert(std::is_arithmetic<
                  typename std::iterator_traits<iterator_t>::value_type>::value,
                  "wrong type");
    using R = typename std::iterator_traits<iterator_t>::value_type;
    using T = typename std::conditional<std::is_integral<R>::value,
                                        int64_t, double>;

    const T sum = std::accumulate(start, end, 0);
    return static_cast<float>(sum) / std::distance(start, end);
}

template<class iterator_t>
float std_deviation(iterator_t start, iterator_t end) noexcept {
    static_assert(std::is_arithmetic<
                  typename std::iterator_traits<iterator_t>::value_type>::value,
                  "wrong type");
    using R = typename std::iterator_traits<iterator_t>::value_type;
    using T = typename std::conditional<std::is_integral<R>::value,
                                        int64_t, double>::type;
    T sum = 0, sum2 = 0;
    for (auto it = start; it < end; it++) {
        sum  += *it;
        sum2 += (*it) * (*it);
    }
    auto num_items = std::distance(start, end);
    return std::sqrt(static_cast<float>(num_items * sum2 - sum * sum))
                     / num_items;
}

template<class iterator_t>
float gini_coefficient(iterator_t start, iterator_t end) noexcept {
    using R = typename std::iterator_traits<iterator_t>::value_type;
    using T = typename std::conditional<std::is_integral<R>::value,
                                        int64_t, double>::type;
    auto size = std::distance(start, end);

    auto tmp = new R[size];
    std::copy(start, end, tmp);
    std::sort(tmp, tmp + size);

    T numerator = 0, denom = 0;
    for (auto i = 0; i < size; i++) {
        numerator += static_cast<T>(i + 1) * tmp[i];
        denom     += tmp[i];
    }
    delete[] tmp;
    float result = static_cast<float>(static_cast<double>(2 * numerator) /
                                      static_cast<double>(size * denom) -
                (static_cast<double>(size + 1) / static_cast<double>(size)));

    return result < 0.0f ? 0.0f : result;
}

/*
template<class iterator_t>
std::pair<float, float>
exponentialFittingY(iterator_t start, iterator_t end,
                    typename std::iterator_traits<iterator_t>::value_type
                    ::first_type threshold) {

    using T = typename std::iterator_traits<iterator_t>::value_type::first_type;
    using T2 = typename std::remove_const<typename Type64<T>::type>::type;

    T2 sigma_x2_y = 0, sigma_x_y = 0, sigma_y = 0;
    double sigma_y_lny = 0.0, sigma_x_y_lny = 0.0;

    int i = 0;
    for (auto it = start; it != end; it++) {
        const T y = (*it).first;//Y[i];
        if (y <= threshold) continue;
        const T x = ++i;
        double y_lny = y * std::log(y);
        T2 x_y =  x * y;

        sigma_x2_y += x * x_y;
        sigma_y_lny += y_lny;
        sigma_x_y += x_y;
        sigma_x_y_lny += x * y_lny;
        sigma_y += y;
    }
    double denom = static_cast<double>(
                    sigma_y * sigma_x2_y - sigma_x_y * sigma_x_y);
    float a = static_cast<float>(static_cast<double>(
                sigma_x2_y * sigma_y_lny - sigma_x_y * sigma_x_y_lny) / denom);
    float b = static_cast<float>(static_cast<double>(
                sigma_y * sigma_x_y_lny - sigma_x_y * sigma_y_lny) / denom);
    return std::pair<float, float>(std::exp(a), b);
}

template<class iterator_t>
typename std::enable_if<
    std::is_arithmetic<
            typename std::iterator_traits<iterator_t>::value_type>::value,
    std::map<typename std::iterator_traits<iterator_t>::value_type,
             typename std::iterator_traits<iterator_t>::value_type>
>::type
convertToDistribution(iterator_t start, iterator_t end) {
    using T = typename std::iterator_traits<iterator_t>::value_type;
    std::map<T, T> distribution_map;
    for (auto it1 = start; it1 != end; it1++) {
        //if (*it1 == 0) continue;
        auto it_map = distribution_map.insert(std::pair<T, T>(*it1, 1));
        if (!it_map.second)
            (*it_map.first).second++;
    }
    return distribution_map;
}


template<class iterator_t>
typename std::enable_if<std::is_arithmetic<
                typename std::iterator_traits<iterator_t>::value_type>::value,
double>::type
meanAbsoluteDifference(iterator_t start, iterator_t end) {
    using T = typename Type64_it<iterator_t>::type;
    T mean_abs_difference_sum = 0;
    for (auto it1 = start; it1 != end; it1++) {
        for (auto it2 = start; it2 != end; it2++)
            mean_abs_difference_sum += std::abs(*it1 - *it2);
    }
    return static_cast<double>(mean_abs_difference_sum) /
           (std::distance(start, end) * std::distance(start, end));
}*/
//mean_abs_difference / avg;

//The Gini coefficient takes values between zero and one, with zero denoting
//total equality between degrees, and one denoting the dominance of a single node.
//REQUIRE SORTED VALUES


} // namespace xlib
