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
 * @file
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

#include <iterator>
#include <map>
#include <type_traits>

namespace xlib {

template<class iterator_t>
float average(iterator_t start, iterator_t end) noexcept;

template<class iterator_t>
float std_deviation(iterator_t start, iterator_t end) noexcept;

template<class iterator_t>
float gini_coefficient(iterator_t start, iterator_t end) noexcept;

/**
    \f$ G = \frac{{2\sum\limits_{i = 1}^V
    {\left( {i \cdot \deg \left( i \right)} \right)} }}{{V\sum\limits_{i = 1}^n
    {\deg \left( i \right)} }} - \frac{{V + 1}}{V} \f$
*/

/**
 *  \f$y = A{e^{Bx}}\f$
 *  \f$A = {e^a}\f$ \f$B = b\f$
 *  \f$a = \frac{{\sum\limits_{i = 0}^n {\left( {x_i^2{y_i}} \right)}
          \sum\limits_{i = 0}^n {\left( {{y_i}\ln {y_i}} \right)}
          - \sum\limits_{i = 0}^n {\left( {{x_i}{y_i}} \right)}
          \sum\limits_{i = 0}^n {\left( {{x_i}{y_i}\ln {y_i}} \right)} }}
          {{\sum\limits_{i = 0}^n {{y_i}} \sum\limits_{i = 0}^n
          {\left( {x_i^2{y_i}} \right)}  -
          {{\left( {\sum\limits_{i = 0}^n {\left( {{x_i}{y_i}} \right)} }
          \right)}^2}}}\f$
 *  \f$b = \frac{{\sum\limits_{i = 0}^n {{y_i}} \sum\limits_{i = 0}^n
          {\left( {{x_i}{y_i}\ln {y_i}} \right)}  - \sum\limits_{i = 0}^n
          {\left( {{x_i}{y_i}} \right)} \sum\limits_{i = 0}^n
          {\left( {{y_i}\ln {y_i}} \right)} }}{{\sum\limits_{i = 0}^n {{y_i}}
          \sum\limits_{i = 0}^n {\left( {x_i^2{y_i}} \right)}  -
          {{\left( {\sum\limits_{i = 0}^n {\left( {{x_i}{y_i}} \right)} }
          \right)}^2}}}\f$
 *//*
 template<class iterator_t>
std::pair<float, float>
exponentialFittingY(iterator_t start, iterator_t end,
                    typename std::iterator_traits<iterator_t>::value_type
                    ::first_type threshold = 0);

template<class iterator_t>
typename std::enable_if<
    std::is_arithmetic<
            typename std::iterator_traits<iterator_t>::value_type>::value,
    std::map<typename std::iterator_traits<iterator_t>::value_type,
             typename std::iterator_traits<iterator_t>::value_type>
>::type
convertToDistribution(iterator_t start, iterator_t end);

template<class iterator_t>
typename std::enable_if<std::is_arithmetic<
                typename std::iterator_traits<iterator_t>::value_type>::value,
double>::type
meanAbsoluteDifference(iterator_t start, iterator_t end);*/



} // namespace xlib

#include "impl/Statistics.i.hpp"
