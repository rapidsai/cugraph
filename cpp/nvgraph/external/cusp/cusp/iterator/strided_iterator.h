/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file cusp/iterator/strided_iterator.h
 *  \brief An iterator which returns strided access to array entries
 */

#pragma once

#include <cusp/detail/config.h>

#include <thrust/distance.h>
#include <thrust/functional.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cusp/functional.h>

namespace cusp
{

/*! \addtogroup iterators Iterators
 *  \brief Various customized Thrust based iterators
 *  \ingroup utilities
 *  \{
 */

/*! \brief RandomAccessIterator for strided access to array entries.
 *
 * \tparam RandomAccessIterator The iterator type used to encapsulate the underlying data.
 *
 * \par Overview
 * \p strided_iterator is an iterator which represents a pointer into
 *  a strided range entries in a underlying array. This iterator is useful
 *  for creating a strided sublist of entries from a larger iterator.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a \p strided_iterator whose
 *  \c value_type is \c int and whose values are gather from a \p counting_array.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/iterator/strided_iterator.h>
 *
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    typedef cusp::counting_array<int>::iterator Iterator;
 *
 *    cusp::counting_array<int> a(30);
 *    cusp::strided_iterator<Iterator> iter(a.begin(), a.end(), 5);
 *
 *    std::cout << iter[0] << std::endl;   // returns 0
 *    std::cout << iter[1] << std::endl;   // returns 5
 *    std::cout << iter[3] << std::endl;   // returns 15
 *
 *    return 0;
 *  }
 *  \endcode
 */
template <typename RandomAccessIterator>
class strided_iterator
{
public:

    /*! \cond */
    typedef typename thrust::iterator_value<RandomAccessIterator>::type                       value_type;
    typedef typename thrust::iterator_system<RandomAccessIterator>::type                      memory_space;
    typedef typename thrust::iterator_pointer<RandomAccessIterator>::type                     pointer;
    typedef typename thrust::iterator_reference<RandomAccessIterator>::type                   reference;
    typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  difference_type;
    typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  size_type;

    typedef cusp::multiplies_value<difference_type>                                           StrideFunctor;
    typedef typename thrust::counting_iterator<difference_type>                               CountingIterator;
    typedef typename thrust::transform_iterator<StrideFunctor, CountingIterator>              TransformIterator;
    typedef typename thrust::permutation_iterator<RandomAccessIterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;
    /*! \endcond */

    /*! \brief Null constructor initializes this \p strided_iterator's stride to zero.
     */
    strided_iterator(void)
        : stride(0) {}

    /*! \brief This constructor builds a \p strided_iterator from a range.
     *  \param first The beginning of the range.
     *  \param last The end of the range.
     *  \param stride The stride between consecutive entries in the iterator.
     */
    strided_iterator(RandomAccessIterator first, RandomAccessIterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    /*! \brief This method returns an iterator pointing to the beginning of
     *  this strided sequence of entries.
     *  \return mStart
     */
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), StrideFunctor(stride)));
    }

    /*! \brief This method returns an iterator pointing to one element past
     *  the last of this strided sequence of entries.
     *  \return mEnd
     */
    iterator end(void) const
    {
        return begin() + (thrust::distance(first,last) + (stride - 1)) / stride;
    }

    /*! \brief Subscript access to the data contained in this iterator.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read/write reference to data.
     *
     *  This operator allows for easy, array-style, data access.
     *  Note that data access with this operator is unchecked and
     *  out_of_range lookups are not defined.
     */
    reference operator[](size_type n) const
    {
        return *(begin() + n);
    }

protected:

    /*! \cond */
    RandomAccessIterator first;
    RandomAccessIterator last;
    difference_type stride;
    /*! \endcond */

}; // end strided_iterator

/*! \} // end iterators
 */

} // end namespace cusp

