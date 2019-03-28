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

/*! \file cusp/iterator/random_iterator.h
 *  \brief An iterator which generates random entries
 */

#pragma once

#include <cusp/detail/config.h>

#include <thrust/functional.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cusp
{

/*! \cond */
namespace detail
{
// Forward definition
template<typename> struct random_functor_type;
template<typename,typename> struct random_integer_functor;
} // end detail
/*! \endcond */

/**
 *  \addtogroup iterators Iterators
 *  \ingroup utilities
 *  \{
 */

/**
 *  \brief Iterator for generating random values.
 *
 *  \tparam T The type used to encapsulate the underlying data.
 *
 *  \par Overview
 *  \p random_iterator is an iterator which represents a pointer into a range
 *  of random values. This iterator is useful for creating a range filled with random
 *  values without explicitly storing it in memory. Using \p random_iterator saves both
 *  memory capacity and bandwidth.
 *
 *  \par Example
 *  The following code snippet demonstrates how to create a \p random_iterator whose
 *  \c value_type is \c int and whose seed is \c 5.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/iterator/random_iterator.h>
 *
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    cusp::random_iterator<int> iter(5);
 *
 *    std::cout << iter[0] << std::endl;
 *    std::cout << iter[1] << std::endl;
 *    std::cout << iter[2] << std::endl;
 *  }
 *  \endcode
 */
template<typename T>
class random_iterator
{
public:

    /*! \cond */
    typedef T                                                                              value_type;
    typedef T*                                                                             pointer;
    typedef T&                                                                             reference;
    typedef size_t                                                                         difference_type;
    typedef size_t                                                                         size_type;
    typedef thrust::random_access_traversal_tag                                            iterator_category;

    typedef std::ptrdiff_t                                                                 IndexType;
    typedef detail::random_integer_functor<IndexType,T>                                    IndexFunctor;
    typedef typename thrust::counting_iterator<IndexType>                                  CountingIterator;
    typedef typename thrust::transform_iterator<IndexFunctor, CountingIterator, IndexType> RandomCountingIterator;

    // type of the random_range iterator
    typedef typename detail::random_functor_type<T>::type                                  Functor;
    typedef typename thrust::transform_iterator<Functor, RandomCountingIterator, T>        RandomTransformIterator;
    typedef RandomTransformIterator                                                        iterator;
    /*! \endcond */

    /*! \brief This constructor builds a \p random_iterator using a specified seed.
     *  \param seed The seed initial value used to generate the random sequence.
     */
    random_iterator(const size_t seed = 0)
        : random_counting_iterator(CountingIterator(0), IndexFunctor(seed)) {}

    /*! \brief This method returns an iterator pointing to the beginning of
     *  this random sequence of entries.
     *  \return mStart
     */
    iterator begin(void) const
    {
        return RandomTransformIterator(random_counting_iterator, Functor());
    }

    /*! \brief Subscript access to the data contained in this iterator.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read reference to data.
     *
     *  This operator allows for easy, array-style, data access.
     */
    value_type operator[](size_type n) const
    {
        return *(begin() + n);
    }

protected:

    /*! \cond */
    RandomCountingIterator random_counting_iterator;
    /*! \endcond */

}; // end random_iterator

/*! \} // end iterators
 */

} // end namespace cusp

#include <cusp/iterator/detail/random_iterator.inl>
