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

/*! \file cusp/iterator/join_iterator.h
 *  \brief An iterator which concatenates two separate iterators.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/memory.h>

#include <thrust/distance.h>
#include <thrust/functional.h>

#include <thrust/iterator/transform_iterator.h>

namespace cusp
{

/*! \cond */
template <int size, typename T>
struct constant_tuple
{
    typedef thrust::detail::identity_<T>                 T_;
    typedef thrust::detail::identity_<thrust::null_type> N_;

    typedef
    thrust::tuple<typename thrust::detail::eval_if<(size > 0),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 1),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 2),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 3),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 4),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 5),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 6),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 7),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 8),T_,N_>::type,
                  typename thrust::detail::eval_if<(size > 9),T_,N_>::type> type;
};

template<typename T, typename V, int SIZE>
struct join_search
{
    template<typename SizesTuple, typename Tuple>
    __host__ __device__
    V operator()(const SizesTuple &t1, const Tuple& t2, const T i) const
    {
        return (i >= T(thrust::get<SIZE-2>(t1))) ? V(thrust::get<SIZE-1>(t2)[i]) : join_search<T,V,SIZE-1>()(t1,t2,i);
    }
};

template<typename T, typename V>
struct join_search<T,V,2>
{
    template<typename SizesTuple, typename Tuple>
    __host__ __device__
    V operator()(const SizesTuple &t1, const Tuple& t2, const T i) const
    {
        return i >= T(thrust::get<0>(t1)) ? thrust::get<1>(t2)[i] : thrust::get<0>(t2)[i];
    }
};
/*! \endcond */


/*! \addtogroup iterators Iterators
 *  \ingroup utilities
 *  \{
 */

/*! \brief RandomAccessIterator for access to array entries from two
 * concatenated iterators.
 *
 * \tparam Iterator1 The iterator type used to encapsulate the first set of
 * entries.
 * \tparam Iterator2 The iterator type used to encapsulate the second set of
 * entries.
 * \tparam Iterator3 The iterator type used to order concatenated entries
 * from two separate iterators.
 *
 * \par Overview
 * \p join_iterator is an iterator which represents a pointer into
 *  a concatenated range of entries from two underlying arrays. This iterator
 *  is useful for creating a single range of permuted entries from two
 *  different iterators.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a \p join_iterator whose
 *  \c value_type is \c int and whose values are gather from a \p counting_iterator
 *  and a \p constant_iterator.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/iterator/join_iterator.h>
 *
 *  #include <thrust/sequence.h>
 *
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    typedef cusp::counting_array<int>                                              CountingArray;
 *    typedef cusp::constant_array<int>                                              ConstantArray;
 *    typedef typename CountingArray::iterator                                       CountingIterator;
 *    typedef typename ConstantArray::iterator                                       ConstantIterator;
 *    typedef cusp::array1d<int,cusp::device_memory>::iterator                       ArrayIterator;
 *    typedef cusp::join_iterator<CountingIterator,ConstantIterator,ArrayIterator>   JoinIterator;
 *
 *    // a = [0, 1, 2, 3]
 *    CountingArray a(4);
 *    // b = [10, 10, 10, 10, 10]
 *    ConstantArray b(5, 10);
 *    cusp::array1d<int,cusp::device_memory> indices(a.size() + b.size());
 *    // set indices to a sequence for simple in order access
 *    thrust::sequence(indices.begin(), indices.end());
 *    // iter = [0, 1, 2, 3, 10, 10, 10, 10, 10]
 *    JoinIterator iter(a.begin(), a.end(), b.begin(), b.end(), indices.begin());
 *
 *    std::cout << iter[0] << std::endl;   // returns 0
 *    std::cout << iter[3] << std::endl;   // returns 3
 *    std::cout << iter[4] << std::endl;   // returns 10
 *
 *    return 0;
 *  }
 *  \endcode
 */
template <typename Tuple>
class join_iterator
{
public:

    /*! \cond */
    typedef typename thrust::tuple_element<0,Tuple>::type          Iterator1;
    typedef typename thrust::iterator_value<Iterator1>::type       value_type;
    typedef typename thrust::iterator_pointer<Iterator1>::type     pointer;
    typedef typename thrust::iterator_reference<Iterator1>::type   reference;
    typedef typename thrust::iterator_difference<Iterator1>::type  difference_type;
    typedef typename thrust::iterator_difference<Iterator1>::type  size_type;
    typedef typename thrust::iterator_system<Iterator1>::type      space;
    typedef typename cusp::iterator_system<space>::type            memory_space;

    const static size_t tuple_size = thrust::tuple_size<Tuple>::value;

    // forward definition
    struct join_select_functor;

    typedef typename constant_tuple<tuple_size-1,size_t>::type            SizesTuple;
    typedef typename thrust::tuple_element<tuple_size-1,Tuple>::type      IndexIterator;
    typedef thrust::transform_iterator<join_select_functor,IndexIterator> TransformIterator;

    struct join_select_functor : public thrust::unary_function<difference_type,value_type>
    {
        SizesTuple t1;
        Tuple t2;

        __host__ __device__
        join_select_functor(void) {}

        __host__ __device__
        join_select_functor(const SizesTuple& t1, const Tuple& t2)
            : t1(t1), t2(t2) {}

        __host__ __device__
        value_type operator()(const difference_type& i)
        {
            return join_search<difference_type,value_type,tuple_size-1>()(t1,t2,i);
        }
    };
    /*! \endcond */

    // type of the join_iterator
    typedef TransformIterator iterator;

    /*! \brief This constructor builds a \p join_iterator from two iterators.
     *  \param first_begin The beginning of the first range.
     *  \param first_end The end of the first range.
     *  \param second_begin The beginning of the second range.
     *  \param second_end The end of the second range.
     *  \param indices_begin The permutation indices used to order entries
     *  from the two joined iterators.
     */
    join_iterator(const SizesTuple& t1, const Tuple& t2) : t1(t1), t2(t2) {}

    /*! \brief This method returns an iterator pointing to the beginning of
     *  this joined sequence of permuted entries.
     *  \return mStart
     */
    iterator begin(void) const
    {
        return TransformIterator(thrust::get<tuple_size-1>(t2), join_select_functor(t1,t2));
    }

    /*! \brief This method returns an iterator pointing to one element past
     *  the last of this joined sequence of permuted entries.
     *  \return mEnd
     */
    iterator end(void) const
    {
        return begin() + thrust::get<tuple_size-2>(t1);
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
    const SizesTuple& t1;
    const Tuple& t2;
    /*! \endcond */
};

template <typename T1, typename T2, typename T3>
typename join_iterator< thrust::tuple<T1,T2,T3> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const T1& t1, const T2& t2, const T3& t3)
{
    typedef thrust::tuple<T1,T2,T3>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2),
                                thrust::make_tuple(t1, t2-s1, t3)).begin();
}

template <typename T1, typename T2, typename T3, typename T4>
typename join_iterator< thrust::tuple<T1,T2,T3,T4> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4)
{
    typedef thrust::tuple<T1,T2,T3,T4>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3),
                                thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4)).begin();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
typename join_iterator< thrust::tuple<T1,T2,T3,T4,T5> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5)
{
    typedef thrust::tuple<T1,T2,T3,T4,T5>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3, s1+s2+s3+s4),
                                thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4-s1-s2-s3, t5)).begin();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
typename join_iterator< thrust::tuple<T1,T2,T3,T4,T5,T6> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6)
{
    typedef thrust::tuple<T1,T2,T3,T4,T5,T6>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3, s1+s2+s3+s4, s1+s2+s3+s4+s5),
                                thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4-s1-s2-s3, t5-s1-s2-s3-s4, t6)).begin();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
typename join_iterator< thrust::tuple<T1,T2,T3,T4,T5,T6,T7> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7)
{
    typedef thrust::tuple<T1,T2,T3,T4,T5,T6,T7>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3, s1+s2+s3+s4, s1+s2+s3+s4+s5, s1+s2+s3+s4+s5+s6),
                                thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4-s1-s2-s3, t5-s1-s2-s3-s4, t6-s1-s2-s3-s4-s5, t7)).begin();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
typename join_iterator< thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8)
{
    typedef thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3, s1+s2+s3+s4, s1+s2+s3+s4+s5, s1+s2+s3+s4+s5+s6, s1+s2+s3+s4+s5+s6+s7),
                                thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4-s1-s2-s3, t5-s1-s2-s3-s4, t6-s1-s2-s3-s4-s5, t7-s1-s2-s3-s4-s5-s6, t8)).begin();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
typename join_iterator< thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7, const size_t s8,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8, const T9& t9)
{
    typedef thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3, s1+s2+s3+s4, s1+s2+s3+s4+s5, s1+s2+s3+s4+s5+s6, s1+s2+s3+s4+s5+s6+s7, s1+s2+s3+s4+s5+s6+s7+s8),
                                thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4-s1-s2-s3, t5-s1-s2-s3-s4, t6-s1-s2-s3-s4-s5, t7-s1-s2-s3-s4-s5-s6, t8-s1-s2-s3-s4-s5-s6-s7, t9)).begin();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
typename join_iterator< thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7, const size_t s8, const size_t s9,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8, const T9& t9, const T10& t10)
{
    typedef thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3, s1+s2+s3+s4, s1+s2+s3+s4+s5, s1+s2+s3+s4+s5+s6, s1+s2+s3+s4+s5+s6+s7, s1+s2+s3+s4+s5+s6+s7+s8, s1+s2+s3+s4+s5+s6+s7+s8+s9),
                                thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4-s1-s2-s3, t5-s1-s2-s3-s4, t6-s1-s2-s3-s4-s5, t7-s1-s2-s3-s4-s5-s6, t8-s1-s2-s3-s4-s5-s6-s7, t9-s1-s2-s3-s4-s5-s6-s7-s8, t10)).begin();
}

/*! \} // end iterators
 */

} // end namespace cusp

