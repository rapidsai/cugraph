/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#pragma once

#include <cusp/complex.h>
#include <cusp/detail/type_traits.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/numeric_traits.h>
#include <thrust/detail/type_traits.h>

#include <cstddef>

namespace cusp
{
namespace detail
{

template<typename T>
struct random_iterator_type
{
    typedef typename thrust::detail::eval_if<
           sizeof(T) <= 4,
           thrust::detail::identity_< unsigned int >,
           thrust::detail::identity_< unsigned long long >
        >::type type;
};

template <typename Real>
struct integer_to_real : public thrust::unary_function<typename random_iterator_type<Real>::type,Real>
{
    typedef typename random_iterator_type<Real>::type UnsignedInteger;

    __host__ __device__
    Real operator()(const UnsignedInteger i) const
    {
        const Real integer_bound = Real(UnsignedInteger(1) << (4 * sizeof(UnsignedInteger))) * Real(UnsignedInteger(1) << (4 * sizeof(UnsignedInteger)));
        return Real(i) / integer_bound;
    }
};

template <typename Complex>
struct integer_to_complex : public thrust::unary_function<typename random_iterator_type<Complex>::type,Complex>
{
    typedef typename random_iterator_type<Complex>::type UnsignedInteger;
    typedef typename cusp::norm_type<Complex>::type Real;
    typedef integer_to_real<Real> IntegerToRealGenerator;

    IntegerToRealGenerator generator;

    __host__ __device__
    Complex operator()(const UnsignedInteger i) const
    {
        return Complex(generator(i), generator(i + (1<<20)-1));
    }
};

template<typename T>
struct random_functor_type
{
    typedef typename thrust::detail::eval_if<
           thrust::detail::is_floating_point<typename cusp::norm_type<T>::type>::value,
              thrust::detail::eval_if<thrust::detail::is_convertible<thrust::complex<float>,T>::value,
                thrust::detail::identity_< integer_to_complex<T> >,
                thrust::detail::identity_< integer_to_real<T> > >,
           thrust::detail::identity_< thrust::identity<T> >
           >::type type;
};

// Integer hash functions
template <typename IndexType, typename BaseType>
struct random_integer_functor : public thrust::unary_function<IndexType,typename random_iterator_type<BaseType>::type>
{
    size_t seed;

    typedef typename random_iterator_type<BaseType>::type T;

    random_integer_functor(const size_t seed = 0)
        : seed(seed) {}

    // source: http://www.concentric.net/~ttwang/tech/inthash.htm
    __host__ __device__
    T hash(const IndexType i, thrust::detail::false_type) const
    {
        unsigned int h = (unsigned int) i ^ (unsigned int) seed;
        h = ~h + (h << 15);
        h =  h ^ (h >> 12);
        h =  h + (h <<  2);
        h =  h ^ (h >>  4);
        h =  h + (h <<  3) + (h << 11);
        h =  h ^ (h >> 16);
        return T(h);
    }

    __host__ __device__
    T hash(const IndexType i, thrust::detail::true_type) const
    {
        unsigned long long h = (unsigned long long) i ^ (unsigned long long) seed;
        h = ~h + (h << 21);
        h =  h ^ (h >> 24);
        h = (h + (h <<  3)) + (h << 8);
        h =  h ^ (h >> 14);
        h = (h + (h <<  2)) + (h << 4);
        h =  h ^ (h >> 28);
        h =  h + (h << 31);
        return T(h);
    }

    __host__ __device__
    T operator()(const IndexType i) const
    {
        return hash(i, typename thrust::detail::integral_constant<bool, sizeof(IndexType) == 8 || sizeof(T) == 8>::type());
    }
};

} // end detail
} // end cusp

