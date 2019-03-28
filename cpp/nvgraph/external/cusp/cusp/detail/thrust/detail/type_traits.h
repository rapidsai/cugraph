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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

namespace detail
{
template<typename T1, typename T2, typename Enable = void> struct promoted_numerical_type;

template<typename T1, typename T2>
struct promoted_numerical_type<T1,T2,typename enable_if<and_
        <typename is_floating_point<T1>::type,typename is_floating_point<T2>::type>
        ::value>::type>
{
    typedef larger_type<T1,T2> type;
};

template<typename T1, typename T2>
struct promoted_numerical_type<T1,T2,typename enable_if<and_
        <typename is_integral<T1>::type,typename is_floating_point<T2>::type>
        ::value>::type>
{
    typedef T2 type;
};

template<typename T1, typename T2>
struct promoted_numerical_type<T1,T2,typename enable_if<and_
        <typename is_floating_point<T1>::type, typename is_integral<T2>::type>
        ::value>::type>
{
    typedef T1 type;
};

} // end detail

} // end thrust

#include <thrust/detail/type_traits/has_trivial_assign.h>

