/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "HostDevice.hpp"
#include <array>
#include <cstdint>

namespace xlib {

// ======================= COMPILE TIME numeric methods ========================
/**
 * @tparam A first parameter
 * @tparam B second parameter
 * @tparam ARGS additional parameters
 */
template<unsigned N, unsigned DIV>      struct CeilDiv;
template<uint64_t N, uint64_t DIV>      struct CeilDivUll;

/**
 * @brief compute
 * \f$ \[\left\lfloor {\frac{N}{{\textif{DIV}}} + 0.5} \right\rfloor \] \f$
 */
template<unsigned N, unsigned DIV>      struct RoundDiv;

template<unsigned N, unsigned EXP>      struct Pow;

template<unsigned N>                    struct Log2;
template<uint64_t N>                    struct Log2Ull;
template<unsigned N>                    struct CeilLog2;
template<uint64_t N>                    struct CeilLog2Ull;
template<unsigned N, unsigned BASE>     struct CeilLog;

template<unsigned N, unsigned K>        struct BinomialCoeff;
template<unsigned LOW, unsigned HIGH>   struct ProductSequence;
template<unsigned N, unsigned HIGH>     struct GeometricSerie;
//------------------------------------------------------------------------------

template<typename... TArgs>
struct SameSize;

template<typename... TArgs>
struct SizeSum;

template<typename... TArgs>
struct MaxSize;

template<int N, typename... TArgs>
struct SelectType {
    using type = int;
};
//------------------------------------------------------------------------------

template<unsigned... Is>
class Seq {
public:
    static constexpr unsigned size();

    constexpr unsigned operator[](int index) const;
private:
    static constexpr unsigned value[] = { Is... };
};

//------------------------------------------------------------------------------

///@cond

/*
constexpr int fun(int i) { return i * 3; };
GenerateSeq<fun, 4>::type table;
f<table[0]>();
f<table[1]>();
*/
template<unsigned(*fun)(unsigned), unsigned MAX, unsigned INDEX = 0,
         unsigned... Is>
struct GenerateSeq : GenerateSeq<fun, MAX, INDEX + 1, Is..., fun(INDEX)>{};

template<unsigned(*fun)(unsigned), unsigned MAX, unsigned... Is>
struct GenerateSeq<fun, MAX, MAX, Is...>  {
    using type = Seq<Is...>;
};

template<typename Seq>
struct IncPrefixSum;

template<typename Seq>
struct ExcPrefixSum;

//@endcond

//==============================================================================

//template<int N, typename... TArgs>
//using NthTypeOf = typename std::tuple_element<N, std::tuple<TArgs...>>::type;

template<typename Tuple1, typename Tuple2>
struct TupleConcat;

template<typename Tuple1, typename Tuple2>
struct tuple_compare;

template<typename Tuple>
struct tuple_rm_pointers;

template<typename Tuple>
struct TupleToTypeSizeSeq;

//==============================================================================

template<typename T>
HOST_DEVICE
constexpr unsigned get_arity(T);

template<typename T>
HOST_DEVICE
constexpr unsigned get_arity();

//==============================================================================
//https://stackoverflow.com/a/12982320/6585879

template<typename T>
using EnableP = decltype( std::declval<std::ostream&>() << std::declval<T>() );

template<typename T, typename = void>
struct is_stream_insertable : std::false_type {};

template<typename T>
struct is_stream_insertable<T, EnableP<T>> : std::true_type {};


template<int, typename>
struct get_type;

template<int INDEX, typename R, typename T, typename... TArgs>
struct get_type<INDEX, R(*)(T, TArgs...)> {
    using type = typename get_type<INDEX - 1, R(*)(TArgs...)>::type;
};

template<typename R, typename T, typename... TArgs>
struct get_type<0, R(*)(T, TArgs...)> {
    using type = T;
};

template<int INDEX, typename R, typename T, typename... TArgs>
struct get_type<INDEX, R (T, TArgs...)> {
    using type = typename get_type<INDEX - 1, R(*)(TArgs...)>::type;
};

template<typename R, typename T, typename... TArgs>
struct get_type<0, R (T, TArgs...)> {
    using type = T;
};

template<typename T>
struct closure_type : public closure_type< decltype(&T::operator()) > {};

template<typename C, typename R, typename... Args>
struct closure_type<R (C::*)(Args...) const> {
  using ReturnType = R;
};

//------------------------------------------------------------------------------

template<typename T>
struct remove_const_ptr {
    using type = T;
};

template<typename T>
struct remove_const_ptr<const T*> {
    using type = T*;
};

} // namespace xlib

#include "impl/Metaprogramming.i.hpp"
