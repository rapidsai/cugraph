/**
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
 */
#include "Host/Numeric.hpp"     //xlib::factorial

namespace xlib {

template<unsigned N, unsigned DIV>
struct CeilDiv {
    static_assert(DIV != 0, "division by zero in integer arithmetic");
    static const unsigned value = N == 0 ? 0 : 1 + ((N - 1) / DIV);
};

template<uint64_t N, uint64_t DIV>
struct CeilDivUll {
    static_assert(DIV != 0, "division by zero in integer arithmetic");
    static const uint64_t value = N == 0 ? 0 : 1 + ((N - 1) / DIV);
};

template<unsigned N, unsigned DIV>
struct RoundDiv {
    static const unsigned value = (N + (DIV / 2)) / DIV;
};

template<unsigned N, unsigned MUL>
struct LowerApprox {
    static const unsigned value = (N / MUL) * MUL;
};
template<uint64_t N, uint64_t MUL>
struct LowerApproxUll {
    static const uint64_t value = (N / MUL) * MUL;
};

template<unsigned N, unsigned MUL>
struct UpperApprox {
    static const unsigned value = CeilDiv<N, MUL>::value * MUL;
};
template<uint64_t N, uint64_t MUL>
struct UpperApproxUll {
    static const uint64_t value = CeilDivUll<N, MUL>::value * MUL;
};

template<unsigned N, unsigned EXP>
struct Pow {
    static const unsigned value = Pow<N, EXP - 1>::value * N;
};
template<unsigned N>
struct Pow<N, 0> {
    static const unsigned value = 1;
};

//------------------------------------------------------------------------------

//lower bound
template<unsigned N, unsigned BASE>
struct Log {
    static_assert(N > 0, "Log : N <= 0");
    static const unsigned value = N < BASE ? 0 :
                                  1 + Log<xlib::max(1u, N / BASE), BASE>::value;
};
template<unsigned BASE>
struct Log<1, BASE> {
    static const unsigned value = 0;
};

//lower bound
template<uint64_t N, uint64_t BASE>
struct LogUll {
    static_assert(N > 0, "Log : N <= 0");
    static const uint64_t value = 1 + LogUll<N / BASE, BASE>::value;
};
template<uint64_t BASE>
struct LogUll<1, BASE> {
    static const uint64_t value = 0;
};

//lower bound
template<unsigned N>
struct Log2 {
    static const unsigned value = Log<N, 2>::value;
};

template<uint64_t N>
struct Log2Ull {
    static const uint64_t value = LogUll<N, 2>::value;
};

template<unsigned N>
struct CeilLog2 {
    static const unsigned value = Log2<xlib::roundup_pow2(N)>::value;
};

template<uint64_t N>
struct CeilLog2Ull {
    static const uint64_t value = Log2Ull<xlib::roundup_pow2(N)>::value;
};

template<unsigned N, unsigned BASE>
struct CeilLog {
private:
    static const unsigned LOG = Log<N, BASE>::value;
public:
    static const unsigned value = Pow<BASE, LOG>::value == N ? LOG : LOG + 1;
};

//------------------------------------------------------------------------------

template<unsigned LOW, unsigned HIGH>
struct ProductSequence {
    static const unsigned value = LOW * ProductSequence<LOW + 1, HIGH>::value;
};
template<unsigned LOW>
struct ProductSequence<LOW, LOW> {
    static const unsigned value = LOW;
};

template<unsigned N, unsigned K>
struct BinomialCoeff {
static_assert(N >= 0 && K >= 0 && K <= N, "BinomialCoeff");
private:
    static const unsigned MIN = xlib::min(K, N - K);
    static const unsigned MAX = xlib::max(K, N - K);
public:
    static const unsigned value = ProductSequence<MAX + 1, N>::value /
                                  xlib::factorial(MIN);
};
template<unsigned N>
struct BinomialCoeff<N ,N> {
    static const unsigned value = 1;
};

template<unsigned N, unsigned HIGH>
struct GeometricSerie {
    static const unsigned value = (Pow<N, HIGH + 1>::value - 1) / (N - 1);
};

//==============================================================================

template<typename T1, typename T2, typename... TArgs>
struct SameSize<T1, T2, TArgs...> {
    static const bool value = sizeof(T1) == sizeof(T2) &&
                              SameSize<T2, TArgs...>::value;
};

template<typename T>
struct SameSize<T> : std::true_type {};

template<typename T, typename... TArgs>
struct SizeSum<T, TArgs...> {
    static const unsigned value = sizeof(T) + SizeSum<TArgs...>::value;
};

template<typename T>
struct SizeSum<T> {
    static const unsigned value = sizeof(T);
};

//FirstNSizeSum<0, T, Ts...>::value should be 0
//FirstNSizeSum<1, T, Ts...>::value should be sizeof(T)
//FirstNSizeSum<2, T0, T1, Ts...>::value should be sizeof(T0) + sizeof(T1)
template<int N, typename... TArgs>
struct FirstNSizeSum {
    static const unsigned value = 0;
};
//------------------------------------------------------------------------------

template<int N, typename T, typename... Ts>
struct FirstNSizeSum<N, T, Ts...> {
    static_assert(N <= 1+(sizeof...(Ts)), "FirstNSizeSum index exceeds parameter pack size");
    static const unsigned value = sizeof(T) + FirstNSizeSum<N-1, Ts...>::value;
};

template<typename T, typename... Ts>
struct FirstNSizeSum<0, T, Ts...> {
    static const unsigned value = 0;
};

template<typename... TArgs>
struct MaxSize {
    static const unsigned value = xlib::max(sizeof(TArgs)...);
};

template<typename T>
struct MaxSize<T> {
    static const unsigned value = sizeof(T);
};

template<int N, typename T, typename... TArgs>
struct SelectType<N, T, TArgs...> {
    using type = typename SelectType<N - 1, TArgs...>::type;
};

template<typename T, typename... TArgs>
struct SelectType<0, T, TArgs...> {
    using type = T;
};

//------------------------------------------------------------------------------

template<unsigned... Is>
constexpr unsigned Seq<Is...>::size() {
    return sizeof...(Is);
}

template<unsigned... Is>
constexpr unsigned Seq<Is...>::operator[](int index) const {
    return value[index];
}

template<unsigned... Is>
constexpr unsigned Seq<Is...>::value[];                                //NOTLINT

//------------------------------------------------------------------------------

template<unsigned, typename, typename>
struct PrefixSumAux;

template<unsigned... Is>
struct IncPrefixSum<Seq<Is...>> :
    PrefixSumAux<sizeof...(Is), Seq<>, Seq<Is...>> {};

template<unsigned... Is>
struct ExcPrefixSum<Seq<Is...>> :
    PrefixSumAux<sizeof...(Is) + 1, Seq<>, Seq<0, Is...>> {};

template<unsigned INDEX, unsigned I1, unsigned I2, unsigned... Is2>
struct PrefixSumAux<INDEX, Seq<>, Seq<I1, I2, Is2...>> :
       PrefixSumAux<INDEX - 1, Seq<I1, I1 + I2>,  Seq<I1 + I2, Is2...>> {};

template<unsigned INDEX, unsigned... Is1,
         unsigned I1, unsigned I2, unsigned... Is2>
struct PrefixSumAux<INDEX, Seq<Is1...>, Seq<I1, I2, Is2...>> :
   PrefixSumAux<INDEX - 1, Seq<Is1..., I1 + I2>,  Seq<I1 + I2, Is2...>> {};

template<unsigned... Is1, unsigned... Is2>
struct PrefixSumAux<1, Seq<Is1...>, Seq<Is2...>> {
    using type = Seq<Is1...>;
};

//------------------------------------------------------------------------------

template<typename, typename>
struct tuple_rm_pointers_aux;

template<typename... TArgs>
struct tuple_rm_pointers<std::tuple<TArgs...>> {
    using type = typename tuple_rm_pointers_aux<std::tuple<>,
                                                std::tuple<TArgs...>>::type;
};

template<typename... TArgs1, typename T2, typename... TArgs2>
struct tuple_rm_pointers_aux<std::tuple<TArgs1...>, std::tuple<T2, TArgs2...>> :
    tuple_rm_pointers_aux<std::tuple<TArgs1...,
                                     typename std::remove_pointer<T2>::type>,
                          std::tuple<TArgs2...>> {};

template<typename... TArgs>
struct tuple_rm_pointers_aux<std::tuple<TArgs...>, std::tuple<>> {
    using type = std::tuple<TArgs...>;
};

//------------------------------------------------------------------------------

template<typename... TArgs1, typename... TArgs2>
struct TupleConcat<std::tuple<TArgs1...>, std::tuple<TArgs2...>> {
    using type = std::tuple<TArgs1..., TArgs2...>;
};

//------------------------------------------------------------------------------

template<typename T1, typename... TArgs1, typename T2, typename... TArgs2>
struct tuple_compare<std::tuple<T1, TArgs1...>, std::tuple<T2, TArgs2...>> {
    static const bool value =
            std::is_same<typename std::remove_cv<T1>::type,
                         typename std::remove_cv<T2>::type>::value &&
            tuple_compare<std::tuple<TArgs1...>, std::tuple<TArgs2...>>::value;
};

template<>
struct tuple_compare<std::tuple<>, std::tuple<>> : std::true_type {};

//------------------------------------------------------------------------------

template<typename... TArgs>
struct TupleToTypeSizeSeq<std::tuple<TArgs...>> {
   using type = Seq<sizeof(TArgs)...>;
};

//==============================================================================

template<typename... TArgs>
struct IsVectorizable {
    static const bool value = xlib::SameSize<TArgs...>::value &&
                              xlib::SizeSum<TArgs...>::value <= 16 &&
                              sizeof...(TArgs) <= 4;
};

//==============================================================================
//==============================================================================
//https://stackoverflow.com/a/27867127/6585879

template <typename T>
struct GetArity : GetArity<decltype(&T::operator())> {};

template<typename R, typename... Args>
struct GetArity<R(*)(Args...)> {
    static const unsigned value = sizeof...(Args);
};

template<typename R, typename C, typename... Args>
struct GetArity<R(C::*)(Args...)> {
    static const unsigned value = sizeof...(Args);
};

template<typename R, typename C, typename... Args>
struct GetArity<R(C::*)(Args...) const> {
    static const unsigned value = sizeof...(Args);
};

template<typename T>
HOST_DEVICE
constexpr unsigned get_arity(T) {
    return GetArity<T>::value;
}

template<typename T>
HOST_DEVICE
constexpr unsigned get_arity() {
    return GetArity<T>::value;
}

//==============================================================================
//==============================================================================

} // namespace xlib
