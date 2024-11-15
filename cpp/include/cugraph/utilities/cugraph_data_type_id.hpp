/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cugraph_c/types.h>

#include <cugraph/utilities/error.hpp>

#include <cstdint>
#include <string>

namespace cugraph {

// Type conversion logic adapted from CUDF

/**
 * @brief Maps a C++ type to its corresponding `cugraph_data_type_id_t`
 *
 * When explicitly passed a template argument of a given type, returns the
 * appropriate `cugraph_data_type_id_t` enum for the specified C++ type.
 *
 * For example:
 *
 * ```
 * return cugraph::type_to_id<int32_t>();        // Returns INT32
 * ```
 *
 * @tparam T The type to map to a `cugraph_data_type_id_t`
 * @return The `cugraph_data_type_id_t` corresponding to the specified type
 */
template <typename T>
inline constexpr cugraph_data_type_id_t type_to_id()
{
  return cugraph_data_type_id_t::NTYPES;
};

/**
 * @brief Maps a `cugraph_data_type_id_t` types to its corresponding C++ type name string
 *
 */
struct type_to_name_impl {
  /**
   * @brief Maps a `cugraph_data_type_id_t` types to its corresponding C++ type name string
   *
   * @return The C++ type name as string
   */
  template <typename T>
  inline std::string operator()()
  {
    return "void";
  }
};

template <cugraph_data_type_id_t t>
struct id_to_type_impl {
  using type = void;
};

/**
 * @brief Maps a `cugraph_data_type_id_t` to its corresponding concrete C++ type
 *
 * Example:
 * ```
 * static_assert(std::is_same<int32_t, id_to_type<cugraph_data_type_id_t::INT32>);
 * ```
 * @tparam t The `cugraph_data_type_id_t` to map
 */
template <cugraph_data_type_id_t Id>
using id_to_type = typename id_to_type_impl<Id>::type;

/**
 * @brief Macro used to define a mapping between a concrete C++ type and a
 *`cugraph_data_type_id_t` enum.

 * @param Type The concrete C++ type
 * @param Id The `cugraph_data_type_id_t` enum
 */
#ifndef CUGRAPH_TYPE_MAPPING
#define CUGRAPH_TYPE_MAPPING(Type, Id)                       \
  template <>                                                \
  constexpr inline cugraph_data_type_id_t type_to_id<Type>() \
  {                                                          \
    return Id;                                               \
  }                                                          \
  template <>                                                \
  inline std::string type_to_name_impl::operator()<Type>()   \
  {                                                          \
    return #Type;                                            \
  }                                                          \
  template <>                                                \
  struct id_to_type_impl<Id> {                               \
    using type = Type;                                       \
  };
#endif

// Defines all of the mappings between C++ types and their corresponding `cugraph_data_type_id_t`
// values.
CUGRAPH_TYPE_MAPPING(int8_t, cugraph_data_type_id_t::INT8)
CUGRAPH_TYPE_MAPPING(int16_t, cugraph_data_type_id_t::INT16)
CUGRAPH_TYPE_MAPPING(int32_t, cugraph_data_type_id_t::INT32)
CUGRAPH_TYPE_MAPPING(int64_t, cugraph_data_type_id_t::INT64)
CUGRAPH_TYPE_MAPPING(uint8_t, cugraph_data_type_id_t::UINT8)
CUGRAPH_TYPE_MAPPING(uint16_t, cugraph_data_type_id_t::UINT16)
CUGRAPH_TYPE_MAPPING(uint32_t, cugraph_data_type_id_t::UINT32)
CUGRAPH_TYPE_MAPPING(uint64_t, cugraph_data_type_id_t::UINT64)
CUGRAPH_TYPE_MAPPING(float, cugraph_data_type_id_t::FLOAT32)
CUGRAPH_TYPE_MAPPING(double, cugraph_data_type_id_t::FLOAT64)

// These are duplicative of uint8_t and uint64_t
// CUGRAPH_TYPE_MAPPING(bool, cugraph_data_type_id_t::BOOL)
// CUGRAPH_TYPE_MAPPING(size_t, cugraph_data_type_id_t::SIZE_T)

/**
 * @brief Invokes an `operator()` template with the type instantiation based on
 * the specified `cudf::data_type`'s `id()`.
 *
 * Example usage with a functor that returns the size of the dispatched type:
 *
 * @code
 * struct size_of_functor{
 *  template <typename T>
 *  int operator()(){
 *    return sizeof(T);
 *  }
 * };
 * cugraph_data_type_id_t t{INT32};
 * cugraph::type_dispatcher(t, size_of_functor{});  // returns 4
 * @endcode
 *
 * The `type_dispatcher` uses `cugraph::type_to_id<t>` to provide a default mapping
 * of `cugraph_data_type_id_t`s to dispatched C++ types. However, this mapping may be
 * customized by explicitly specifying a user-defined trait struct for the
 * `IdTypeMap`. For example, to always dispatch `int32_t`
 *
 * @code
 * template<cugraph_data_type_id_t t> struct always_int{ using type = int32_t; }
 *
 * // This will always invoke operator()<int32_t>
 * cugraph::type_dispatcher<always_int>(data_type, f);
 * @endcode
 *
 * It is sometimes necessary to customize the dispatched functor's
 * `operator()` for different types.  This can be done in several ways.
 *
 * The first method is to use explicit template specialization. This is useful
 * for specializing behavior for single types. For example, a functor that
 * prints `int32_t` or `double` when invoked with either of those types, else it
 * prints `unhandled type`:
 *
 * @code
 * struct type_printer {
 *   template <typename T>
 *   void operator()() { std::cout << "unhandled type\n"; }
 * };
 *
 * // Due to a bug in g++, explicit member function specializations need to be
 * // defined outside of the class definition
 * template <>
 * void type_printer::operator()<int32_t>() { std::cout << "int32_t\n"; }
 *
 * template <>
 * void type_printer::operator()<double>() { std::cout << "double\n"; }
 * @endcode
 *
 * A second method is to use SFINAE with `std::enable_if_t`. This is useful for
 * specializing for a set of types that share some property. For example, a
 * functor that prints `integral` or `floating point` for integral or floating
 * point types:
 *
 * @code
 * struct integral_or_floating_point {
 *   template <typename T,
 *             std::enable_if_t<not std::is_integral_v<T>  and
 *                              not std::is_floating_point_v<T> >* = nullptr>
 *   void operator()() {
 *     std::cout << "neither integral nor floating point\n "; }
 *
 *   template <typename T,
 *             std::enable_if_t<std::is_integral_v<T> >* = nullptr>
 *   void operator()() { std::cout << "integral\n"; }
 *
 *   template <typename T,
 *             std::enable_if_t<std::is_floating_point_v<T> >* = nullptr>
 *   void operator()() { std::cout << "floating point\n"; }
 * };
 * @endcode
 *
 * For more info on SFINAE and `std::enable_if`, see
 * https://eli.thegreenplace.net/2014/sfinae-and-enable_if/
 *
 * The return type for all template instantiations of the functor's "operator()"
 * lambda must be the same, else there will be a compiler error as you would be
 * trying to return different types from the same function.
 *
 * @tparam id_to_type_impl Maps a `cugraph_data_type_id_t` its dispatched C++ type
 * @tparam Functor The callable object's type
 * @tparam Ts Variadic parameter pack type
 * @param dtype The `cugraph_data_type_id_t` whose `id()` determines which template
 * instantiation is invoked
 * @param f The callable whose `operator()` template is invoked
 * @param args Parameter pack of arguments forwarded to the `operator()`
 * invocation
 * @return Whatever is returned by the callable's `operator()`
 */
// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <template <cugraph_data_type_id_t> typename IdTypeMap = id_to_type_impl,
          typename Functor,
          typename... Ts>
#ifdef __CUDACC__
__host__ __device__
#endif
  inline constexpr decltype(auto)
  type_dispatcher(cugraph_data_type_id_t dtype, Functor f, Ts&&... args)
{
  switch (dtype) {
    case cugraph_data_type_id_t::INT8:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::INT8>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::INT16:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::INT16>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::INT32:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::INT32>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::INT64:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::INT64>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::UINT8:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::UINT8>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::UINT16:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::UINT16>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::UINT32:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::UINT32>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::UINT64:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::UINT64>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::FLOAT32:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::FLOAT32>::type>(
        std::forward<Ts>(args)...);
    case cugraph_data_type_id_t::FLOAT64:
      return f.template operator()<typename IdTypeMap<cugraph_data_type_id_t::FLOAT64>::type>(
        std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUGRAPH_FAIL("Invalid type_id.");
#else
      // Invalid type_id, unchecked in device code, this should be unreachable
#endif
    }
  }
}

/**
 * @brief Return a name for a given type.
 *
 * The returned type names are intended for error messages and are not
 * guaranteed to be stable.
 *
 * @param type The `data_type`
 * @return Name of the type
 */
std::string type_to_name(cugraph_data_type_id_t type);

// End of code adapted from CUDF

/**
 * @brief Return the size (in bytes) of the specified data type
 *
 * @param type The data type
 * @return Size of the specified data type
 */
std::size_t data_type_size(cugraph_data_type_id_t type);

}  // namespace cugraph
