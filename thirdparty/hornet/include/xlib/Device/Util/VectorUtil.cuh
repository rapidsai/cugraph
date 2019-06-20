/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
 * @version v1.4
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
#include <cuda_runtime.h>
#include <limits>
#include <ostream>

namespace std {

/** \addtogroup VectorTypeLimits
 *  Provides numeric_limits max, min, lowest for most common CUDA vector types.
 *  In particular, it supports short2, ushort2, short4, ushort4, short2, ushort2,
 *  short4, ushort4, int2, uint2, int4, uint4, longlong2, ulonglong2, float2,
 *  float4, double2
 *  @{
 */

template<>
class numeric_limits<char2> {
public:
    static char2 min()    noexcept;
    static char2 max()    noexcept;
    static char2 lowest() noexcept;
};

template<>
class numeric_limits<uchar2> {
public:
    static uchar2 min()    noexcept;
    static uchar2 max()    noexcept;
    static uchar2 lowest() noexcept;
};

template<>
class numeric_limits<char4> {
public:
    static char4 min()    noexcept;
    static char4 max()    noexcept;
    static char4 lowest() noexcept;
};

template<>
class numeric_limits<uchar4> {
public:
    static uchar4 min()    noexcept;
    static uchar4 max()    noexcept;
    static uchar4 lowest() noexcept;
};

template<>
class numeric_limits<short2> {
public:
    static short2 min()    noexcept;
    static short2 max()    noexcept;
    static short2 lowest() noexcept;
};

template<>
class numeric_limits<ushort2> {
public:
    static ushort2 min()    noexcept;
    static ushort2 max()    noexcept;
    static ushort2 lowest() noexcept;
};

template<>
class numeric_limits<short4> {
public:
    static short4 min()    noexcept;
    static short4 max()    noexcept;
    static short4 lowest() noexcept;
};

template<>
class numeric_limits<ushort4> {
public:
    static ushort4 min()    noexcept;
    static ushort4 max()    noexcept;
    static ushort4 lowest() noexcept;
};

template<>
class numeric_limits<int2> {
public:
    static int2 min()    noexcept;
    static int2 max()    noexcept;
    static int2 lowest() noexcept;
};

template<>
class numeric_limits<uint2> {
public:
    static uint2 min()    noexcept;
    static uint2 max()    noexcept;
    static uint2 lowest() noexcept;
};

template<>
class numeric_limits<int4> {
public:
    static int4 min()    noexcept;
    static int4 max()    noexcept;
    static int4 lowest() noexcept;
};

template<>
class numeric_limits<uint4> {
public:
    static uint4 min()    noexcept;
    static uint4 max()    noexcept;
    static uint4 lowest() noexcept;
};

template<>
class numeric_limits<longlong2> {
public:
    static longlong2 min()    noexcept;
    static longlong2 max()    noexcept;
    static longlong2 lowest() noexcept;
};

template<>
class numeric_limits<ulonglong2> {
public:
    static ulonglong2 min()    noexcept;
    static ulonglong2 max()    noexcept;
    static ulonglong2 lowest() noexcept;
};

template<>
class numeric_limits<float2> {
public:
    static float2 min()    noexcept;
    static float2 max()    noexcept;
    static float2 lowest() noexcept;
};

template<>
class numeric_limits<float4> {
public:
    static float4 min()    noexcept;
    static float4 max()    noexcept;
    static float4 lowest() noexcept;
};

template<>
class numeric_limits<double2> {
public:
    static double2 min()    noexcept;
    static double2 max()    noexcept;
    static double2 lowest() noexcept;
};

} // namespace std

/**
 * @}
 */
//==============================================================================

/** \addtogroup VectorTypeOstream
 *  Provides ostream utilities for most common CUDA vector types.
 *  In particular, it provides operator<< for short2, ushort2, short4, ushort4,
 *  short2, ushort2, ushort4, int2, uint2, int4, uint4, long2, ulong2, float2,
 *  float4, double2
 *  @{
 */

inline std::ostream& operator<< (std::ostream& out, const short2& value);
inline std::ostream& operator<< (std::ostream& out, const ushort2& value);
inline std::ostream& operator<< (std::ostream& out, const short4& value);
inline std::ostream& operator<< (std::ostream& out, const ushort4& value);
inline std::ostream& operator<< (std::ostream& out, const short2& value);
inline std::ostream& operator<< (std::ostream& out, const ushort2& value);
inline std::ostream& operator<< (std::ostream& out, const short4& value);
inline std::ostream& operator<< (std::ostream& out, const ushort4& value);
inline std::ostream& operator<< (std::ostream& out, const int2& value);
inline std::ostream& operator<< (std::ostream& out, const uint2& value);
inline std::ostream& operator<< (std::ostream& out, const int4& value);
inline std::ostream& operator<< (std::ostream& out, const uint4& value);
inline std::ostream& operator<< (std::ostream& out, const long2& value);
inline std::ostream& operator<< (std::ostream& out, const ulong2& value);
inline std::ostream& operator<< (std::ostream& out, const float2& value);
inline std::ostream& operator<< (std::ostream& out, const double2& value);

/**
 * @}
 */

//==============================================================================

/** \addtogroup VectorTypeCompare
 *  Provides compare utilities for most common CUDA vector types.
 *  In particular, it provides equal, not equal, less, less equal, greater,
 *  greater equal for short2, ushort2, short4, ushort4, short2, ushort2, short4,
 *  ushort4,int2, uint2, int4, uint4, long2, float2, float4, double2
 *  @{
 */

HOST_DEVICE bool operator== (const char2& A, const char2& B);
HOST_DEVICE bool operator!= (const char2& A, const char2& B);
HOST_DEVICE bool operator<  (const char2& A, const char2& B);
HOST_DEVICE bool operator<= (const char2& A, const char2& B);
HOST_DEVICE bool operator>  (const char2& A, const char2& B);
HOST_DEVICE bool operator>= (const char2& A, const char2& B);

HOST_DEVICE bool operator== (const uchar2& A, const uchar2& B);
HOST_DEVICE bool operator!= (const uchar2& A, const uchar2& B);
HOST_DEVICE bool operator<  (const uchar2& A, const uchar2& B);
HOST_DEVICE bool operator<= (const uchar2& A, const uchar2& B);
HOST_DEVICE bool operator>  (const uchar2& A, const uchar2& B);
HOST_DEVICE bool operator>= (const uchar2& A, const uchar2& B);

HOST_DEVICE bool operator== (const char4& A, const char4& B);
HOST_DEVICE bool operator!= (const char4& A, const char4& B);
HOST_DEVICE bool operator<  (const char4& A, const char4& B);
HOST_DEVICE bool operator<= (const char4& A, const char4& B);
HOST_DEVICE bool operator>  (const char4& A, const char4& B);
HOST_DEVICE bool operator>= (const char4& A, const char4& B);

HOST_DEVICE bool operator== (const uchar4& A, const uchar4& B);
HOST_DEVICE bool operator!= (const uchar4& A, const uchar4& B);
HOST_DEVICE bool operator<  (const uchar4& A, const uchar4& B);
HOST_DEVICE bool operator<= (const uchar4& A, const uchar4& B);
HOST_DEVICE bool operator>  (const uchar4& A, const uchar4& B);
HOST_DEVICE bool operator>= (const uchar4& A, const uchar4& B);

HOST_DEVICE bool operator== (const short2& A, const short2& B);
HOST_DEVICE bool operator!= (const short2& A, const short2& B);
HOST_DEVICE bool operator<  (const short2& A, const short2& B);
HOST_DEVICE bool operator<= (const short2& A, const short2& B);
HOST_DEVICE bool operator>  (const short2& A, const short2& B);
HOST_DEVICE bool operator>= (const short2& A, const short2& B);

HOST_DEVICE bool operator== (const ushort2& A, const ushort2& B);
HOST_DEVICE bool operator!= (const ushort2& A, const ushort2& B);
HOST_DEVICE bool operator<  (const ushort2& A, const ushort2& B);
HOST_DEVICE bool operator<= (const ushort2& A, const ushort2& B);
HOST_DEVICE bool operator>  (const ushort2& A, const ushort2& B);
HOST_DEVICE bool operator>= (const ushort2& A, const ushort2& B);

HOST_DEVICE bool operator== (const short4& A, const short4& B);
HOST_DEVICE bool operator!= (const short4& A, const short4& B);
HOST_DEVICE bool operator<  (const short4& A, const short4& B);
HOST_DEVICE bool operator<= (const short4& A, const short4& B);
HOST_DEVICE bool operator>  (const short4& A, const short4& B);
HOST_DEVICE bool operator>= (const short4& A, const short4& B);

HOST_DEVICE bool operator== (const ushort4& A, const ushort4& B);
HOST_DEVICE bool operator!= (const ushort4& A, const ushort4& B);
HOST_DEVICE bool operator<  (const ushort4& A, const ushort4& B);
HOST_DEVICE bool operator<= (const ushort4& A, const ushort4& B);
HOST_DEVICE bool operator>  (const ushort4& A, const ushort4& B);
HOST_DEVICE bool operator>= (const ushort4& A, const ushort4& B);

HOST_DEVICE bool operator== (const int2& A, const int2& B);
HOST_DEVICE bool operator!= (const int2& A, const int2& B);
HOST_DEVICE bool operator<  (const int2& A, const int2& B);
HOST_DEVICE bool operator<= (const int2& A, const int2& B);
HOST_DEVICE bool operator>  (const int2& A, const int2& B);
HOST_DEVICE bool operator>= (const int2& A, const int2& B);

HOST_DEVICE bool operator== (const uint2& A, const uint2& B);
HOST_DEVICE bool operator!= (const uint2& A, const uint2& B);
HOST_DEVICE bool operator<  (const uint2& A, const uint2& B);
HOST_DEVICE bool operator<= (const uint2& A, const uint2& B);
HOST_DEVICE bool operator>  (const uint2& A, const uint2& B);
HOST_DEVICE bool operator>= (const uint2& A, const uint2& B);

HOST_DEVICE bool operator== (const int4& A, const int4& B);
HOST_DEVICE bool operator!= (const int4& A, const int4& B);
HOST_DEVICE bool operator<  (const int4& A, const int4& B);
HOST_DEVICE bool operator<= (const int4& A, const int4& B);
HOST_DEVICE bool operator>  (const int4& A, const int4& B);
HOST_DEVICE bool operator>= (const int4& A, const int4& B);

HOST_DEVICE bool operator== (const uint4& A, const uint4& B);
HOST_DEVICE bool operator!= (const uint4& A, const uint4& B);
HOST_DEVICE bool operator<  (const uint4& A, const uint4& B);
HOST_DEVICE bool operator<= (const uint4& A, const uint4& B);
HOST_DEVICE bool operator>  (const uint4& A, const uint4& B);
HOST_DEVICE bool operator>= (const uint4& A, const uint4& B);

HOST_DEVICE bool operator== (const longlong2& A, const longlong2& B);
HOST_DEVICE bool operator!= (const longlong2& A, const longlong2& B);
HOST_DEVICE bool operator<  (const longlong2& A, const longlong2& B);
HOST_DEVICE bool operator<= (const longlong2& A, const longlong2& B);
HOST_DEVICE bool operator>  (const longlong2& A, const longlong2& B);
HOST_DEVICE bool operator>= (const longlong2& A, const longlong2& B);

HOST_DEVICE bool operator== (const ulonglong2& A, const ulonglong2& B);
HOST_DEVICE bool operator!= (const ulonglong2& A, const ulonglong2& B);
HOST_DEVICE bool operator<  (const ulonglong2& A, const ulonglong2& B);
HOST_DEVICE bool operator<= (const ulonglong2& A, const ulonglong2& B);
HOST_DEVICE bool operator>  (const ulonglong2& A, const ulonglong2& B);
HOST_DEVICE bool operator>= (const ulonglong2& A, const ulonglong2& B);

HOST_DEVICE bool operator== (const float2& A, const float2& B);
HOST_DEVICE bool operator!= (const float2& A, const float2& B);
HOST_DEVICE bool operator<  (const float2& A, const float2& B);
HOST_DEVICE bool operator<= (const float2& A, const float2& B);
HOST_DEVICE bool operator>  (const float2& A, const float2& B);
HOST_DEVICE bool operator>= (const float2& A, const float2& B);

HOST_DEVICE bool operator== (const float4& A, const float4& B);
HOST_DEVICE bool operator!= (const float4& A, const float4& B);
HOST_DEVICE bool operator<  (const float4& A, const float4& B);
HOST_DEVICE bool operator<= (const float4& A, const float4& B);
HOST_DEVICE bool operator>  (const float4& A, const float4& B);
HOST_DEVICE bool operator>= (const float4& A, const float4& B);

HOST_DEVICE bool operator== (const double2& A, const double2& B);
HOST_DEVICE bool operator!= (const double2& A, const double2& B);
HOST_DEVICE bool operator<  (const double2& A, const double2& B);
HOST_DEVICE bool operator<= (const double2& A, const double2& B);
HOST_DEVICE bool operator>  (const double2& A, const double2& B);
HOST_DEVICE bool operator>= (const double2& A, const double2& B);

/**
 * @}
 */

//==============================================================================

namespace xlib {

/**
 * @brief Provides the Vector Type of dimension 2 of a given type
 * @details It supports char, unsigned char, short, unsigned short, int,
 * unsigned, long long, long long unsigned, float, double <br>
 * e.g. using int2 = typename Make2Str<int>::type.
 */
template<typename T>
struct Make2Str {
    using type = void;
};

/**
 * @brief Provides the Vector Type of dimension 4 of a given type
 * @details It supports char, unsigned char, short, unsigned short, int,
 * unsigned, float
 * e.g. using int4 = typename Make4Str<int>::type
 */
template<typename T>
struct Make4Str {
    using type = void;
};

/**
 * @brief Returns vector value of dimension 2 of two given values
 * @param[in] a first component
 * @param[in] b second component
 * @return vector value of dimension 2 <a, b>
 * @see Make2Str
 */
template<typename T>
__host__ __device__ __forceinline__
typename Make2Str<T>::type make2(T a, T b);

/**
 * @brief Returns vector value of dimension 2 of two given values
 * @param[in] a first component
 * @param[in] b second component
 * @param[in] c third component
 * @param[in] d fourth component
 * @return vector value of dimension 4 <a, b, c, d>
 * @see Make4Str
 */
template<typename T>
__host__ __device__ __forceinline__
typename Make4Str<T>::type make4(T a, T b, T c, T d);

} // namespace xlib

#include "impl/VectorUtil.i.cuh"
