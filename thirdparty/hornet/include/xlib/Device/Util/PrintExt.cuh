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
#include <string>  //std::string

namespace xlib {
namespace gpu {

/**
 * @brief
 */
template<typename T>
void printArray(const T* d_array, size_t size, const std::string& title = "",
                const std::string& sep = " ") noexcept;

/**
 * @brief
 */
template<typename T, int SIZE>
void printArray(const T (&d_array)[SIZE], const std::string& title = "",
                const std::string& sep = " ") noexcept;

/**
 * @brief
 */
template<typename T>
void printSymbol(const T& d_symbol, const std::string& title = "") noexcept;

/**
 * @brief row-major
 */
template<typename T>
void printMatrix(const T* d_matrix, size_t rows, size_t cols,
                 const std::string& title = "") noexcept;

/**
 * @brief row-major
 */
template<typename T>
void printMatrix(const T* d_matrix, size_t rows, size_t cols, size_t ld,
                 const std::string& title = "") noexcept;

/**
 * @brief column-major (blas and lapack compatibility)
 */
template<typename T>
void printMatrixCM(const T* d_matrix, size_t rows, size_t cols,
                   const std::string& title = "") noexcept;

/**
 * @brief column-major (blas and lapack compatibility)
 */
template<typename T>
void printMatrixCM(const T* d_matrix, size_t rows, size_t cols, size_t ld,
                   const std::string& title = "") noexcept;

//------------------------------------------------------------------------------
/**
 * @brief same as std::cout but on the GPU
 */
class Cout {
public:
    __device__ __forceinline__
    explicit Cout() {};
};

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, const char* string)       noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, uint64_t value)           noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, int64_t value)            noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, long long int value)      noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, long long unsigned value) noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, int value)                noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, unsigned value)           noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, short value)              noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, unsigned short value)     noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, char value)               noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, unsigned char value)      noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, float value)              noexcept;

__device__ __forceinline__
const Cout& operator<<(const Cout& obj, double value)             noexcept;

template<typename T>
__device__ __forceinline__
typename std::enable_if<std::is_pointer<T>::value, const Cout&>::type
operator<<(const Cout& obj, const T pointer) noexcept;

} // namespace gpu

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
void printfArray(T* array, int size);

template<typename T, int SIZE>
__device__ __forceinline__
void printfArray(T (&array)[SIZE]);

namespace warp {

template<typename T, int SIZE>
__device__ __forceinline__
void printfArray(T (&array)[SIZE]);

} // namespace warp
} // namespace xlib

#include "impl/PrintExt.i.cuh"
