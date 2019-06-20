/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
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

#include <type_traits>
#include "Device/Util/DeviceProperties.cuh"  //xlib::WARP_SIZE

/** \namespace PTX
 *  provide simple interfaces for low-level PTX instructions
 */
namespace xlib {
/*
#if defined(_WIN32) || defined(__i386__)
    //#define ASM_PTR "r"
    #pragma error(ERR_START "32-bit architectures are not supported" ERR_END)
#elif defined(__x86_64__) || defined(__ia64__) ||                              \
      defined(_WIN64) || defined(__ppc64__)
    #define ASM_PTR "l"
#endif*/

// ---------------------------- THREAD PTX -------------------------------------

/**
 *  @brief return the lane ID within the current warp
 *
 *  Provide the thread ID within the current warp (called lane).
 *  \return identification ID in the range 0 &le; ID &le; 31
 */
template<unsigned WARP_SZ = xlib::WARP_SIZE>
__device__ __forceinline__
unsigned lane_id();

/** @fn unsigned int LaneMaskEQ()
 *  @brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return 1 << laneid
 */
__device__ __forceinline__
unsigned lanemask_eq();

/** @fn unsigned int LaneMaskLT()
 *  @brief 32-bit mask with bits set in positions less than the thread's lane
 *         number in the warp
 *  \return (1 << laneid) - 1
 */
__device__ __forceinline__
unsigned lanemask_lt();

/** @fn unsigned int LaneMaskLE()
 *  @brief 32-bit mask with bits set in positions less than or equal to the
 *         thread's lane number in the warp
 *  \return (1 << (laneid + 1)) - 1
 */
__device__ __forceinline__
unsigned lanemask_le();

/** @fn unsigned int LaneMaskGT()
 *  @brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return ~((1 << (laneid + 1)) - 1)
 */
__device__ __forceinline__
unsigned lanemask_gt();

/** @fn unsigned int LaneMaskGE()
 *  @brief 32-bit mask with bits set in positions greater than or equal to the
 *         thread's lane number in the warp
 *  \return ~((1 << laneid) - 1)
 */
__device__ __forceinline__
unsigned lanemask_ge();

/**
 *  @brief terminate the current thread
 */
__device__ __forceinline__
void thread_exit();

// --------------------------------- MATH --------------------------------------

__device__ __forceinline__
unsigned SM_id();

__device__ __forceinline__
unsigned num_warps();

// --------------------------------- MATH --------------------------------------

/**
 *  @brief sum three operands with one instruction
 *
 *  Sum three operand with one instruction. Only in Maxwell architecture
 *  IADD3 is implemented in hardware, otherwise involves multiple instructions.
 *  \return x + y + z
 */
__device__ __forceinline__
unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z);

/** @fn unsigned int __msb(unsigned int word)
 *  @brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 31.
 *  0xFFFFFFFF if no bit is found.
 */

/** @fn unsigned int __msb(unsigned long long int dword)
 *  @brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 63.
 *          0xFFFFFFFF if no bit is found.
 */
//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) < 4, unsigned>::type
__msb(T word);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 4, unsigned>::type
__msb(T word);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, unsigned>::type
__msb(T dword);

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) <= 4, unsigned>::type
__be(T word, unsigned pos);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__be(T dword, unsigned pos);

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 4, T>::type
__bi(T word, unsigned pos);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, T>::type
__bi(T dword, unsigned pos);

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) <= 4, unsigned>::type
__bfe(T word, unsigned pos, unsigned length);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__bfe(T dword, unsigned pos, unsigned length);

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 4>::type
__bfi(T& word, unsigned bitmask, unsigned pos, unsigned length);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8>::type
__bfi(T& dword, long long unsigned bitmask, unsigned pos, unsigned length);

//------------------------------------------------------------------------------

template<typename T, int _DATA_SIZE>
struct WordArray {
    const unsigned WORD_SIZE = sizeof(T) * 8;
    const unsigned DATA_SIZE = _DATA_SIZE;
public:
    __device__ __forceinline__
    WordArray(T* array) : _array(array) {}

    __device__ __forceinline__
    T operator[](int index) const {
        unsigned     start = index * DATA_SIZE;
        unsigned       end = start + DATA_SIZE;
        unsigned start_pos = start / WORD_SIZE;
        unsigned mod_start = start % WORD_SIZE;
        auto         data1 = __bfe(_array[start_pos], mod_start, DATA_SIZE);
        if (start != end) {
            unsigned   head = WORD_SIZE - mod_start;
            unsigned remain = DATA_SIZE - head;
            auto      data2 = __bfe(_array[start + 1], 0, remain);
            return (data2 << head) | data1;
        }
        return data1;
    }

    __device__ __forceinline__
    void insert(T data, int index) {
        unsigned     start = index * DATA_SIZE;
        unsigned       end = start + DATA_SIZE;
        unsigned start_pos = start / WORD_SIZE;
        unsigned mod_start = start % WORD_SIZE;
        auto         data1 = __bfi(_array[start_pos], data, mod_start,
                                   DATA_SIZE);
        if (start != end) {
            unsigned   head = WORD_SIZE - mod_start;
            unsigned remain = DATA_SIZE - head;
            auto      data2 = __bfi(_array[start + 1], data >> head, 0, remain);
        }
    }

private:
    T* _array;
};

} // namespace xlib

#include "impl/PTX.i.cuh"
