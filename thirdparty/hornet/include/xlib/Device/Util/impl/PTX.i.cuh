/**
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
 */
#include "Device/Util/DeviceProperties.cuh" //xlib::WARP_SIZE

namespace xlib {

template<unsigned WARP_SZ>
__device__ __forceinline__
unsigned lane_id() {
    unsigned ret;
    asm ("mov.u32 %0, %laneid;" : "=r"(ret) );
    return WARP_SZ == xlib::WARP_SIZE ? ret : ret % WARP_SZ;
}

__device__ __forceinline__
unsigned lanemask_eq() {
    unsigned ret;
    asm("mov.u32 %0, %lanemask_eq;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__
unsigned lanemask_lt() {
    unsigned ret;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__
unsigned lanemask_le() {
    unsigned ret;
    asm("mov.u32 %0, %lanemask_le;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__
unsigned lanemask_gt() {
    unsigned ret;
    asm("mov.u32 %0, %lanemask_gt;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__
unsigned lanemask_ge() {
    unsigned ret;
    asm("mov.u32 %0, %lanemask_ge;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__
void thread_exit() {
    asm("exit;");
}

//==============================================================================

__device__ __forceinline__
unsigned SM_id() {
    unsigned ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__
unsigned num_warps() {
    unsigned ret;
    asm("mov.u32 %0, %nwarpid;" : "=r"(ret));
    return ret;
}

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) < 4, unsigned>::type
__msb(T word) {
    const unsigned mask = (1u << (sizeof(T) * 8u)) - 1;
    return __msb(reinterpret_cast<unsigned>(word) & mask);
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 4, unsigned>::type
__msb(T word) {
    unsigned ret;
    asm ("bfind.u32 %0, %1;" : "=r"(ret) :
         "r"(reinterpret_cast<unsigned&>(word)));
    return ret;
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, unsigned>::type
__msb(T dword) {
    unsigned ret;
    asm ("bfind.u64 %0, %1;" : "=r"(ret) :
         "l"(reinterpret_cast<uint64_t&>(dword)));
    return ret;
}

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) <= 4, unsigned>::type
__be(T word, unsigned pos) {
    assert(pos <= sizeof(T) * 8);
    unsigned ret;
    asm ("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) :
         "r"(reinterpret_cast<unsigned&>(word)), "r"(pos), "r"(1u));
    return ret;
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__be(T dword, unsigned pos) {
    assert(pos <= sizeof(T) * 8);
    uint64_t ret;
    asm ("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) :
         "l"(reinterpret_cast<uint64_t&>(dword)), "r"(pos), "r"(1u));
    return ret;
}

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 4, T>::type
__bi(T word, unsigned pos) {
    unsigned ret;
    asm ("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(ret)
          : "r"(1u), "r"(reinterpret_cast<unsigned&>(word)), "r"(pos), "r"(1u));
    return reinterpret_cast<T&>(ret);
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, T>::type
__bi(T dword, unsigned pos) {
    uint64_t ret;
    asm ("bfi.b64 %0, %1, %2, %3, %4;" :
            "=l"(ret) : "l"(reinterpret_cast<uint64_t&>(dword)),
            "l"(1ull), "r"(pos), "r"(1u));
    return reinterpret_cast<T&>(ret);
}

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) <= 4, unsigned>::type
__bfe(T word, unsigned pos, unsigned length) {
    assert(pos <= sizeof(T) * 8);
    unsigned ret;
    asm ("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) :
         "r"(reinterpret_cast<unsigned&>(word)), "r"(pos), "r"(length));
    return ret;
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__bfe(T dword, unsigned pos, unsigned length) {
    assert(pos <= sizeof(T) * 8);
    uint64_t ret;
    asm ("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) :
         "l"(reinterpret_cast<uint64_t&>(dword)),
         "r"(pos), "r"(length));
    return ret;
}

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 4>::type
__bfi(T& word, unsigned bitmask, unsigned pos, unsigned length) {
    assert(pos + length <= sizeof(T) * 8);
    asm ("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(word) : "r"(bitmask),
         "r"(word), "r"(pos), "r"(length));
}

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8>::type
__bfi(T& dword, uint64_t bitmask, unsigned pos, unsigned length) {
    assert(pos + length <= sizeof(T) * 8);
    asm ("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(dword) : "l"(bitmask),
         "l"(dword), "r"(pos), "r"(length));
}

} // namespace xlib
