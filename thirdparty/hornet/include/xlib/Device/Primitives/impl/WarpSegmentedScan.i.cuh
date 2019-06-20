/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
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
#include "Device/WarpScan2.i.cuh"

namespace xlib {
namespace detail {

#define WARP_SEG_INCLUSIVE_SCAN(ASM_OP, ASM_T, ASM_CL)                         \
    const unsigned    max_lane = xlib::max_lane(mask);                         \
    const unsigned member_mask = xlib::member_mask<WARP_SZ>();                 \
    WARP_INCLUSIVE_SCAN_AUX(ASM_OP, ASM_T, ASM_CL)

//==============================================================================

template<unsigned WARP_SZ, typename T>
struct WarpInclusiveScanHelper {

    __device__ __forceinline__
    static void add(T& value, unsigned mask) {
        const unsigned    min_lane = xlib::min_lane(mask);
        const unsigned member_mask = member_mask<WARP_SZ>();
        int vlane = xlib::lane_id<xlib::WARP_SIZE>() :

        #pragma unroll
        for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_up(member_mask, value, STEP, WARP_SZ);
            if (vlane >= STEP && vlane - STEP >= min_lane)
                value += tmp;
        }
    }
};

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, int> {

    __device__ __forceinline__
    static void add(int& value, unsigned mask) {
        WARP_SEG_INCLUSIVE_SCAN(add, s32, r)
    }
};

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, unsigned> {

    __device__ __forceinline__
    static void add(unsigned& value, unsigned mask) {
        WARP_SEG_INCLUSIVE_SCAN(add, u32, r)
    }
};

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, float> {

    __device__ __forceinline__
    static void add(float& value, unsigned mask) {
        WARP_SEG_INCLUSIVE_SCAN(add, f32, f)
    }
};

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, double> {
    __device__ __forceinline__
    static void add(double& value, unsigned mask) {
        WARP_SEG_INCLUSIVE_SCAN(add, f64, d)
    }
};

#undef WARP_SEG_INCLUSIVE_SCAN

} // namespace detail
