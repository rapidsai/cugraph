/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
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

namespace xlib {

template<int NUM_THREADS, typename = void>
class ThreadGroup;

template<int NUM_THREADS>
using EnableBlock = typename std::enable_if<(NUM_THREADS > 32)>::type;

template<int NUM_THREADS>
using EnableWarp = typename std::enable_if<(NUM_THREADS > 32)>::type;

template<int NUM_THREADS>
class ThreadGroup<NUM_THREADS, EnableBlock<NUM_THREADS>> {
public:
    explicit ThreadGroup();

    int rank() const;

    void sync() const;

    constexpr int size() const;

    bool any(int pred) const;

    bool all(int pred) const;

    unsigned count(int pred) const;

    ThreadGroup(const ThreadGroup& thread_group) = delete;

private:
    int _rank;
};

//------------------------------------------------------------------------------

template<int NUM_THREADS>
class ThreadGroup<NUM_THREADS, EnableWarp<NUM_THREADS>> {
public:
    explicit ThreadGroup();

    int rank() const;

    int warp_base() const;

    void sync() const;


    bool any(int pred) const;

    bool all(int pred) const;

    unsigned match_any() const;

    unsigned match_all() const;

    unsigned count(int pred) const;

    unsigned ballot(int pred) const;

    template<typename T>
    T warp_broadcast(T value, int pred) const;


    constexpr int size() const;

    ThreadGroup(const ThreadGroup& thread_group) = delete;

private:
    int      _rank;
    unsigned _mask;
    const unsigned _active_mask;
};






#define BLOCK_MACRO(RETURN_TYPE)                                               \
    template<int NUM_THREADS>                                                  \
    __device__ __forceinline__                                                 \
    RETURN_TYPE ThreadGroup<NUM_THREADS, EnableBlock<NUM_THREADS>>             \

#define WARP_MACRO(RETURN_TYPE)                                                \
    template<int NUM_THREADS>                                                  \
    __device__ __forceinline__                                                 \
    RETURN_TYPE ThreadGroup<NUM_THREADS, EnableWarp<NUM_THREADS>>              \

BLOCK_MACRO()::ThreadGroup() : _rank(threadIdx.x) {}

BLOCK_MACRO(int)::rank() {
    return _rank;
}

BLOCK_MACRO(void)::sync() {
    __sync_warp();
    __syncthreads();
}

//==============================================================================
//==============================================================================

WARP_MACRO(int)::_set_rank() {
    return (NUM_THREADS == xlib::WARP_SIZE) ? xlib::lane_id() :
           (NUM_THREADS == 1) ? 0 : xlib::lane_id() % xlib::WARP_SIZE;
}


WARP_MACRO(unsigned)::_set_mask() {
    const unsigned MASK = 0xFFFFFFFF >> (xlib::WARP_SIZE - NUM_THREADS);
    return (NUM_THREADS == xlib::WARP_SIZE) ?
                0xFFFFFFFF : MASK << (xlib::lane_id() & ~(NUM_THREADS - 1));
}

//------------------------------------------------------------------------------

WARP_MACRO()::ThreadGroup() : _rank(_set_rank()), _mask(_set_mask()) {}

WARP_MACRO(int)::rank() {
    return _rank;
}

WARP_MACRO()::warp_base() {
    return xlib::lane_id() & ~(NUM_THREADS - 1u);
}

WARP_MACRO()::block_base() {
    return threadIdx.x & ~(NUM_THREADS - 1u);
}

WARP_MACRO(void)::sync() {
#if __CUDA_ARCH__ >= 700
    __sync_warp(_active_mask);
#endif
}

WARP_MACRO(bool)::any(int pred) const {
    return (NUM_THREADS == 1) ? pred :
           (NUM_THREADS == xlib::WARP_SIZE) ? __any_sync(_active_mask, pred) :
           (__ballot_sync(_active_mask, pred) & _mask) == _mask;
}

WARP_MACRO(bool)::all(bool pred) const {
    return (NUM_THREADS == 1) ? pred :
           (NUM_THREADS == xlib::WARP_SIZE) ? __all_sync(_active_mask, pred) :
           (__ballot_sync(_active_mask, pred) & _mask) == _mask;
}

WARP_MACRO(unsigned)::ballot(bool pred) const {
    return (NUM_THREADS == 1) ? pred << xlib::lane_id() :
          (NUM_THREADS == xlib::WARP_SIZE) ? __ballot_sync(_active_mask, pred) :
          (__ballot_sync(_active_mask, pred) & _mask);
}

template<typename T>
WARP_MACRO(T)::warp_broadcast(T value, int pred) const {
    unsigned elected_lane = xlib::__msb(__ballot_sync(_active_mask, pred));
    return xlib::shfl(_active_mask, value, elected_lane);
}

WARP_MACRO(unsigned)::match_any(T value) const {
    return __match_any_sync(_active_mask, value);
}

WARP_MACRO(unsigned)::match_all() const {
    int pred;
    return __match_all_sync(_active_mask, value, &pred);
}

} // namespace xlib
