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

#include "Device/Util/DeviceProperties.cuh"

namespace xlib {

template<int WARP_SZ = 32>
struct WarpInclusiveScan {
    /// @cond
    static_assert(xlib::is_power2(WARP_SZ) &&
                  WARP_SZ >= 1 && WARP_SZ <= WARP_SIZE,
                  "WarpInclusiveScan : WARP_SZ must be a power of 2\
                                       and 2 <= WARP_SZ <= WARP_SIZE");
    /// @endcond

    template<typename T>
    __device__ __forceinline__
    static void add(T& value);

    template<typename T>
    __device__ __forceinline__
    static void add(T& value, T& total);

    template<typename T, typename R>
    __device__ __forceinline__
    static void add(T& value, R* total_ptr);
};

//------------------------------------------------------------------------------

/** \struct WarpExclusiveScan WarpScan.cuh
 *  \brief Support structure for warp-level exclusive scan
 *  <pre>
 *  Input:  1 2 3 4
 *  Output: 0 1 3 6 (10)
 *  </pre>
 *  \callergraph \callgraph
 *  @pre WARP_SZ must be a power of 2 in the range 1 &le; WARP_SZ &le; 32
 *  @tparam WARP_SZ     split the warp in WARP_SIZE / WARP_SZ groups and
 *                      perform the exclusive prefix-scan in each groups.
 *                      Require log2 ( WARP_SZ ) steps
 */
template<int WARP_SZ = 32>
struct WarpExclusiveScan {
    /// @cond
    static_assert(xlib::is_power2(WARP_SZ) &&
                  WARP_SZ >= 2 && WARP_SZ <= WARP_SIZE,
                  "WarpExclusiveScan : WARP_SZ must be a power of 2\
                             and 2 <= WARP_SZ <= WARP_SIZE");
    /// @endcond

    template<typename T>
     __device__ __forceinline__
    static void add(T& value);

    /** @fn void Add(T& value, T& total)
     *  \brief warp sum
     *  @param[in] value    input value of each thread
     *  @param[out] total   total sum of all values
     *  \warning only the last thread in the WARP_SZ group has the total sum
     */
    template<typename T>
     __device__ __forceinline__
    static void add(T& value, T& total);

    /** @fn void AddBcast(T& value, T& total)
     *  \brief warp sum
     *
     *  The result is broadcasted to all warp threads
     *  @param[in] value    input value of each thread
     *  @param[out] total   total sum of all values
     */
    /*template<typename T>
     __device__ __forceinline__
    static void addAll(T& value, T& total);*/

    /** @fn void Add(T& value, T* total_ptr)
     *  \brief warp sum
     *
     *  @warning only LaneID equal to (WARP_SZ - 1) stores the result
     *  @param[in] value    input value of each thread
     *  @param[out] total_ptr   ptr to store the sum of all values
     */
    template<typename T, typename R>
    __device__ __forceinline__
    static void add(T& value, R* total_ptr);

    /** @fn T AddAtom(T& value, T* total_ptr)
     *  \brief warp sum
     *
     *  Compute the warp-level prefix-sum of 'value' and add the total sum on
     *  'total_ptr' with an atomic operation.
     *  @warning only LaneID equal to (WARP_SZ - 1) stores the result
     *  @param[in] value    input value of each thread
     *  @param[out] total_ptr   ptr to store the sum of all values
     *  @return old value of total_ptr before atomicAdd operation
     */
    template<typename T, typename R>
     __device__ __forceinline__
    static T atomicAdd(T& value, R* total_ptr);

    template<typename T, typename R>
     __device__ __forceinline__
    static T atomicAdd(T& value, R* total_ptr, T& total);

    /** @fn void Add(T* in_ptr, T* total_ptr)
     *  \brief warp sum
     *
     *  Compute the warp-level prefix-sum of the first 32 values of 'in_ptr'
     *  and store the result in same locations. The total sum is stored in
     *  'total_ptr'.
     *  @warning only LaneID equal to (WARP_SZ - 1) stores the result
     *  @param[in,out] in_ptr  input/output values
     *  @param[out] total_ptr  ptr to store the sum of all values
     */
    template<typename T, typename R>
     __device__ __forceinline__
    static void add(T* in_ptr, R* total_ptr);
};

} // namespace xlib

#include "impl/WarpScan2.i.cuh"
