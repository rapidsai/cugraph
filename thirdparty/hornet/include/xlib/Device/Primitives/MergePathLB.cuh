/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date January, 2018
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
 */
namespace xlib {

template<unsigned ITEMS_PER_BLOCK, typename T>
__global__
void mergePathLBPartition(const T* __restrict__ d_prefixsum,
                          int                   prefixsum_size,
                          T                     max_value,
                          int                   num_merge,
                          int*     __restrict__ d_partitions,
                          int                   num_partitions);

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK,
         typename T, typename Lambda>
__device__ __forceinline__
void mergePathLB(const int*    d_partitions,
                 int           num_partitions,
                 const T*      d_prefixsum,
                 int           prefixsum_size,
                 void*         smem,
                 const Lambda& lambda);

//--------------------------------------------------------------------------

namespace merge_path_lb {

static constexpr int BLOCK_SIZE      = 128;
static constexpr int THREAD_ITEMS    = 11;
static constexpr int ITEMS_PER_BLOCK = BLOCK_SIZE * THREAD_ITEMS;

template<typename T, typename Lambda, typename... TArgs>
void run(const T*      d_prefixsum,
         int           prefixsum_size,
         T             last_value,
         const Lambda& lambda,
         TArgs*...     forward_args) noexcept;

template<typename T, typename Lambda, typename... TArgs>
void run_kernel_fusion(const T*      d_prefixsum,
                       int           prefixsum_size,
                       T             last_value,
                       const Lambda& lambda,
                       TArgs*...     forward_args) noexcept;

template<typename T, typename Lambda, typename... TArgs>
void run(const T*      d_prefixsum,
         int           prefixsum_size,
         T             last_value,
         int*          d_partitions,
         const Lambda& lambda,
         TArgs*...     forward_args) noexcept;

template<typename T, typename Lambda, typename... TArgs>
void run_kernel_fusion(const T*      d_prefixsum,
                       int           prefixsum_size,
                       T             last_value,
                       int*          d_partitions,
                       const Lambda& lambda,
                       TArgs*...     forward_args) noexcept;

} // namespace merge_path_lb

//--------------------------------------------------------------------------

template<typename T>
class MergePathLB {
    static constexpr int BLOCK_SIZE      = 128;
    static constexpr int THREAD_ITEMS    = 11;
    static constexpr int ITEMS_PER_BLOCK = BLOCK_SIZE * THREAD_ITEMS;
public:
    explicit MergePathLB() = default;

    explicit MergePathLB(int max_prefixsum_size,
                         T   max_last_value) noexcept;

    explicit MergePathLB(const T* d_prefixsum,
                         int      prefixsum_size,
                         T        last_value) noexcept;

    ~MergePathLB() noexcept;

    void init(int max_prefixsum_size,
              T   max_last_value) noexcept;

    void init(const T* d_prefixsum,
              int      prefixsum_size,
              T        last_value) noexcept;

    template<typename Lambda>
    void run(const T*      d_prefixsum,
             int           prefixsum_size,
             T             max_last_value,
             const Lambda& lambda) const noexcept;

    template<typename Lambda>
    void run_kernel_fusion(const T*      d_prefixsum,
                           int           prefixsum_size,
                           T             last_value,
                           const Lambda& lambda) const noexcept;

    template<typename Lambda>
    void run(const Lambda& lambda) const noexcept;

private:
    const int*  _d_partitions     { nullptr };
    const T*    _d_prefixsum      { nullptr };
    int         _prefixsum_size   { 0 };
    int         _num_partitions   { 0 };
    int         _num_merge_blocks { 0 };
};

} // namespace xlib

#include "impl/MergePathLB.i.cuh"
