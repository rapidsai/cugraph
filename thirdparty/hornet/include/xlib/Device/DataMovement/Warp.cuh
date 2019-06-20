#pragma once

namespace xlib {
namespace warp {

template<typename T>
__device__ __forceinline__
void stride_load_smem(const T* __restrict__ d_in,
                      int                   num_items,
                      T*       __restrict__ smem_out);

} // namespace warp
} // namespace xlib

#include "impl/Warp.i.cuh"
