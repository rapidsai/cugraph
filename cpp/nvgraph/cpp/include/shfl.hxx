 /*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "sm_utils.h"

namespace nvgraph{

    __device__ __forceinline__ float shflFPAdd(
        float           input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        float output;
        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input), "r"(mask));

#else
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input));
#endif

 		return output;

    }

    //incorporate into cusparse and try to remove
    // Inclusive prefix scan step speciliazed for summation of doubles
    __device__ __forceinline__ double shflFPAdd(
        double          input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        double output;

        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
            "    shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p add.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.up.b32 lo|p, lo, %2, %3;"
            "    shfl.up.b32 hi|p, hi, %2, %3;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p add.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input));
#endif

        return output;
    }

    __device__ __forceinline__ float shflFPMin(
        float           input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        float output;
        //if (threadIdx.x + blockDim.x*blockIdx.x < 4)device_printf("Thread = %d %f\n", threadIdx.x + blockDim.x*blockIdx.x, input);
        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
            "  @p min.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p min.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input));
#endif
        return output;
    }

    //incorporate into cusparse and try to remove
    // Inclusive prefix scan step speciliazed for summation of doubles
    __device__ __forceinline__ double shflFPMin(
        double          input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        double output;

        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
            "    shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p min.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.up.b32 lo|p, lo, %2, %3;"
            "    shfl.up.b32 hi|p, hi, %2, %3;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p min.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input));
#endif

        return output;
    }

    __device__ __forceinline__ float shflFPMax(
        float           input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        float output;
        //if (threadIdx.x + blockDim.x*blockIdx.x < 4)device_printf("Thread = %d %f\n", threadIdx.x + blockDim.x*blockIdx.x, input);
        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
            "  @p max.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p max.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input));
#endif
        return output;
   
        //return output;
    }

    //incorporate into cusparse and try to remove
    // Inclusive prefix scan step speciliazed for summation of doubles
    __device__ __forceinline__ double shflFPMax(
        double          input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        double output;

        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
            "    shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p max.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.up.b32 lo|p, lo, %2, %3;"
            "    shfl.up.b32 hi|p, hi, %2, %3;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p max.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input));
#endif

        return output;
    }

    __device__ __forceinline__ float shflFPOr(
        float           input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        float output;
        //if (threadIdx.x + blockDim.x*blockIdx.x < 4)device_printf("Thread = %d %f\n", threadIdx.x + blockDim.x*blockIdx.x, input);
        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
            "  @p or.b32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p or.b32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(firstLane), "f"(input));
#endif
   
        return output;
    }

    __device__ __forceinline__ double shflFPOr(
        double          input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        double output;

        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
            "    shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p or.b64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.up.b32 lo|p, lo, %2, %3;"
            "    shfl.up.b32 hi|p, hi, %2, %3;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p or.b64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(firstLane), "d"(input));
#endif

        return output;
    }
//Need to write correct instructions in asm for the operation -log(exp(-x) + exp(-y))
 __device__ __forceinline__ float shflFPLog(
        float           input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        float output;
        float expinput = expf(-input); //this must be shuffled and adding
        float baseChange = log2(expf(1.0)); //for change of base formaula
        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "  @p lg2.approx.f32  %0, r0;" //convert to natural logarithm!!
            //add another variable for e in change of base compute log_e(x) = log_2(x) / log_2(e) 
            "  @p neg.f32  %0, r0;"
            "}"
            : "=f"(output) : "f"(expinput), "r"(offset), "r"(firstLane), "f"(expinput), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "  @p lg2.approx.f32  %0, r0;" //convert to natural logarithm!!
            //add another variable for e in change of base compute log_e(x) = log_2(x) / log_2(e) 
            "  @p neg.f32  %0, r0;"
            "}"
            : "=f"(output) : "f"(expinput), "r"(offset), "r"(firstLane), "f"(expinput));
#endif
        return output;
    }
//check this!!
    __device__ __forceinline__ double shflFPLog(
        double          input,              //Calling thread's input item.
        int             firstLane,         //Index of first lane in segment
        int             offset,             //Upstream offset to pull from
        int             mask = DEFAULT_MASK) // lane mask for operation
    {
        double output;
        double expinput = exp(-input);
        double baseChange = log2(exp(1.0));//divide byt his

        // Use predicate set from SHFL to guard against invalid peers
#if USE_CG
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"        
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
            "    shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p add.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
           // "  @p lg2.approx.f32  %0, r0;" //f64 not supported!!
            "  @p neg.f64  %0, r0;"
            "}"
            : "=d"(output) : "d"(expinput), "r"(offset), "r"(firstLane), "d"(expinput), "r"(mask));
#else
        asm volatile(
            "{"
            "  .reg .f64 r0;"
            "  .reg .pred p;"        
            "  {"
            "    .reg .u32 lo;"
            "    .reg .u32 hi;"
            "    mov.b64 {lo, hi}, %1;"
            "    shfl.up.b32 lo|p, lo, %2, %3;"
            "    shfl.up.b32 hi|p, hi, %2, %3;"
            "    mov.b64 r0, {lo, hi};"
            "  }"
            "  @p add.f64 r0, r0, %4;"
            "  mov.f64 %0, r0;"
           // "  @p lg2.approx.f32  %0, r0;" //f64 not supported!!
            "  @p neg.f64  %0, r0;"
            "}"
            : "=d"(output) : "d"(expinput), "r"(offset), "r"(firstLane), "d"(expinput));
#endif

        return output;
    }

} //end namespace

