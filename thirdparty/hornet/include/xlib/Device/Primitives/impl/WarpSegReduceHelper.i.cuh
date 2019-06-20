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
#include "Host/Numeric.hpp"

namespace xlib {
namespace detail {

#define WARP_SEG_REDUCE_32BIT(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r1;\n\t\t"                                         \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r1|p, %0, %1, %2, %3;\n\t\t"                   \
            "@p "#ASM_OP"."#ASM_T" %0, r1, %0;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value) : "r"(1 << STEP),                              \
              "r"(max_lane), "r"(member_mask));                                \
    }

//to check
#define WARP_SEG_REDUCE_64BIT(ASM_OP, ASM_T, ASM_CL)                          \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg .u32 lo;\n\t\t"                                              \
            ".reg .u32 hi;\n\t\t"                                              \
            ".reg ."#ASM_T" r1;\n\t\t"                                         \
            ".reg .pred p;\n\t\t"                                              \
            "mov.b64 {lo, hi}, %0;\n\t\t"                                      \
            "shfl.sync.down.b32 lo|p, lo, %1, %2, %3;\n\t\t"                   \
            "shfl.sync.down.b32 hi|p, hi, %1, %2, %3;\n\t\t"                   \
            "mov.b64 r1, {lo, hi};\n\t\t"                                      \
            "@p "#ASM_OP"."#ASM_T" %0, r1, %0;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value) : "r"(1 << STEP),                              \
              "r"(max_lane), "r"(member_mask));                                \
    }

/*
if (xlib::lane_id() + (1 << STEP) <= max_lane) {
    if (predicate)
        right += tmp;
    else
        left += tmp;
}
*/
#define WARP_SEG_REDUCE_MACRO3(ASM_OP, ASM_T, ASM_CL)                          \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r1;\n\t\t"                                         \
            ".reg .pred p, q, s;\n\t\t"                                        \
            "shfl.sync.down.b32 r1|p, %0, %2, %3, %4;\n\t\t"                   \
            "setp.ne.and.b32 s|q, %5, 0, p;\n\t\t"                             \
            "@s "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@q "#ASM_OP"."#ASM_T" %0, r1, %0;\n\t"                            \
            "}"                                                                \
            : "+r"(left), "+r"(right) : "r"(1 << STEP),                        \
              "r"(max_lane), "r"(member_mask), "r"(predicate));                \
    }

//==============================================================================

#define WARP_SEG_REDUCE_GEN2(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1;\n\t\t"                                     \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %2, %3, %4;\n\t\t"                     \
            "shfl.sync.down.b32 r1|p, %1, %2, %3, %4;\n\t\t"                   \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1])                       \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN3(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2;\n\t\t"                                 \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %3, %4, %5;\n\t\t"                     \
            "shfl.sync.down.b32 r1, %1, %3, %4, %5;\n\t\t"                     \
            "shfl.sync.down.b32 r2|p, %2, %3, %4, %5;\n\t\t"                   \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2])                                             \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN4(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3;\n\t\t"                             \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %4, %5, %6;\n\t\t"                     \
            "shfl.sync.down.b32 r1, %1, %4, %5, %6;\n\t\t"                     \
            "shfl.sync.down.b32 r2, %2, %4, %5, %6;\n\t\t"                     \
            "shfl.sync.down.b32 r3|p, %3, %4, %5, %6;\n\t\t"                   \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3])                       \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN5(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4;\n\t\t"                         \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %5, %6, %7;\n\t\t"                     \
            "shfl.sync.down.b32 r1, %1, %5, %6, %7;\n\t\t"                     \
            "shfl.sync.down.b32 r2, %2, %5, %6, %7;\n\t\t"                     \
            "shfl.sync.down.b32 r3, %3, %5, %6, %7;\n\t\t"                     \
            "shfl.sync.down.b32 r4|p, %4, %5, %6, %7;\n\t\t"                   \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4])                                             \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN6(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5;\n\t\t"                     \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %6, %7, %8;\n\t\t"                     \
            "shfl.sync.down.b32 r1, %1, %6, %7, %8;\n\t\t"                     \
            "shfl.sync.down.b32 r2, %2, %6, %7, %8;\n\t\t"                     \
            "shfl.sync.down.b32 r3, %3, %6, %7, %8;\n\t\t"                     \
            "shfl.sync.down.b32 r4, %4, %6, %7, %8;\n\t\t"                     \
            "shfl.sync.down.b32 r5|p, %5, %6, %7, %8;\n\t\t"                   \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5])                       \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN7(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6;\n\t\t"                 \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %7, %8, %9;\n\t\t"                     \
            "shfl.sync.down.b32 r1, %1, %7, %8, %9;\n\t\t"                     \
            "shfl.sync.down.b32 r2, %2, %7, %8, %9;\n\t\t"                     \
            "shfl.sync.down.b32 r3, %3, %7, %8, %9;\n\t\t"                     \
            "shfl.sync.down.b32 r4, %4, %7, %8, %9;\n\t\t"                     \
            "shfl.sync.down.b32 r5, %5, %7, %8, %9;\n\t\t"                     \
            "shfl.sync.down.b32 r6|p, %6, %7, %8, %9;\n\t\t"                   \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6])                                             \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN8(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7;\n\t\t"             \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %8, %9, %10;\n\t\t"                    \
            "shfl.sync.down.b32 r1, %1, %8, %9, %10;\n\t\t"                    \
            "shfl.sync.down.b32 r2, %2, %8, %9, %10;\n\t\t"                    \
            "shfl.sync.down.b32 r3, %3, %8, %9, %10;\n\t\t"                    \
            "shfl.sync.down.b32 r4, %4, %8, %9, %10;\n\t\t"                    \
            "shfl.sync.down.b32 r5, %5, %8, %9, %10;\n\t\t"                    \
            "shfl.sync.down.b32 r6, %6, %8, %9, %10;\n\t\t"                    \
            "shfl.sync.down.b32 r7|p, %7, %8, %9, %10;\n\t\t"                  \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7])                       \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN9(ASM_OP, ASM_T, ASM_CL)                            \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8;\n\t\t"         \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r1, %1, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r2, %2, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r3, %3, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r4, %4, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r5, %5, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r6, %6, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r7, %7, %9, %10, %11;\n\t\t"                   \
            "shfl.sync.down.b32 r8|p, %8, %9, %10, %11;\n\t\t"                 \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8])                                             \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN10(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;\n\t\t"     \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r1, %1, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r2, %2, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r3, %3, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r4, %4, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r5, %5, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r6, %6, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r7, %7, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r8, %8, %10, %11, %12;\n\t\t"                  \
            "shfl.sync.down.b32 r9|p, %9, %10, %11, %12;\n\t\t"                \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %9, r9, %9;\n\t"                            \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8]), "+"#ASM_CL(value[9])                       \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN11(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;\n\t\t"\
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r1, %1, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r2, %2, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r3, %3, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r4, %4, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r5, %5, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r6, %6, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r7, %7, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r8, %8, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r9, %9, %11, %12, %13;\n\t\t"                  \
            "shfl.sync.down.b32 r10|p, %10, %11, %12, %13;\n\t\t"              \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %9, r9, %9;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %10, r10, %10;\n\t"                         \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8]), "+"#ASM_CL(value[9]),                      \
              "+"#ASM_CL(value[10])                                            \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN12(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;" \
            "\n\t\t"                                                           \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r1, %1, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r2, %2, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r3, %3, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r4, %4, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r5, %5, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r6, %6, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r7, %7, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r8, %8, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r9, %9, %12, %13, %14;\n\t\t"                  \
            "shfl.sync.down.b32 r10, %10, %12, %13, %14;\n\t\t"                \
            "shfl.sync.down.b32 r11|p, %11, %12, %13, %14;\n\t\t"              \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %9, r9, %9;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %10, r10, %10;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %11, r11, %11;\n\t"                         \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8]), "+"#ASM_CL(value[9]),                      \
              "+"#ASM_CL(value[10]), "+"#ASM_CL(value[11])                     \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN13(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11," \
            "r12;\n\t\t"                                                       \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r1, %1, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r2, %2, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r3, %3, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r4, %4, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r5, %5, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r6, %6, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r7, %7, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r8, %8, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r9, %9, %13, %14, %15;\n\t\t"                  \
            "shfl.sync.down.b32 r10, %10, %13, %14, %15;\n\t\t"                \
            "shfl.sync.down.b32 r11, %11, %13, %14, %15;\n\t\t"                \
            "shfl.sync.down.b32 r12|p, %12, %13, %14, %15;\n\t\t"              \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %9, r9, %9;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %10, r10, %10;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %11, r11, %11;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %12, r12, %12;\n\t"                         \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8]), "+"#ASM_CL(value[9]),                      \
              "+"#ASM_CL(value[10]), "+"#ASM_CL(value[11]),                    \
              "+"#ASM_CL(value[12])                                            \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN14(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11," \
            "r12, r13;\n\t\t"                                                  \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r1, %1, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r2, %2, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r3, %3, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r4, %4, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r5, %5, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r6, %6, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r7, %7, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r8, %8, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r9, %9, %14, %15, %16;\n\t\t"                  \
            "shfl.sync.down.b32 r10, %10, %14, %15, %16;\n\t\t"                \
            "shfl.sync.down.b32 r11, %11, %14, %15, %16;\n\t\t"                \
            "shfl.sync.down.b32 r12, %12, %14, %15, %16;\n\t\t"                \
            "shfl.sync.down.b32 r13|p, %13, %14, %15, %16;\n\t\t"              \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %9, r9, %9;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %10, r10, %10;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %11, r11, %11;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %12, r12, %12;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %13, r13, %13;\n\t"                         \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8]), "+"#ASM_CL(value[9]),                      \
              "+"#ASM_CL(value[10]), "+"#ASM_CL(value[11]),                    \
              "+"#ASM_CL(value[12]), "+"#ASM_CL(value[13])                     \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN15(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11," \
            "r12, r13, r14;\n\t\t"                                             \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r1, %1, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r2, %2, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r3, %3, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r4, %4, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r5, %5, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r6, %6, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r7, %7, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r8, %8, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r9, %9, %15, %16, %17;\n\t\t"                  \
            "shfl.sync.down.b32 r10, %10, %15, %16, %17;\n\t\t"                \
            "shfl.sync.down.b32 r11, %11, %15, %16, %17;\n\t\t"                \
            "shfl.sync.down.b32 r12, %12, %15, %16, %17;\n\t\t"                \
            "shfl.sync.down.b32 r13, %13, %15, %16, %17;\n\t\t"                \
            "shfl.sync.down.b32 r14|p, %14, %15, %16, %17;\n\t\t"              \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %9, r9, %9;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %10, r10, %10;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %11, r11, %11;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %12, r12, %12;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %13, r13, %13;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %14, r14, %14;\n\t"                         \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8]), "+"#ASM_CL(value[9]),                      \
              "+"#ASM_CL(value[10]), "+"#ASM_CL(value[11]),                    \
              "+"#ASM_CL(value[12]), "+"#ASM_CL(value[13]),                    \
              "+"#ASM_CL(value[14])                                            \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }

#define WARP_SEG_REDUCE_GEN16(ASM_OP, ASM_T, ASM_CL)                           \
    _Pragma("unroll")                                                          \
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {            \
        asm(                                                                   \
            "{\n\t\t"                                                          \
            ".reg ."#ASM_T" r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11," \
            "r12, r13, r14, r15;\n\t\t"                                        \
            ".reg .pred p;\n\t\t"                                              \
            "shfl.sync.down.b32 r0, %0, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r1, %1, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r2, %2, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r3, %3, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r4, %4, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r5, %5, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r6, %6, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r7, %7, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r8, %8, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r9, %9, %16, %17, %18;\n\t\t"                  \
            "shfl.sync.down.b32 r10, %10, %16, %17, %18;\n\t\t"                \
            "shfl.sync.down.b32 r11, %11, %16, %17, %18;\n\t\t"                \
            "shfl.sync.down.b32 r12, %12, %16, %17, %18;\n\t\t"                \
            "shfl.sync.down.b32 r13, %13, %16, %17, %18;\n\t\t"                \
            "shfl.sync.down.b32 r14, %14, %16, %17, %18;\n\t\t"                \
            "shfl.sync.down.b32 r15|p, %15, %16, %17, %18;\n\t\t"              \
            "@p "#ASM_OP"."#ASM_T" %0, r0, %0;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %1, r1, %1;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %2, r2, %2;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %3, r3, %3;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %4, r4, %4;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %5, r5, %5;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %6, r6, %6;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %7, r7, %7;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %8, r8, %8;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %9, r9, %9;\n\t\t"                          \
            "@p "#ASM_OP"."#ASM_T" %10, r10, %10;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %11, r11, %11;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %12, r12, %12;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %13, r13, %13;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %14, r14, %14;\n\t\t"                       \
            "@p "#ASM_OP"."#ASM_T" %15, r15, %15;\n\t"                         \
            "}"                                                                \
            : "+"#ASM_CL(value[0]), "+"#ASM_CL(value[1]),                      \
              "+"#ASM_CL(value[2]), "+"#ASM_CL(value[3]),                      \
              "+"#ASM_CL(value[4]), "+"#ASM_CL(value[5]),                      \
              "+"#ASM_CL(value[6]), "+"#ASM_CL(value[7]),                      \
              "+"#ASM_CL(value[8]), "+"#ASM_CL(value[9]),                      \
              "+"#ASM_CL(value[10]), "+"#ASM_CL(value[11]),                    \
              "+"#ASM_CL(value[12]), "+"#ASM_CL(value[13]),                    \
              "+"#ASM_CL(value[14]), "+"#ASM_CL(value[15])                     \
            : "r"(1 << STEP), "r"(max_lane), "r"(member_mask));                \
    }


} // namespace detail
} // namespace xlib
