/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef B40C_LOG_MEM_BANKS
#    pragma push_macro("B40C_LOG_MEM_BANKS")
#    undef B40C_LOG_MEM_BANKS
#    define B40C_LOG_MEM_NEEDS_RESTORE
#  endif
#  ifdef B40C_LOG_WARP_THREADS
#    pragma push_macro("B40C_LOG_WARP_THREADS")
#    undef B40C_LOG_WARP_THREADS
#    define B40C_LOG_WARP_THREADS_NEEDS_RESTORE
#  endif
#  ifdef B40C_WARP_THREADS
#    pragma push_macro("B40C_WARP_THREADS")
#    undef B40C_WARP_THREADS
#    define B40C_WARP_THREADS_NEEDS_RESTORE
#  endif
#  ifdef TallyWarpVote
#    pragma push_macro("TallyWarpVote")
#    undef TallyWarpVote
#    define TallyWarpVote_NEEDS_RESTORE
#  endif
#  ifdef WarpVoteAll
#    pragma push_macro("WarpVoteAll")
#    undef WarpVoteAll
#    define WarpVoteAll_NEEDS_RESTORE
#  endif
#  ifdef FastMul
#    pragma push_macro("FastMul")
#    undef FastMul
#    define FastMul_NEEDS_RESTORE
#  endif
#endif // __GNUC__

// define the macros while we #include our version of cub
#define B40C_NS_PREFIX namespace cusp { namespace system { namespace cuda { namespace detail {
#define B40C_NS_POSTFIX               }                  }                }                  }

#include <cusp/system/cuda/detail/graph/b40c/graph/bfs/csr_problem.cuh>
#include <cusp/system/cuda/detail/graph/b40c/graph/bfs/enactor_hybrid.cuh>

// undef the macros
#undef B40C_NS_PREFIX
#undef B40C_NS_POSTFIX

// redefine the macros if they were defined previously

#if !defined(__GNUC__) || (THRUST_GCC_VERSION >= 40300)
#  ifdef B40C_LOG_MEM_BANKS_NEEDS_RESTORE
#    pragma pop_macro("B40C_LOG_MEM_BANKS")
#    undef B40C_LOG_MEM_BANKS_NEEDS_RESTORE
#  endif
#  ifdef B40C_LOG_WARP_THREADS_NEEDS_RESTORE
#    pragma pop_macro("B40C_LOG_WARP_THREADS")
#    undef B40C_LOG_WARP_THREADS_NEEDS_RESTORE
#  endif
#  ifdef B40C_WARP_THREADS_NEEDS_RESTORE
#    pragma pop_macro("B40C_WARP_THREADS")
#    undef B40C_WARP_THREADS_NEEDS_RESTORE
#  endif
#  ifdef TallyWarpVote_NEEDS_RESTORE
#    pragma pop_macro("TallyWarpVote")
#    undef TallyWarpVote_NEEDS_RESTORE
#  endif
#  ifdef WarpVoteAll_NEEDS_RESTORE
#    pragma pop_macro("WarpVoteAll")
#    undef WarpVoteAll_NEEDS_RESTORE
#  endif
#  ifdef FastMul_NEEDS_RESTORE
#    pragma pop_macro("FastMul")
#    undef FastMul_NEEDS_RESTORE
#  endif
#endif // __GNUC__
