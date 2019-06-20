/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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

namespace hornets_nest {

/**
 * @brief
 */
const bool PREFER_FASTER_UPDATE = false;

/**
 * @brief minimum number of edges for a **block**
 * @remark `MIN_EDGES_PER_BLOCK` must be a power of two
 */
const size_t MIN_EDGES_PER_BLOCK = 1;

/**
 * @brief number of edges for a **BlockArray**
 * @remark `EDGES_PER_BLOCKARRAY` must be a power of two
 */
const size_t EDGES_PER_BLOCKARRAY = 1 << 23;

///@brief Eanble B+Tree container for BitTree
//#define B_PLUS_TREE

///@brief Eanble RedBlack-Tree container for BitTree
//#define RB_TREE

//------------------------------------------------------------------------------

static_assert(xlib::is_power2(MIN_EDGES_PER_BLOCK)  &&
              xlib::is_power2(EDGES_PER_BLOCKARRAY) &&
              MIN_EDGES_PER_BLOCK <= EDGES_PER_BLOCKARRAY,
              "Memory Management Constrains");

} // namespace hornets_nest
