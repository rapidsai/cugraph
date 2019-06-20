/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
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

#include "BasicTypes.hpp"

namespace hornets_nest {
/**
 * @brief
 */
namespace load_balancing {

template<unsigned VW_SIZE>
class VertexBased {
    static_assert(xlib::is_power2(VW_SIZE) && VW_SIZE <= xlib::WARP_SIZE,
                 "VW_SIZE must be a power of two such that 0 <= VW_SIZE <= 32");
public:
    VertexBased() = default;

    template<typename T>
    VertexBased(const T&) {}

    template<typename HornetClass, typename Operator>
    void apply(const HornetClass& hornet,
               const vid_t*       d_input,
               int                num_vertices,
               Operator&&         op) const noexcept;

    template<typename HornetClass, typename Operator>
    void apply(const HornetClass& hornet, Operator&& op) const noexcept;

    template<typename HornetClass, typename Operator>
    void applyVertexPairs(const HornetClass& hornet, Operator&& op) const noexcept;

private:
    static const unsigned BLOCK_SIZE = 128;
};

using  VertexBased1 = VertexBased<1>;
using  VertexBased2 = VertexBased<2>;
using  VertexBased4 = VertexBased<4>;
using  VertexBased8 = VertexBased<8>;
using VertexBased16 = VertexBased<16>;
using VertexBased32 = VertexBased<32>;

} // namespace load_balancing
} // namespace hornets_nest

#include "VertexBased.i.cuh"
