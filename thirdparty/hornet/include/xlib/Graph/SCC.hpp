/**
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

#include "Graph/GraphStd.hpp"
#include "Host/Classes/Bitmask.hpp"
#include "Host/Classes/Queue.hpp"
#include <vector>

namespace graph {

template<typename vid_t, typename eoff_t>
class SCC {
public:
    explicit SCC(const GraphStd<vid_t, eoff_t>& _graph) noexcept;
    ~SCC() noexcept;

    void run() noexcept;

    void reset() noexcept;

    const std::vector<vid_t>& list() const noexcept;

    vid_t size() const noexcept;

    vid_t largest() const noexcept;

    vid_t num_trivial() const noexcept;

    const vid_t* result() const noexcept;

    void print() const noexcept;

    void print_histogram() const noexcept;
private:
#if defined(STACK)
    struct StackNode {
        vid_t  source;
        eoff_t i;
    };

    using StackT = xlib::QueueStack<StackNode, 4 * xlib::MB / sizeof(StackNode),
                                    xlib::QueuePolicy::LIFO>

    StackT _stack;
#endif

    using color_t = vid_t;

    const color_t NO_COLOR = std::numeric_limits<color_t>::max();
    const   vid_t NO_INDEX = std::numeric_limits<vid_t>::max();
    const   vid_t MAX_LINK = std::numeric_limits<vid_t>::max();

    const GraphStd<vid_t, eoff_t>& _graph;
    std::vector<vid_t>             _scc_vector;

    xlib::Bitmask                               _in_stack;
    xlib::Queue<vid_t, xlib::QueuePolicy::LIFO> _queue;


    vid_t*   _low_link   { nullptr };
    vid_t*   _indices    { nullptr };
    color_t* _color      { nullptr };
    vid_t    _curr_index { 0 };
    int      _scc_index  { 0 };

    void single_scc(vid_t source) noexcept;
};

} // namespace graph
