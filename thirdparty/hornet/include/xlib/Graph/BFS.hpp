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
#include <array>
#include <vector>

namespace graph {

template<typename vid_t, typename eoff_t>
class BFS {
public:
    using dist_t = int;
    enum { PARENT = 0, PEER = 1, VALID = 2, NOT_VALID = 3 };

    explicit BFS(const GraphStd<vid_t, eoff_t>& graph) noexcept;
    ~BFS() noexcept;

    void run(vid_t source) noexcept;
    void run(const vid_t* sources, int num_sources) noexcept;
    void reset() noexcept;

    const dist_t* result() const noexcept;

    vid_t  visited_nodes() const noexcept;
    eoff_t visited_edges() const noexcept;
    dist_t eccentricity()  const noexcept;

    std::vector<std::array<vid_t, 4>> statistics(vid_t source) noexcept;

    vid_t radius() noexcept;
    vid_t diameter() noexcept;
private:
    const dist_t INF = std::numeric_limits<dist_t>::max();

    const GraphStd<vid_t, eoff_t>&  _graph;
    xlib::Bitmask                   _bitmask;
    xlib::Queue<vid_t>              _queue;
    dist_t*                         _distances   { nullptr };
    int                             _num_visited { 0 };
    bool                            _reset       { false };
};

} // namespace graph
