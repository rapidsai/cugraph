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
class WCC {
    using color_t = vid_t;
public:
    explicit WCC(const GraphStd<vid_t, eoff_t>& graph) noexcept;
    ~WCC() noexcept;

    void run() noexcept;

    const std::vector<vid_t>& list() const noexcept;

    vid_t size() const noexcept;

    vid_t largest() const noexcept;

    vid_t num_trivial() const noexcept;

    const color_t* result() const noexcept;

    void print() const noexcept;

    void print_histogram() const noexcept;

    void print_statistics() const noexcept;

    bool compare_result(const color_t* result) const noexcept;
private:
    const color_t NO_COLOR = std::numeric_limits<color_t>::max();

    const GraphStd<vid_t, eoff_t>& _graph;
    xlib::Queue<vid_t>             _queue;
    std::vector<vid_t>             _wcc_vector;
    color_t*                       _color { nullptr };
};

} // namespace graph
