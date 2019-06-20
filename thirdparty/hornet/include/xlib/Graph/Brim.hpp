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

#include "Graph/GraphWeight.hpp"
#include "Host/Classes/Bitmask.hpp"
#include "Host/Classes/Queue.hpp"

namespace graph {

template<typename vid_t, typename eoff_t, typename weight_t>
class Brim {
    using potential_t = weight_t;
public:
    explicit Brim(const GraphWeight<vid_t, eoff_t, weight_t>& _graph) noexcept;

    ~Brim() noexcept;

    void run() noexcept;

    void reset() noexcept;

    const potential_t* result() const noexcept;

    void set_player_TH(vid_t index) noexcept;

    void print_potential() const noexcept;

    void print_potential_to_file() const;

    bool check() const noexcept;

	void check_from_file(const std::string& file);

private:
    const GraphWeight<vid_t, eoff_t, weight_t>& _graph;

    xlib::Bitmask                               _in_queue;
    xlib::Queue<vid_t, xlib::QueuePolicy::LIFO> _queue;

    const potential_t TT          { std::numeric_limits<potential_t>::max() };
    potential_t       Mg          { 0 };
	int               _player_TH  { 0 };
	potential_t*      _potentials { nullptr };
	int*              _counters   { nullptr };

    void lift_count(vid_t vertex_id) noexcept;

    void findMg() noexcept;

    bool is_player0(vid_t vertex_id) const noexcept;

    auto minus(potential_t a, potential_t b) const noexcept;
};

} // namespace graph
