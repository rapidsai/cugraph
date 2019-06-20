/**
 * @brief Device-wide Binary Search load balancing
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

#include "Queue/TwoLevelQueue.cuh"

namespace hornets_nest {
/**
 * @brief The namespace provides all load balancing methods to traverse vertices
 */
namespace load_balancing {

/**
 * @brief The class implements the BinarySearch load balancing
 */
class BinarySearch {
public:
    /**
     * @brief Default costructor
     * @param[in] hornet Hornet instance
     */
    template<typename HornetClass>
    explicit BinarySearch(const HornetClass& hornet,
                          const float work_factor = 2.0f) noexcept;

    /**
     * @brief Decostructor
     */
    ~BinarySearch() noexcept;

    /**
     * @brief Traverse the edges in a vertex queue (C++11-Style API)
     * @tparam Operator function to apply at each edge
     * @param[in] queue input vertex queue
     * @param[in] op struct/lambda expression that implements the operator
     * @remark    all algorithm-dependent data must be capture by `op`
     * @remark    the Operator typename must implement the method
     *            `void operator()(Vertex, Edge)` or the lambda expression
     *            `[=](Vertex, Edge){}`
     */
     template<typename HornetClass, typename Operator>
     void apply(const HornetClass& hornet,
                const vid_t*       d_input,
                int                num_vertices,
                const Operator&    op) const noexcept;

    template<typename HornetClass, typename Operator>
    void apply(const HornetClass& hornet, const Operator& op) const noexcept;

private:
    static const unsigned BLOCK_SIZE = 128;

    xlib::CubExclusiveSum<int> prefixsum;

    int* _d_work { nullptr };
    const size_t _work_size;
};

} // namespace load_balancing
} // namespace hornets_nest

#include "LoadBalancing/BinarySearch.i.cuh"
