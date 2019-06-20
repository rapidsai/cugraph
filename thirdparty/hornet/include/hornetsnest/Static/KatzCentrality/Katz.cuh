/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
 *   ogreen@gatech.edu
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

#include "HornetAlg.hpp"

namespace hornets_nest {

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;
//using HornetGraph = gpu::Csr<EMPTY, EMPTY>;

using ulong_t = long long unsigned;

struct KatzData {
    ulong_t*  num_paths_data;
    ulong_t** num_paths; // Will be used for dynamic graph algorithm which
                          // requires storing paths of all iterations.

    ulong_t*  num_paths_curr;
    ulong_t*  num_paths_prev;

    double*   KC;
    double*   lower_bound;
    double*   upper_bound;

    double alpha;
    double alphaI; // Alpha to the power of I  (being the iteration)

    double lower_bound_const;
    double upper_bound_const;

    int K;

    int max_degree;
    int iteration;
    int max_iteration;

    int num_active;    // number of active vertices at each iteration
    int num_prev_active;
    int nV;

    bool*   is_active;
    double* lower_bound_unsorted;
    double* lower_bound_sorted;
    int*    vertex_array_unsorted; // Sorting
    int*    vertex_array_sorted;   // Sorting
};

// Label propogation is based on the values from the previous iteration.
class KatzCentrality : public StaticAlgorithm<HornetGraph> {
public:
    KatzCentrality(HornetGraph& hornet, int max_iteration,
                   int K, int max_degree, bool is_static = true);
    ~KatzCentrality();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    int get_iteration_count();

    void copyKCToHost(double* host_array);
    void copyNumPathsToHost(ulong_t* host_array);

    KatzData katz_data();

private:
    load_balancing::BinarySearch load_balancing;
    HostDeviceVar<KatzData>     hd_katzdata;
    ulong_t**                   h_paths_ptr;
    bool                        is_static;

    void printKMostImportant();
};

} // hornetAlgs namespace
