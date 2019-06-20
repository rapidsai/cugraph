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
#include "Static/KatzCentrality/Katz.cuh"

namespace hornets_nest {

struct KatzDynamicData : KatzData {
    KatzDynamicData() = default;
    KatzDynamicData(KatzData data) : KatzData(data) {}
    __device__
    void operator=(const KatzDynamicData&) { assert(false); }

    TwoLevelQueue<vid_t> active_queue;

    ulong_t* new_paths_curr;
    ulong_t* new_paths_prev;
    int*     active;
    int      iteration_static;
};

class KatzCentralityDynamic : public StaticAlgorithm<HornetGraph> {
public:
    KatzCentralityDynamic(HornetGraph& hornet,
                          HornetGraph& inverted_graph,
                          int max_iteration, int K,
                          degree_t max_degree);

    KatzCentralityDynamic(HornetGraph& hornet,
                          int max_iteration, int K,
                          degree_t max_degree);

    ~KatzCentralityDynamic();

    void setInitParametersUndirected(int max_iteration, int K,
                                     degree_t max_degree);
    void setInitParametersDirected(int max_iteration, int K,
                                   degree_t max_degree);

    void reset()    override;
    void run()      override;
    void run_static();
    void release()  override;
    bool validate() override;

    int get_iteration_count();

    void copyKCToHost(double* host_array);
    void copyNumPathsToHost(ulong_t* host_array);

    void batchUpdateInserted(BatchUpdate& batch_update);
    void batchUpdateDeleted(BatchUpdate&  batch_update);

private:
    HostDeviceVar<KatzDynamicData> hd_katzdata;

    HornetGraph&                  inverted_graph;
    load_balancing::BinarySearch load_balancing;
    KatzCentrality              kc_static;
    bool is_directed;

    void processUpdate(BatchUpdate& batch_update, bool is_insert);
};

} // namespace hornets_nest
