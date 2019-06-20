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
#include "Static/PageRank/PageRank.cuh"

//namespace hornets_nest {
namespace hornets_nest {

using ulong_t = long long unsigned;

struct PrDynamicData : PrData {
    PrDynamicData() = default;
    PrDynamicData(PrData data) : PrData(data) {}
    //__device__
    //void operator=(const PrDynamicData&) {}

    //TwoLevelQueue<vid_t> active_queue;
    TwoLevelQueue<vid_t> queue1;
    TwoLevelQueue<vid_t> queue2;
    TwoLevelQueue<vid_t> queue3;
    TwoLevelQueue<vid_t> queueDlt;

    pr_t epsilon; //determinant for enqueuing
    int* visited; //size of NV
    int* visitedDlt; //size of NV
    int* usedOld; //size of NV
    pr_t* diffPR;
    pr_t* delta;
};


//class PageRankDynamic : public StaticAlgorithm<HornetGraph> {
class PageRankDynamic : public StaticAlgorithm<HornetGPU> {
public:
//    PageRankDynamic(HornetGraph& hornet,
//                    HornetGraph& inverted_graph,
//                    int max_iteration, int K,
//                    degree_t max_degree);

//    PageRankDynamic(HornetGraph& hornet,
//                    int max_iteration, int K,
//                    degree_t max_degree);

//    PageRankDynamic(HornetGPU& hornet, //HornetGraph& hornet,
//                    HornetGPU& inverted_graph, //HornetGraph& inverted_graph,
//                    int  iteration_max,
//                    pr_t threshold,
//                    pr_t damp);

    PageRankDynamic(HornetGPU& hornet, //HornetGraph& hornet,
                    int  iteration_max,
                    pr_t threshold,
                    pr_t damp);

    ~PageRankDynamic();

    void setInitParametersUndirected(int max_iteration, pr_t threshold,
                                     pr_t damp);
    void setInitParametersDirectedint(int max_iteration, pr_t threshold,
                                     pr_t damp);

    void reset()    override;
    //void run()      override;
    void run() override;
    void release()  override;
    bool validate() override;

    //void Reset();
#if 0
//    void RecomputeInsertionContriUndirected();
    void RecomputeContri();
    void RecomputeDeletionContriUndirected();
    void UpdateDeltaAndMove();
    void UpdateDeltaAndCopy();
    void UpdateContributionsUndirected();
    void RemoveContributionsUndirected();
    void PrevEqualCurr();
    void ResetCurr();
#endif

    int get_iteration_count();

    void copyPRToHost(double* host_array);
    void copyNumPathsToHost(ulong_t* host_array);

    void batchUpdateInserted(BatchUpdate& batch_update);
    void batchUpdateDeleted(BatchUpdate&  batch_update);

    void processUpdate(BatchUpdate& batch_update, bool is_insert);

private:
    HostDeviceVar<PrDynamicData> hd_prdata;

//    HornetGraph&             inverted_graph;
    //HornetGPU&                 inverted_graph;
//    load_balancing::BinarySearch load_balancing;
    load_balancing::BinarySearch load_balancing;
//    PageRank                   pr_static;
    StaticPageRank           pr_static;
    bool                       is_directed;

};

} // namespace hornets_nest

