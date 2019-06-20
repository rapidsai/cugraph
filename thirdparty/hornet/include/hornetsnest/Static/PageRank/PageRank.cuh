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

using HornetGPU = gpu::Hornet<EMPTY, EMPTY>;

using pr_t = float;

struct PrData {
	pr_t* prev_pr;
	pr_t* curr_pr;
	pr_t* abs_diff;

	pr_t* reduction_out;
	pr_t* contri;

	int   iteration;
	int   iteration_max;
	int   nV;
	pr_t  threshold;
	pr_t  damp;
	pr_t  normalized_damp;
};

// Label propogation is based on the values from the previous iteration.
class StaticPageRank : public StaticAlgorithm<HornetGPU> {
public:
    StaticPageRank(HornetGPU& hornet,
	            	int  iteration_max = 20,
	            	pr_t     threshold = 0.001f,
	            	pr_t          damp = 0.85f,
		        	bool  isUndirected = false);
    ~StaticPageRank();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

	void setInputParameters(int iteration_max = 20,
                            pr_t    threshold = 0.001f,
                            pr_t         damp = 0.85f,
		                    bool isUndirected  = false);

	int get_iteration_count();

	const pr_t* get_page_rank_score_host();

	void printRankings();

    PrData pr_data();

private:
    load_balancing::BinarySearch 	load_balancing;
    HostDeviceVar<PrData>       	hd_prdata;
    pr_t*                       	host_page_rank { nullptr };
    bool 							isUndirected;
};

} // hornets_nest namespace
