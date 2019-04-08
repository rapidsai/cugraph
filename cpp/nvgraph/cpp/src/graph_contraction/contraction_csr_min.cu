/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <graph_contracting_visitor.hxx>

namespace nvgraph
{
  //------------------------- Graph Contraction: ----------------------
  //
  CsrGraph<int>* contract_graph_csr_min(CsrGraph<int>& graph,
                                int* pV, size_t n,
                                cudaStream_t stream,
                                const int& VCombine,
                                const int& VReduce,
                                const int& ECombine,
                                const int& EReduce)
  {
    return contract_from_aggregates_t<int, double, SemiRingFctrSelector<Min, double>::FctrType >(graph, pV, n, stream,
                                                                                                       static_cast<SemiRingFunctorTypes>(VCombine),
                                                                                                       static_cast<SemiRingFunctorTypes>(VReduce),
                                                                                                       static_cast<SemiRingFunctorTypes>(ECombine),
                                                                                                       static_cast<SemiRingFunctorTypes>(EReduce));
  }
 
}
