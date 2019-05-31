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

#include "include/graph_concrete_visitors.hxx"



namespace nvgraph
{
  //------------------------- SubGraph Extraction: ----------------------
  //
  CsrGraph<int>* extract_subgraph_by_vertices(CsrGraph<int>& graph,
											  int* pV, size_t n, cudaStream_t stream)
  {
	return extract_from_vertex_subset<int, double>(graph, pV, n, stream);
  }

  MultiValuedCsrGraph<int, float>* extract_subgraph_by_vertices(MultiValuedCsrGraph<int, float>& graph, 
																int* pV, size_t n, cudaStream_t stream)
  {
	return static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(extract_from_vertex_subset<int, float>(graph, pV, n, stream));
  }

  MultiValuedCsrGraph<int, double>* extract_subgraph_by_vertices(MultiValuedCsrGraph<int, double>& graph, 
																int* pV, size_t n, cudaStream_t stream)
  {
	return static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(extract_from_vertex_subset<int, double>(graph, pV, n, stream));
  }

  CsrGraph<int>* extract_subgraph_by_edges(CsrGraph<int>& graph,
										   int* pV, size_t n, cudaStream_t stream)
  {
	return extract_from_edge_subset<int, double>(graph, pV, n, stream);
  }

  MultiValuedCsrGraph<int, float>* extract_subgraph_by_edges(MultiValuedCsrGraph<int, float>& graph,
															 int* pV, size_t n, cudaStream_t stream)
  {
	return static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(extract_from_edge_subset<int, float>(graph, pV, n, stream));
  }

  MultiValuedCsrGraph<int, double>* extract_subgraph_by_edges(MultiValuedCsrGraph<int, double>& graph,
															 int* pV, size_t n, cudaStream_t stream)
  {
	return static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(extract_from_edge_subset<int, double>(graph, pV, n, stream));
  }


  

	
  
}// end namespace nvgraph

