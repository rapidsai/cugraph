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

#pragma once

#include <climits>

//Used in nvgraph.h
#define TRAVERSAL_DEFAULT_ALPHA 15
#define TRAVERSAL_DEFAULT_BETA 18

#include "nvgraph_error.hxx"
#include "2d_partitioning.h"

namespace nvgraph {
	template<typename GlobalType, typename LocalType, typename ValueType>
	class Bfs2d {
	private:
		Matrix2d<GlobalType, LocalType, ValueType>* M;

		bool directed;
		bool deterministic;
		GlobalType alpha;
		GlobalType beta;

		// edgemask, distances, predecessors are set/read by users - using Vectors
		bool useEdgeMask;
		bool computeDistances;
		bool computePredecessors;
		int32_t vertices_bmap_size;
		VertexData2D<GlobalType, LocalType, LocalType>* distances;
		VertexData2D<GlobalType, LocalType, GlobalType>* predecessors;

		//Working data
		VertexData2D<GlobalType, LocalType, int32_t>* frontier_bmap;
		VertexData2D<GlobalType, LocalType, int32_t>* visited_bmap;
		VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>* frontier;
		VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>* trim_frontier;
		VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>* frontierSize;
		VertexData2D_Unbuffered<GlobalType, LocalType, int8_t>* degreeFlags;
		std::vector<LocalType> frontierSize_h;
		VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>* exSumDegree;
		VertexData2D_Unbuffered<GlobalType, LocalType, int8_t>* exSumStorage;
		VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>* bucketOffsets;
		std::vector<LocalType> frontierDegree_h;

		// Output locations
		GlobalType* distances_out;
		GlobalType* predecessors_out;

		NVGRAPH_ERROR setup();

		void clean();

	public:
		virtual ~Bfs2d(void) {
			clean();
		};

		Bfs2d(Matrix2d<GlobalType, LocalType, ValueType>* _M,
				bool _directed,
				GlobalType _alpha,
				GlobalType _beta) :
						M(_M),
						directed(_directed),
						alpha(_alpha),
						beta(_beta){
			distances = NULL;
			predecessors = NULL;
			frontier_bmap = NULL;
			visited_bmap = NULL;
			setup();
		}

		NVGRAPH_ERROR configure(GlobalType *distances, GlobalType *predecessors);

		NVGRAPH_ERROR traverse(GlobalType source_vertex);

		//Used only for benchmarks
		NVGRAPH_ERROR traverse(GlobalType *source_vertices, int32_t nsources);
	};
} // end namespace nvgraph

