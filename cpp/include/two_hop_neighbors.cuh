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
/** ---------------------------------------------------------------------------*
 * @brief Functions for computing the two hop neighbor pairs of a graph
 *
 * @file two_hop_neighbors.cuh
 * ---------------------------------------------------------------------------**/

#include <cugraph.h>

#define TWO_HOP_BLOCK_SIZE 512

template <typename IndexType>
struct degree_iterator {
	IndexType* offsets;
	degree_iterator(IndexType* _offsets) : offsets(_offsets){};
	__host__ __device__
	IndexType operator[](IndexType place){
		return offsets[place + 1] - offsets[place];
	}
};

template <typename It, typename IndexType>
struct deref_functor {
	It iterator;
	deref_functor(It it) : iterator(it) {};
	__host__ __device__
	IndexType operator()(IndexType in) {
		return iterator[in];
	}
};
