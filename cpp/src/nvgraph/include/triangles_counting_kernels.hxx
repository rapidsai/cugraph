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

#include <triangles_counting.hxx>

namespace nvgraph
{

namespace triangles_counting
{

template <typename T>
void tricnt_bsh(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, size_t bmld, cudaStream_t stream);
template <typename T>
void tricnt_wrp(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, unsigned int *bmap_d, size_t bmld, cudaStream_t stream);
template <typename T>
void tricnt_thr(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, cudaStream_t stream);
template <typename T>
void tricnt_b2b(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, unsigned int *bmapL0_d, size_t bmldL0, unsigned int *bmapL1_d, size_t bmldL1, cudaStream_t stream);

template <typename T>
uint64_t reduce(uint64_t *v_d, T n, cudaStream_t stream);
template <typename T>
void create_nondangling_vector(const T *roff, T *p_nonempty, T *n_nonempty, size_t n, cudaStream_t stream);

void myCudaMemset(unsigned long long *p, unsigned long long v, long long n, cudaStream_t stream);

} // namespace triangles_counting

} // namespace nvgraph
