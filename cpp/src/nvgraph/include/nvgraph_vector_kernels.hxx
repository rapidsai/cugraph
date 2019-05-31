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
namespace nvgraph
{
	template <typename ValueType_>
	void nrm1_raw_vec (ValueType_* vec, size_t n, ValueType_* res, cudaStream_t stream = 0);

 	template <typename ValueType_>
	void fill_raw_vec (ValueType_* vec, size_t n, ValueType_ value, cudaStream_t stream = 0);

	template <typename ValueType_>
	void dump_raw_vec (ValueType_* vec, size_t n, int offset, cudaStream_t stream = 0);

	template <typename ValueType_>
	void dmv (size_t num_vertices, ValueType_ alpha, ValueType_* D, ValueType_* x, ValueType_ beta, ValueType_* y, cudaStream_t stream = 0);

	template<typename ValueType_>
	void copy_vec(ValueType_ *vec1, size_t n, ValueType_ *res, cudaStream_t stream = 0);

	template <typename ValueType_>
	void flag_zeros_raw_vec(size_t num_vertices, ValueType_* vec, int* flag, cudaStream_t stream = 0 );

	template <typename IndexType_, typename ValueType_>
	void set_connectivity( size_t n, IndexType_ root, ValueType_ self_loop_val, ValueType_ unreachable_val, ValueType_* res, cudaStream_t stream = 0);

} // end namespace nvgraph

