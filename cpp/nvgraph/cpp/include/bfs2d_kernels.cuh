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
#include <cub/cub.cuh>
#include "nvgraph_error.hxx"

#define MAXBLOCKS 65535
#define WARP_SIZE 32
#define INT_SIZE 32
#define FILL_QUEUE_DIMX 256
#define COMPUTE_BUCKET_OFFSETS_DIMX 512
#define TOP_DOWN_EXPAND_DIMX 256
#define TOP_DOWN_BUCKET_SIZE 32
#define NBUCKETS_PER_BLOCK (TOP_DOWN_EXPAND_DIMX/TOP_DOWN_BUCKET_SIZE)
#define TOP_DOWN_BATCH_SIZE 2
#define MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD (TOP_DOWN_BUCKET_SIZE - 1)

using namespace nvgraph;
namespace bfs_kernels {

	struct popCount : public thrust::unary_function<int,int> {
	  __device__
	  int operator()(int x) const
	  {
	    return __popc(x);
	  }
	};

	template<typename >
	struct vec_t {
		typedef int4 vec4;
		typedef int2 vec2;
	};

	template<>
	struct vec_t<int> {
		typedef int4 vec4;
		typedef int2 vec2;
		static const int max = INT_MAX;
	};

	template<>
	struct vec_t<long long int> {
		typedef longlong4 vec4;
		typedef longlong2 vec2;
		static const long long int max = LLONG_MAX;
	};

	struct BitwiseOr {
		template<typename T>
		__host__  __device__  __forceinline__ T operator()(const T &a, const T &b) const {
			return (a | b);
		}
	};

	struct predMerge {
		template<typename T>
		__host__  __device__  __forceinline__ T operator()(const T &a, const T &b) const {
			if (a != -1 && b != -1)
				return min(a, b);
			if (a != -1)
				return a;
			if (b != -1)
				return b;
			return -1;
		}
	};

	__forceinline__ __device__ int getMaskNRightmostBitSet(int n) {
		if (n == INT_SIZE)
			return (~0);
		int mask = (1 << n) - 1;
		return mask;
	}

	__forceinline__ __device__ int getMaskNLeftmostBitSet(int n) {
		if (n == 0)
			return 0;
		int mask = ~((1 << (INT_SIZE - n)) - 1);
		return mask;
	}

	/**
	 * Finds the position of the next non-zero bit in the given value. The value is
	 * re-written with the found bit unset.
	 * @param val The integer to find the next non-zero bit in.
	 * @return The position of the next non-zero bit
	 */
	__forceinline__ __device__ int getNextNonZeroBit(int32_t& val) {
		int ibit = __ffs(val) - 1;
		val &= ~(1 << ibit);

		return ibit;
	}

	template<typename IndexType>
	__device__ IndexType binsearch_maxle(const IndexType *vec,
														const IndexType val,
														IndexType low,
														IndexType high) {
		while (true) {
			if (low == high)
				return low; //we know it exists
			if ((low + 1) == high)
				return (vec[high] <= val) ? high : low;

			IndexType mid = low + (high - low) / 2;

			if (vec[mid] > val)
				high = mid - 1;
			else
				low = mid;

		}
	}

	template<typename IndexType>
	class degreeIterator: public std::iterator<std::input_iterator_tag, IndexType, size_t,
			IndexType*, IndexType> {
		IndexType* offsets;
		size_t pos;
		public:
		__host__ __device__ degreeIterator(IndexType* _offsets) :
				offsets(_offsets), pos(0) {
		}
		__host__ __device__ degreeIterator(IndexType* _offsets, size_t _pos) :
				offsets(_offsets), pos(_pos) {
		}
		__host__  __device__ IndexType operator[](int loc) {
			return offsets[loc + 1] - offsets[loc];
		}
		__host__  __device__ IndexType operator*() {
			return offsets[pos + 1] - offsets[pos];
		}
		__host__  __device__ degreeIterator operator+(int inc) {
			degreeIterator it(offsets, pos + inc);
			return it;
		}
	};

	template<typename IndexType>
	size_t getCubExclusiveSumStorageSize(IndexType n) {
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		IndexType *d_in = NULL, *d_out = NULL;
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
		return temp_storage_bytes;
	}

	template<typename IndexType>
	size_t getCubSelectFlaggedStorageSize(IndexType n) {
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		IndexType *d_in = NULL, *d_out = NULL, *size_out = NULL;
		degreeIterator<IndexType> degreeIt(NULL);
		cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, degreeIt, d_out, size_out, n);
		return temp_storage_bytes;
	}

	/**
	 * Takes in the bitmap frontier and outputs the frontier as a queue of ids.
	 * @param bmap Pointer to the bitmap
	 * @param bmap_nints The number of ints used to store the bitmap
	 * @param n The number of bits in the bitmap
	 * @param outputQueue Pointer to the output queue
	 * @param output_cnt Pointer to counter for output size
	 */
	template<typename IndexType>
	__global__ void convert_bitmap_to_queue_kernel(int32_t *bmap,
																	IndexType bmap_nints,
																	IndexType n,
																	IndexType *outputQueue,
																	IndexType *output_cnt) {
		typedef cub::BlockScan<int, FILL_QUEUE_DIMX> BlockScan;
		__shared__ typename BlockScan::TempStorage scan_temp_storage;

		// When filling the output queue, we use output_cnt to know where to write in the queue
		// (equivalent of int off = atomicAddd(unvisited_cnt, 1)) We will actually do only one
		// atomicAdd per block - we first do a scan, then call one atomicAdd, and store the common
		// offset for the block in common_block_offset
		__shared__ IndexType common_block_offset;

		// We don't want threads divergence in the loop (we're going to call __syncthreads)
		// Using a block-only dependent in the condition of the loop
		for (IndexType block_v_idx = blockIdx.x * blockDim.x;
				block_v_idx < bmap_nints;
				block_v_idx += blockDim.x * gridDim.x) {

			// Index of bmap that this thread will compute
			IndexType v_idx = block_v_idx + threadIdx.x;

			int thread_int = (v_idx < bmap_nints) ? bmap[v_idx] : 0;

			// The last int can be only partially valid
			// If we are indeed taking care of the last int in this thread,
			// We need to first disable the inactive bits (vertices >= n)
			if (v_idx == (bmap_nints - 1)) {
				int active_bits = n - (INT_SIZE * v_idx);
				int inactive_bits = INT_SIZE - active_bits;
				int mask = getMaskNLeftmostBitSet(inactive_bits);
				thread_int &= (~mask);
			}

			//Counting number of set bits in this int
			int n_in_int = __popc(thread_int);
			int thread_offset;

			// We will need to write n_unvisited_in_int unvisited vertices to the unvisited queue
			// We ask for that space when computing the block scan, that will tell where to write those
			// vertices in the queue, using the common offset of the block (see below)
			BlockScan(scan_temp_storage).ExclusiveSum(n_in_int, thread_offset);

			// Last thread knows how many vertices will be written to the queue by this block
			// Asking for that space in the queue using the global count, and saving the common offset
			if (threadIdx.x == (FILL_QUEUE_DIMX - 1)) {
				IndexType total = thread_offset + n_in_int;
				common_block_offset = atomicAdd(output_cnt, total);
			}

			// syncthreads for two reasons :
			// - we need to broadcast common_block_offset
			// - we will reuse scan_temp_storage (cf CUB doc)
			__syncthreads();

			IndexType current_index = common_block_offset + thread_offset;
			int nvertices_to_write = n_in_int;

			// getNextNonZeroBit uses __ffs, which gives least significant bit set
			// which means that as long as n_unvisited_in_int is valid,
			// we will use valid bits

			while (nvertices_to_write > 0) {
				if (nvertices_to_write >= 4 && (current_index % 4) == 0) {
					typename vec_t<IndexType>::vec4 vec_v;

					vec_v.x = v_idx * INT_SIZE + getNextNonZeroBit(thread_int);
					vec_v.y = v_idx * INT_SIZE + getNextNonZeroBit(thread_int);
					vec_v.z = v_idx * INT_SIZE + getNextNonZeroBit(thread_int);
					vec_v.w = v_idx * INT_SIZE + getNextNonZeroBit(thread_int);

					typename vec_t<IndexType>::vec4 *unvisited_i4 = reinterpret_cast<typename vec_t<
							IndexType>::vec4*>(&outputQueue[current_index]);
					*unvisited_i4 = vec_v;

					current_index += 4;
					nvertices_to_write -= 4;
				}
				else if (nvertices_to_write >= 2 && (current_index % 2) == 0) {
					typename vec_t<IndexType>::vec2 vec_v;

					vec_v.x = v_idx * INT_SIZE + getNextNonZeroBit(thread_int);
					vec_v.y = v_idx * INT_SIZE + getNextNonZeroBit(thread_int);

					typename vec_t<IndexType>::vec2 *unvisited_i2 = reinterpret_cast<typename vec_t<
							IndexType>::vec2*>(&outputQueue[current_index]);
					*unvisited_i2 = vec_v;

					current_index += 2;
					nvertices_to_write -= 2;
				} else {
					IndexType v = v_idx * INT_SIZE + getNextNonZeroBit(thread_int);

					outputQueue[current_index] = v;

					current_index += 1;
					nvertices_to_write -= 1;
				}

			}
		}
	}

	template<typename IndexType>
	void convert_bitmap_to_queue(int32_t *bmap,
											IndexType bmap_nints,
											IndexType n,
											IndexType *outputQueue,
											IndexType *output_cnt,
											cudaStream_t stream) {
		dim3 grid, block;
		block.x = FILL_QUEUE_DIMX;
		grid.x = min((IndexType) MAXBLOCKS, (bmap_nints + block.x - 1) / block.x);
		convert_bitmap_to_queue_kernel<<<grid, block, 0, stream>>>(bmap,
																						bmap_nints,
																						n,
																						outputQueue,
																						output_cnt);
		cudaCheckError()
					;
	}

	/**
	 * Kernel to compute bucket offsets for load balancing main top-down expand kernel
	 * @param frontier_degrees_exclusive_sum Exclusive sum of the local degrees of the frontier
	 * elements.
	 * @param bucket_offsets Output location for the bucket offsets.
	 * @param frontier_size Number of elements in the frontier.
	 * @param total_degree Total local degree of frontier elements.
	 */
	template<typename IndexType>
	__global__ void compute_bucket_offsets_kernel(const IndexType *frontier_degrees_exclusive_sum,
																	IndexType *bucket_offsets,
																	const IndexType frontier_size,
																	IndexType total_degree) {
		IndexType end = ((total_degree - 1 + TOP_DOWN_EXPAND_DIMX) / TOP_DOWN_EXPAND_DIMX
				* NBUCKETS_PER_BLOCK + 1);

		for (IndexType bid = blockIdx.x * blockDim.x + threadIdx.x;
				bid <= end;
				bid += gridDim.x * blockDim.x) {

			IndexType eid = min(bid * TOP_DOWN_BUCKET_SIZE, total_degree - 1);

			bucket_offsets[bid] = binsearch_maxle(frontier_degrees_exclusive_sum,
																eid,
																(IndexType) 0,
																frontier_size - 1);

		}
	}

	/**
	 * Wrapper function around compute_bucket_offsets_kernel.
	 * @param cumul Exclusive sum of the local degrees of the frontier elements.
	 * @param bucket_offsets Output location for the bucket offsets.
	 * @param frontier_size Number of elements in the frontier.
	 * @param total_degree Total local degree of frontier elements.
	 * @param m_stream Stream to use for execution.
	 */
	template<typename IndexType>
	void compute_bucket_offsets(IndexType *cumul,
											IndexType *bucket_offsets,
											IndexType frontier_size,
											IndexType total_degree,
											cudaStream_t m_stream) {
		dim3 grid, block;
		block.x = COMPUTE_BUCKET_OFFSETS_DIMX;

		grid.x = min((IndexType) MAXBLOCKS,
							((total_degree - 1 + TOP_DOWN_EXPAND_DIMX) / TOP_DOWN_EXPAND_DIMX
									* NBUCKETS_PER_BLOCK + 1 + block.x - 1) / block.x);

		compute_bucket_offsets_kernel<<<grid, block, 0, m_stream>>>(cumul,
																						bucket_offsets,
																						frontier_size,
																						total_degree);
		cudaCheckError();
	}

	/**
	 * Kernel for setting the degree of each frontier element.
	 * @param frontier_degree Output to store frontier degrees.
	 * @param frontier The frontier elements.
	 * @param degreeIt Iterator providing the degree of a given vertex ID
	 * @param n The number of elements in the frontier.
	 */
	template<typename IndexType, typename InputIterator>
	__global__ void set_frontier_degree_kernel(IndexType *frontier_degree,
																IndexType *frontier,
																InputIterator degreeIt,
																IndexType n) {
		for (IndexType idx = blockDim.x * blockIdx.x + threadIdx.x;
				idx < n;
				idx += gridDim.x * blockDim.x) {
			IndexType u = frontier[idx];
			frontier_degree[idx] = degreeIt[u];
		}
	}

	/**
	 * Wrapper function for calling set_frontier_degree_kernel
	 * @param frontier_degree Output to store frontier degrees.
	 * @param frontier The frontier elements.
	 * @param degreeIt Iterator providing the degree of a given vertex ID.
	 * @param n The number of elements in the frontier.
	 * @param m_stream The stream to use for the kernel call.
	 */
	template<typename IndexType, typename InputIterator>
	void set_frontier_degree(IndexType *frontier_degree,
										IndexType *frontier,
										InputIterator degreeIt,
										IndexType n,
										cudaStream_t m_stream) {
		dim3 grid, block;
		block.x = 256;
		grid.x = min((n + block.x - 1) / block.x, (IndexType) MAXBLOCKS);
		set_frontier_degree_kernel<<<grid, block, 0, m_stream>>>(frontier_degree,
																					frontier,
																					degreeIt,
																					n);
		cudaCheckError();
	}

	/**
	 * Kernel for setting the degree of each frontier element.
	 * @param frontier_degree Output to store frontier degrees.
	 * @param frontier The frontier elements.
	 * @param degreeIt Iterator providing the degree of a given vertex ID
	 * @param n The number of elements in the frontier.
	 */
	template<typename IndexType, typename InputIterator>
	__global__ void set_degree_flags_kernel(int8_t *degree_flags,
														 IndexType *frontier,
														 InputIterator degreeIt,
														 IndexType n) {
		for (IndexType idx = blockDim.x * blockIdx.x + threadIdx.x;
				idx < n;
				idx += gridDim.x * blockDim.x) {
			IndexType u = frontier[idx];
			degree_flags[idx] = (degreeIt[u] == 0) ? 0 : 1;
		}
	}

	/**
	 * Wrapper function for calling set_frontier_degree_kernel
	 * @param frontier_degree Output to store frontier degrees.
	 * @param frontier The frontier elements.
	 * @param degreeIt Iterator providing the degree of a given vertex ID.
	 * @param n The number of elements in the frontier.
	 * @param m_stream The stream to use for the kernel call.
	 */
	template<typename IndexType, typename InputIterator>
	void set_degree_flags(int8_t *degree_flags,
								 IndexType *frontier,
								 InputIterator degreeIt,
								 IndexType n,
								 cudaStream_t m_stream) {
		dim3 grid, block;
		block.x = 256;
		grid.x = min((n + block.x - 1) / block.x, (IndexType) MAXBLOCKS);
		set_degree_flags_kernel<<<grid, block, 0, m_stream>>>(degree_flags,
																				frontier,
																				degreeIt,
																				n);
		cudaCheckError();
	}

	/**
	 * Kernel for globalizing an array of ids using a given offset. Values of -1 remain
	 * unchanged, other values are incremented by the offset.
	 * @param ids The array of ids to globalize (input and output)
	 * @param offset The offset to be applied to each id.
	 * @param n The number of ids in the array.
	 */
	template<typename IndexType>
	__global__ void globalize_ids_kernel(IndexType *ids,
													 IndexType offset,
													 IndexType n) {
		for (IndexType idx = blockDim.x * blockIdx.x + threadIdx.x;
				idx < n;
				idx += gridDim.x * blockDim.x) {
			IndexType id = ids[idx];
			ids[idx] = (id == -1) ? -1 : id + offset;
		}
	}

	/**
	 * Wrapper function for calling globalize_ids_kernel
	 * @param ids The array of ids to globalize (input and output)
	 * @param offset The offset to be applied to each id.
	 * @param n The number of ids in the array.
	 * @param m_stream The stream to use for the kernel call.
	 */
	template<typename IndexType>
	void globalize_ids(IndexType *ids,
							 IndexType offset,
							 IndexType n,
							 cudaStream_t m_stream) {
		dim3 grid, block;
		block.x = 256;
		grid.x = min((n + block.x - 1) / block.x, (IndexType) MAXBLOCKS);
		globalize_ids_kernel<<<grid, block, 0, m_stream>>>(ids, offset, n);
		cudaCheckError();
	}

	template<typename IndexType, typename GlobalType>
	__global__ void topdown_expand_kernel(	const IndexType *row_ptr,
														const IndexType *col_ind,
														const IndexType *frontier,
														const IndexType frontier_size,
														const IndexType totaldegree,
														const IndexType max_items_per_thread,
														const IndexType lvl,
														int *frontier_bmap,
														const IndexType *frontier_degrees_exclusive_sum,
														const IndexType *frontier_degrees_exclusive_sum_buckets_offsets,
														int *visited_bmap,
														IndexType *distances,
														GlobalType *predecessors) {
		__shared__ IndexType shared_buckets_offsets[TOP_DOWN_EXPAND_DIMX - NBUCKETS_PER_BLOCK + 1];
		__shared__ IndexType shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX + 1];

		IndexType block_offset = (blockDim.x * blockIdx.x) * max_items_per_thread;
		IndexType n_items_per_thread_left = (totaldegree - block_offset + TOP_DOWN_EXPAND_DIMX - 1)
				/ TOP_DOWN_EXPAND_DIMX;

//		if (threadIdx.x == 0)
//			printf("n_items_per_thread_left=%d max_items_per_thread=%d\n", n_items_per_thread_left, max_items_per_thread);
		n_items_per_thread_left = min(max_items_per_thread, n_items_per_thread_left);

		for (;
				(n_items_per_thread_left > 0) && (block_offset < totaldegree);
				block_offset += MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD * blockDim.x,
						n_items_per_thread_left -= MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD) {

			// In this loop, we will process batch_set_size batches
			IndexType nitems_per_thread = min(n_items_per_thread_left,
															(IndexType) MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD);

			// Loading buckets offset (see compute_bucket_offsets_kernel)

			if (threadIdx.x < (nitems_per_thread * NBUCKETS_PER_BLOCK + 1))
				shared_buckets_offsets[threadIdx.x] =
						frontier_degrees_exclusive_sum_buckets_offsets[block_offset / TOP_DOWN_BUCKET_SIZE
								+ threadIdx.x];

			// We will use shared_buckets_offsets
			__syncthreads();

			//
			// shared_buckets_offsets gives us a range of the possible indexes
			// for edge of linear_threadx, we are looking for the value k such as
			// k is the max value such as frontier_degrees_exclusive_sum[k] <= linear_threadx
			//
			// we have 0 <= k < frontier_size
			// but we also have :
			//
			// frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE]
			// <= k
			// <= frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE + 1]
			//
			// To find the exact value in that range, we need a few values from frontier_degrees_exclusive_sum (see below)
			// We will load them here
			// We will load as much as we can - if it doesn't fit we will make multiple iteration of the next loop
			// Because all vertices in frontier have degree > 0, we know it will fits if left + 1 = right (see below)

			//We're going to load values in frontier_degrees_exclusive_sum for batch [left; right[
			//If it doesn't fit, --right until it does, then loop
			//It is excepted to fit on the first try, that's why we start right = nitems_per_thread

			IndexType left = 0;
			IndexType right = nitems_per_thread;

			while (left < nitems_per_thread) {
				//
				// Values that are necessary to compute the local binary searches
				// We only need those with indexes between extremes indexes of buckets_offsets
				// We need the next val for the binary search, hence the +1
				//

				IndexType nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK]
						- shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;

				//If left = right + 1 we are sure to have nvalues_to_load < TOP_DOWN_EXPAND_DIMX+1
				while (nvalues_to_load > (TOP_DOWN_EXPAND_DIMX + 1)) {
					--right;

					nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK]
							- shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;
				}

				IndexType nitems_per_thread_for_this_load = right - left;

				IndexType frontier_degrees_exclusive_sum_block_offset = shared_buckets_offsets[left
						* NBUCKETS_PER_BLOCK];

				//TODO put again the nvalues_to_load == 1
				if (threadIdx.x < nvalues_to_load) {
					shared_frontier_degrees_exclusive_sum[threadIdx.x] =
							frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset
									+ threadIdx.x];
				}

				if (nvalues_to_load == (TOP_DOWN_EXPAND_DIMX + 1) && threadIdx.x == 0) {
					shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX] =
							frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset
									+ TOP_DOWN_EXPAND_DIMX];
				}

				//shared_frontier_degrees_exclusive_sum is in shared mem, we will use it, sync
				//TODO we don't use it if nvalues_to_load == 1
				__syncthreads();

				// Now we will process the edges
				// Here each thread will process nitems_per_thread_for_this_load
				for (IndexType item_index = 0;
						item_index < nitems_per_thread_for_this_load;
						item_index += TOP_DOWN_BATCH_SIZE) {

					// We process TOP_DOWN_BATCH_SIZE edge in parallel (instruction parallism)
					// Reduces latency

					IndexType current_max_edge_index = min(block_offset
																				+ (left
																						+ nitems_per_thread_for_this_load)
																						* blockDim.x,
																		totaldegree);

					/**
					 * We will need vec_u (source of the edge) until the end if we need to save the
					 * predecessors. For others informations, we will reuse pointers on the go
					 * (nvcc does not color well the registers in that case)
					 */
					IndexType vec_u[TOP_DOWN_BATCH_SIZE];
					IndexType local_buf1[TOP_DOWN_BATCH_SIZE];
					IndexType local_buf2[TOP_DOWN_BATCH_SIZE];

					IndexType *vec_frontier_degrees_exclusive_sum_index = &local_buf2[0];

#pragma unroll
					for (IndexType iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {

						IndexType ibatch = left + item_index + iv;
						IndexType gid = block_offset + ibatch * blockDim.x + threadIdx.x;

						if (gid < current_max_edge_index) {
							IndexType start_off_idx = (ibatch * blockDim.x + threadIdx.x)
									/ TOP_DOWN_BUCKET_SIZE;
							IndexType bucket_start = shared_buckets_offsets[start_off_idx]
									- frontier_degrees_exclusive_sum_block_offset;
							IndexType bucket_end = shared_buckets_offsets[start_off_idx + 1]
									- frontier_degrees_exclusive_sum_block_offset;

							IndexType k = binsearch_maxle(shared_frontier_degrees_exclusive_sum,
																	gid,
																	bucket_start,
																	bucket_end)
									+ frontier_degrees_exclusive_sum_block_offset;
							vec_u[iv] = frontier[k]; // origin of this edge
							vec_frontier_degrees_exclusive_sum_index[iv] =
									frontier_degrees_exclusive_sum[k];
						} else {
							vec_u[iv] = -1;
							vec_frontier_degrees_exclusive_sum_index[iv] = -1;
						}

					}

					IndexType *vec_row_ptr_u = &local_buf1[0];

#pragma unroll
					for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
						IndexType u = vec_u[iv];
						//row_ptr for this vertex origin u
						vec_row_ptr_u[iv] = (u != -1) ? row_ptr[u] : -1;
					}

					//We won't need row_ptr after that, reusing pointer
					IndexType *vec_dest_v = vec_row_ptr_u;

#pragma unroll
					for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
						IndexType thread_item_index = left + item_index + iv;
						IndexType gid = block_offset + thread_item_index * blockDim.x + threadIdx.x;

						IndexType row_ptr_u = vec_row_ptr_u[iv];
						IndexType edge = row_ptr_u + gid - vec_frontier_degrees_exclusive_sum_index[iv];

						//Destination of this edge
						vec_dest_v[iv] = (row_ptr_u != -1) ? col_ind[edge] : -1;
//						if (vec_u[iv] != -1 && vec_dest_v[iv] != -1)
//						printf("Edge to examine: %d, %d\n", vec_u[iv],vec_dest_v[iv]);
					}

					//We don't need vec_frontier_degrees_exclusive_sum_index anymore
					IndexType *vec_v_visited_bmap = vec_frontier_degrees_exclusive_sum_index;

#pragma unroll
					for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
						IndexType v = vec_dest_v[iv];
						vec_v_visited_bmap[iv] = (v != -1) ? visited_bmap[v / INT_SIZE] : (~0); //will look visited
					}

					// From now on we will consider v as a frontier candidate
					// If for some reason vec_candidate[iv] should be put in the new_frontier
					// Then set vec_candidate[iv] = -1
					IndexType *vec_frontier_candidate = vec_dest_v;

#pragma unroll

					for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
						IndexType v = vec_frontier_candidate[iv];
						int m = 1 << (v % INT_SIZE);

						int is_visited = vec_v_visited_bmap[iv] & m;

						if (is_visited)
							vec_frontier_candidate[iv] = -1;
					}

#pragma unroll
					/**
					 * Here is where the distances, predecessors, new bitmap frontier and visited bitmap
					 * get written out.
					 */
					for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
						IndexType v = vec_frontier_candidate[iv];
						if (v != -1) {
							int m = 1 << (v % INT_SIZE);
							int q = atomicOr(&visited_bmap[v / INT_SIZE], m); //atomicOr returns old
							int f = atomicOr(&frontier_bmap[v / INT_SIZE], m);
							if (!(m & q)) { //if this thread was the first to discover this node
								if (distances)
									distances[v] = lvl;

								if (predecessors) {
									IndexType pred = vec_u[iv];
									predecessors[v] = pred;
								}
							}
						}
					}

					//We need naccepted_vertices to be ready
					__syncthreads();
				}

				//We need to keep shared_frontier_degrees_exclusive_sum coherent
				__syncthreads();

				//Preparing for next load
				left = right;
				right = nitems_per_thread;
			}

			//we need to keep shared_buckets_offsets coherent
			__syncthreads();
		}
	}

	template<typename IndexType, typename GlobalType>
	void frontier_expand(const IndexType *row_ptr,
								const IndexType *col_ind,
								const IndexType *frontier,
								const IndexType frontier_size,
								const IndexType totaldegree,
								const IndexType lvl,
								IndexType *frontier_bmap,
								const IndexType *frontier_degrees_exclusive_sum,
								const IndexType *frontier_degrees_exclusive_sum_buckets_offsets,
								int *visited_bmap,
								IndexType *distances,
								GlobalType *predecessors,
								cudaStream_t m_stream) {
		if (!totaldegree)
			return;

		dim3 block;
		block.x = TOP_DOWN_EXPAND_DIMX;

		IndexType max_items_per_thread = (totaldegree + MAXBLOCKS * block.x - 1)
				/ (MAXBLOCKS * block.x);

		dim3 grid;
		grid.x = min((totaldegree + max_items_per_thread * block.x - 1)
									/ (max_items_per_thread * block.x),
							(IndexType) MAXBLOCKS);

		topdown_expand_kernel<<<grid, block, 0, m_stream>>>(	row_ptr,
																				col_ind,
																				frontier,
																				frontier_size,
																				totaldegree,
																				max_items_per_thread,
																				lvl,
																				frontier_bmap,
																				frontier_degrees_exclusive_sum,
																				frontier_degrees_exclusive_sum_buckets_offsets,
																				visited_bmap,
																				distances,
																				predecessors);
		cudaCheckError();
	}
}
