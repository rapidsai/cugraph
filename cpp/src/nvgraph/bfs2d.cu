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
#include "include/bfs2d.hxx"
#include "include/bfs2d_kernels.cuh"
#include "include/debug_help.h"

namespace nvgraph {
	using namespace bfs_kernels;
	template<typename GlobalType, typename LocalType, typename ValueType>
	NVGRAPH_ERROR Bfs2d<GlobalType, LocalType, ValueType>::setup() {
		// Setup the frontier and visited bitmaps
		int32_t offset = M->getMatrixDecompositionDescription().getOffset();
		int32_t bitmap_n = (offset + 31) / 32;
		const MatrixDecompositionDescription<GlobalType, LocalType>* descr;
		descr = &(M->getMatrixDecompositionDescription());
		frontier_bmap = new VertexData2D<GlobalType, LocalType, int32_t>(descr, bitmap_n);
		visited_bmap = new VertexData2D<GlobalType, LocalType, int32_t>(descr, bitmap_n);

		// Setup frontier and frontierSize
		frontier = new VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>(descr);
		trim_frontier = new VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>(descr);
		frontierSize = new VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>(descr, 1);
		frontierSize_h.resize(descr->getNumBlocks());
		frontierDegree_h.resize(descr->getNumBlocks());
		degreeFlags = new VertexData2D_Unbuffered<GlobalType, LocalType, int8_t>(descr);

		// Setup the 2d distances and predecessors
		distances = new VertexData2D<GlobalType, LocalType, int32_t>(descr);
		predecessors = new VertexData2D<GlobalType, LocalType, GlobalType>(descr);

		// Setup degree exclusive sum and cub storage space
		LocalType n_exSum = offset + 1;
		size_t temp_bytes = getCubExclusiveSumStorageSize(n_exSum);
		size_t temp_bytes_compact = getCubSelectFlaggedStorageSize(n_exSum - 1);
		if (temp_bytes_compact > temp_bytes)
			temp_bytes = temp_bytes_compact;
		exSumStorage = new VertexData2D_Unbuffered<GlobalType, LocalType, int8_t>(descr, temp_bytes);
		exSumDegree = new VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>(descr,
																												offset + 1);

		// Setup bucketOffsets. Size is based on nnz, so we find the largest nnz over all blocks and use that.
		int32_t numBlocks = descr->getNumBlocks();
		size_t blockNnz = 0;
		for (int32_t i = 0; i < numBlocks; i++) {
			MultiValuedCsrGraph<LocalType, ValueType>* block = M->getBlockMatrix(i);
			blockNnz = max(block->get_num_edges(), blockNnz);
		}
		size_t bucketAllocSize = ((blockNnz / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2);
		bucketOffsets =
				new VertexData2D_Unbuffered<GlobalType, LocalType, LocalType>(descr, bucketAllocSize);
		// Size bucketOffsets based on blockNnz

		return NVGRAPH_OK;
	}

	template<typename GlobalType, typename LocalType, typename ValueType>
	NVGRAPH_ERROR Bfs2d<GlobalType, LocalType, ValueType>::configure(GlobalType *_distances,
																							GlobalType *_predecessors) {
		// Set the output locations.
		distances_out = _distances;
		predecessors_out = _predecessors;

		return NVGRAPH_OK;
	}

	template<typename GlobalType, typename LocalType, typename ValueType>
	void Bfs2d<GlobalType, LocalType, ValueType>::clean() {
		// Delete allocated data:
		if (distances)
			delete distances;
		if (predecessors)
			delete predecessors;
		if (frontier_bmap)
			delete frontier_bmap;
		if (visited_bmap)
			delete visited_bmap;
		if (frontier)
			delete frontier;
		if (trim_frontier)
			delete trim_frontier;
		if (frontierSize)
			delete frontierSize;
		if (exSumDegree)
			delete exSumDegree;
		if (exSumStorage)
			delete exSumStorage;
		if (bucketOffsets)
			delete bucketOffsets;
		if (degreeFlags)
			delete degreeFlags;
	}

	template<typename GlobalType, typename LocalType, typename ValueType>
	NVGRAPH_ERROR Bfs2d<GlobalType, LocalType, ValueType>::traverse(GlobalType source_vertex) {
		// Setup and get references for things
		const MatrixDecompositionDescription<GlobalType, LocalType>& description =
				M->getMatrixDecompositionDescription();
		const std::vector<int32_t>& deviceAssignments = description.getDeviceAssignments();
		const std::vector<cudaStream_t>& blockStreams = description.getBlockStreams();
		int32_t numBlocks = description.getNumBlocks();
		LocalType offset = description.getOffset();
		int32_t current_device;
		cudaGetDevice(&current_device);

		// Initialize the frontier bitmap with the source vertex set
		frontier_bmap->fillElements(0);
		LocalType blockRow = source_vertex / offset;
		LocalType blockOffset = source_vertex % offset;
		LocalType intId = blockOffset / 32;
		LocalType bitOffset = blockOffset % 32;
		int32_t bmapElement = 1 << bitOffset;
		int32_t bId = description.getBlockId(blockRow, blockRow);
		int32_t* copyTo = frontier_bmap->getCurrent(bId) + intId;
		cudaMemcpy(copyTo, &bmapElement, sizeof(int32_t), cudaMemcpyDefault);
		frontier_bmap->rowScatter();

		// Initialize frontierSizes to zero
		frontierSize->fillElements(0);
		frontierSize->rowScatter();

		// Initialize the visited bitmap with the source vertex set
		frontier_bmap->copyTo(visited_bmap);
		visited_bmap->columnScatter();

		// Initialize the distances and predecessors
		distances->fillElements((LocalType) -1);
		distances->setElement(source_vertex, (LocalType) 0);
		distances->columnScatter();
		predecessors->fillElements((GlobalType) -1);
		predecessors->columnScatter();

		// Setup initial frontier from bitmap frontier
		for (int i = 0; i < numBlocks; i++) {
			cudaStream_t stream = blockStreams[i];
			int32_t device = deviceAssignments[i];
			cudaSetDevice(device);
			convert_bitmap_to_queue(frontier_bmap->getCurrent(i),
											frontier_bmap->getN(),
											offset,
											frontier->get(i),
											frontierSize->get(i),
											stream);
			cudaMemcpyAsync(&frontierSize_h[i],
									frontierSize->get(i),
									sizeof(LocalType),
									cudaMemcpyDefault,
									stream);
		}
		description.syncAllStreams();

		// Main iteration loop
		int32_t globalSources = 1;
		LocalType level = 1;
		while (globalSources > 0) {

//			std::cout << "Starting with level " << level << "\n";

			// Remove frontier nodes with locally zero degree
			for (int i = 0; i < numBlocks; i++) {
				// Checking that there is work to be done for this block
				if (frontierSize_h[i] > 0) {
					// Write out the degree of each frontier node into exSumDegree
					degreeIterator<LocalType> degreeIt(M->getBlockMatrix(i)->get_raw_row_offsets());
					cudaStream_t stream = blockStreams[i];
					cudaSetDevice(deviceAssignments[i]);
					set_degree_flags(	degreeFlags->get(i),
											frontier->get(i),
											degreeIt,
											frontierSize_h[i],
											stream);
//					set_frontier_degree(exSumDegree->get(i),
//												frontier->get(i),
//												degreeIt,
//												frontierSize_h[i],
//												stream);
//
//					cudaStreamSynchronize(stream);
//					std::cout << "Block " << i << " before compaction.\n";
//					debug::printDeviceVector(frontier->get(i), frontierSize_h[i], "Frontier");
//					debug::printDeviceVector(exSumDegree->get(i), frontierSize_h[i], "Frontier Degree");

					// Use degreeIterator as flags to compact the frontier
					cudaSetDevice(deviceAssignments[i]);
					size_t numBytes = exSumStorage->getN();
					cub::DeviceSelect::Flagged(exSumStorage->get(i),
														numBytes,
														frontier->get(i),
														degreeFlags->get(i),
														trim_frontier->get(i),
														frontierSize->get(i),
														frontierSize_h[i],
														stream);
					cudaMemcpyAsync(&frontierSize_h[i],
											frontierSize->get(i),
											sizeof(LocalType),
											cudaMemcpyDefault,
											stream);
				}
			}
			description.syncAllStreams();

			// Setup load balancing for main kernel call
			for (int i = 0; i < numBlocks; i++) {
				// Checking that there is work to be done for this block:
				if (frontierSize_h[i] > 0) {
					// Write out the degree of each frontier node into exSumDegree
					degreeIterator<LocalType> degreeIt(M->getBlockMatrix(i)->get_raw_row_offsets());
					cudaStream_t stream = blockStreams[i];
					cudaSetDevice(deviceAssignments[i]);
					set_frontier_degree(exSumDegree->get(i),
												trim_frontier->get(i),
												degreeIt,
												frontierSize_h[i],
												stream);

//					cudaStreamSynchronize(stream);
//					std::cout << "Block " << i << " after compaction.\n";
//					debug::printDeviceVector(trim_frontier->get(i), frontierSize_h[i], "Frontier");
//					debug::printDeviceVector(exSumDegree->get(i), frontierSize_h[i], "Frontier Degree");

					// Get the exclusive sum of the frontier degrees, store in exSumDegree
					size_t numBytes = exSumStorage->getN();
					cub::DeviceScan::ExclusiveSum(exSumStorage->get(i),
															numBytes,
															exSumDegree->get(i),
															exSumDegree->get(i),
															frontierSize_h[i] + 1,
															stream);
					cudaMemcpyAsync(&frontierDegree_h[i],
											exSumDegree->get(i) + frontierSize_h[i],
											sizeof(LocalType),
											cudaMemcpyDefault,
											stream);
				}
			}
			description.syncAllStreams();

//			for (int i = 0; i < numBlocks; i++) {
//				std::cout << "Block " << i << " frontierNodes " << frontierSize_h[i]
//						<< " frontierDegree " << frontierDegree_h[i] << "\n";
//			}

			for (int i = 0; i < numBlocks; i++) {
				// Checking that there is work to be done for this block:
				if (frontierSize_h[i] > 0) {
					cudaStream_t stream = blockStreams[i];
					cudaSetDevice(deviceAssignments[i]);
					compute_bucket_offsets(exSumDegree->get(i),
													bucketOffsets->get(i),
													frontierSize_h[i],
													frontierDegree_h[i],
													stream);
				}
			}

			// Call main kernel to get new frontier
			frontier_bmap->fillElements(0);
			frontier_bmap->rowScatter();
			for (int i = 0; i < numBlocks; i++) {
				// Checking that there is work to be done for this block:
				if (frontierDegree_h[i] > 0) {
					cudaSetDevice(deviceAssignments[i]);
					frontier_expand(M->getBlockMatrix(i)->get_raw_row_offsets(),
											M->getBlockMatrix(i)->get_raw_column_indices(),
											trim_frontier->get(i),
											frontierSize_h[i],
											frontierDegree_h[i],
											level,
											frontier_bmap->getCurrent(i),
											exSumDegree->get(i),
											bucketOffsets->get(i),
											visited_bmap->getCurrent(i),
											distances->getCurrent(i),
											predecessors->getCurrent(i),
											blockStreams[i]);

//					cudaStreamSynchronize(blockStreams[i]);
//					int bitsSet =
//							thrust::reduce(thrust::device,
//												thrust::make_transform_iterator(frontier_bmap->getCurrent(i),
//																							popCount()),
//												thrust::make_transform_iterator(frontier_bmap->getCurrent(i)
//																									+ frontier_bmap->getN(),
//																							popCount()));
//					std::cout << "Block " << i << " Level " << level << " has " << bitsSet << " bits set\n";
				}
			}
			description.syncAllStreams();

			// Update and propogate new frontier and visited bitmaps
			frontier_bmap->template columnReduce<BitwiseOr>();
			frontier_bmap->rowScatter();
			visited_bmap->template columnReduce<BitwiseOr>();
			visited_bmap->columnScatter();

			// Convert bitmap frontier to list frontier and update globalSources
			frontierSize->fillElements(0);
			frontierSize->rowScatter();
			for (int i = 0; i < numBlocks; i++) {
				cudaStream_t stream = blockStreams[i];
				int32_t device = deviceAssignments[i];
				cudaSetDevice(device);
				convert_bitmap_to_queue(frontier_bmap->getCurrent(i),
												frontier_bmap->getN(),
												offset,
												frontier->get(i),
												frontierSize->get(i),
												stream);
				cudaMemcpyAsync(&frontierSize_h[i],
										frontierSize->get(i),
										sizeof(LocalType),
										cudaMemcpyDefault,
										stream);
			}
			description.syncAllStreams();
			GlobalType blockRows = description.getBlockRows();
			globalSources = 0;
			for (int i = 0; i < blockRows; i++) {
				int32_t bId = description.getBlockId(i, i);
				globalSources += frontierSize_h[bId];
			}

//			std::cout << "Finished with level " << level << " frontiers:\n";
//			for (int i = 0; i < numBlocks; i++)
//				std::cout << "\tBlock " << i << " : " << frontierSize_h[i] << "\n";

			// Increment level
			level++;
		}

		// Globalize the predecessors by row
		for (int i = 0; i < numBlocks; i++) {
			cudaStream_t stream = blockStreams[i];
			int32_t device = deviceAssignments[i];
			cudaSetDevice(device);
			int32_t rowId = description.getBlockRow(i);
			GlobalType globalOffset = rowId * description.getOffset();
			globalize_ids(predecessors->getCurrent(i),
								globalOffset,
								(GlobalType) predecessors->getN(),
								stream);
		}
		description.syncAllStreams();

		// Propogate predecessors and distances
		predecessors->template columnReduce<predMerge>();
		distances->template columnReduce<predMerge>();

		// Copy out predecessors and distances to user provided locations
		LocalType* temp = (LocalType*) malloc(distances->getN() * sizeof(LocalType));
		int32_t writeOffset = 0;
		int32_t numRows = description.getNumRows();
		int32_t blockRows = description.getBlockRows();
		for (int i = 0; i < blockRows; i++) {
			// Copy out the data for the block on the diagonal
			int32_t bId = description.getBlockId(i, i);
			int32_t n = predecessors->getN();
			cudaMemcpy(temp, predecessors->getCurrent(bId), n * sizeof(LocalType), cudaMemcpyDefault);
			for (int j = 0; j < n; j++) {
				if (writeOffset + j < numRows)
					predecessors_out[writeOffset + j] = temp[j];
			}
			cudaMemcpy(temp, distances->getCurrent(bId), n * sizeof(LocalType), cudaMemcpyDefault);
			for (int j = 0; j < n; j++) {
				if (writeOffset + j < numRows)
					distances_out[writeOffset + j] = temp[j];
			}
			writeOffset += n;
		}

		return NVGRAPH_OK;
	}

	template<typename GlobalType, typename LocalType, typename ValueType>
	NVGRAPH_ERROR Bfs2d<GlobalType, LocalType, ValueType>::traverse(GlobalType *source_vertices,
																							int32_t nsources) {
		for (int32_t i = 0; i < nsources; i++) {
			traverse(source_vertices[i]);
		}
		return NVGRAPH_OK;
	}

	template class Bfs2d<int, int, int> ;
}
