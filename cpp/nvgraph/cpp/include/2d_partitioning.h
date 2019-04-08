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
 /*
 * 2d_partitioning.h
 *
 *  Created on: Apr 9, 2018
 *      Author: jwyles
 */

#pragma once

#include <stdint.h>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include <multi_valued_csr_graph.hxx>
#include <nvgraph_vector.hxx>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace nvgraph {

	template<typename T, typename W>
	struct CSR_Result_Weighted {
		int64_t size;
		int64_t nnz;
		T* rowOffsets;
		T* colIndices;
		W* edgeWeights;

		CSR_Result_Weighted() :
				size(0), nnz(0), rowOffsets(NULL), colIndices(NULL), edgeWeights(NULL) {
		}

		void Destroy() {
			if (rowOffsets)
				cudaFree(rowOffsets);
			if (colIndices)
				cudaFree(colIndices);
			if (edgeWeights)
				cudaFree(edgeWeights);
		}
	};

	// Define kernel for copying run length encoded values into offset slots.
	template<typename T>
	__global__ void offsetsKernel(T runCounts, T* unique, T* counts, T* offsets) {
		for (int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
				idx < runCounts;
				idx += gridDim.x * blockDim.x) {
			offsets[unique[idx]] = counts[idx];
		}
	}

	/**
	 * Method for converting COO to CSR format
	 * @param sources The array of source indices
	 * @param destinations The array of destination indices
	 * @param edgeWeights The array of edge weights
	 * @param nnz The number of non zero values
	 * @param maxId The largest id contained in the matrix
	 * @param result The result is stored here.
	 */
	template<typename T, typename W>
	void ConvertCOOtoCSR_weighted(T* sources,
											T* destinations,
											W* edgeWeights,
											int64_t nnz,
											T maxId,
											CSR_Result_Weighted<T, W>& result) {
		// Sort source and destination columns by source
		// Allocate local memory for operating on
		T* srcs, *dests;
		W* weights = NULL;
		cudaMalloc(&srcs, sizeof(T) * nnz);
		cudaMalloc(&dests, sizeof(T) * nnz);
		if (edgeWeights)
			cudaMalloc(&weights, sizeof(W) * nnz);
		cudaMemcpy(srcs, sources, sizeof(T) * nnz, cudaMemcpyDefault);
		cudaMemcpy(dests, destinations, sizeof(T) * nnz, cudaMemcpyDefault);
		if (edgeWeights)
			cudaMemcpy(weights, edgeWeights, sizeof(W) * nnz, cudaMemcpyDefault);

		// Call Thrust::sort_by_key to sort the arrays with srcs as keys:
		if (edgeWeights)
			thrust::sort_by_key(thrust::device,
										srcs,
										srcs + nnz,
										thrust::make_zip_iterator(thrust::make_tuple(dests, weights)));
		else
			thrust::sort_by_key(thrust::device, srcs, srcs + nnz, dests);

		result.size = maxId + 1;

		// Allocate offsets array
		cudaMalloc(&result.rowOffsets, (maxId + 2) * sizeof(T));

		// Set all values in offsets array to zeros
		cudaMemset(result.rowOffsets, 0, (maxId + 2) * sizeof(T));

		// Allocate temporary arrays same size as sources array, and single value to get run counts
		T* unique, *counts, *runCount;
		cudaMalloc(&unique, (maxId + 1) * sizeof(T));
		cudaMalloc(&counts, (maxId + 1) * sizeof(T));
		cudaMalloc(&runCount, sizeof(T));

		// Use CUB run length encoding to get unique values and run lengths
		void *tmpStorage = NULL;
		size_t tmpBytes = 0;
		cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz);
		cudaMalloc(&tmpStorage, tmpBytes);
		cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz);
		cudaFree(tmpStorage);

		// Set offsets to run sizes for each index
		T runCount_h;
		cudaMemcpy(&runCount_h, runCount, sizeof(T), cudaMemcpyDefault);
		int threadsPerBlock = 1024;
		int numBlocks = min(65535, (runCount_h + threadsPerBlock - 1) / threadsPerBlock);
		offsetsKernel<<<numBlocks, threadsPerBlock>>>(runCount_h, unique, counts, result.rowOffsets);

		// Scan offsets to get final offsets
		thrust::exclusive_scan(thrust::device,
										result.rowOffsets,
										result.rowOffsets + maxId + 2,
										result.rowOffsets);

		// Clean up temporary allocations
		result.nnz = nnz;
		result.colIndices = dests;
		result.edgeWeights = weights;
		cudaFree(srcs);
		cudaFree(unique);
		cudaFree(counts);
		cudaFree(runCount);
	}

	/**
	 * Describes the 2D decomposition of a partitioned matrix.
	 */
	template<typename GlobalType, typename LocalType>
	class MatrixDecompositionDescription {
	protected:
		GlobalType numRows; 	// Global number of rows in matrix
		GlobalType numCols; 	// Global number of columns in matrix
		GlobalType nnz;			// Global number of non-zeroes in matrix
		GlobalType blockRows;	// Number of rows of blocks in the decomposition
		GlobalType blockCols;	// Number of columns of rows in the decomposition
		LocalType offset;
		// Offsets-like arrays for rows and columns defining the start/end of the
		// sections of the global id space belonging to each row and column.
		std::vector<GlobalType> rowOffsets;
		std::vector<GlobalType> colOffsets;
		// Array of integers one for each block, defining the device it is assigned to
		std::vector<int32_t> deviceAssignments;
		std::vector<cudaStream_t> blockStreams;
		public:

		MatrixDecompositionDescription() :
				numRows(0), numCols(0), nnz(0), blockRows(0), blockCols(0) {
			rowOffsets.push_back(0);
			colOffsets.push_back(0);
			deviceAssignments.push_back(0);
		}

		// Basic constructor, just takes in the values of its members.
		MatrixDecompositionDescription(GlobalType numRows,
													GlobalType numCols,
													GlobalType nnz,
													GlobalType blockRows,
													GlobalType blockCols,
													std::vector<GlobalType> rowOffsets,
													std::vector<GlobalType> colOffsets,
													std::vector<int32_t> deviceAssignments) :
				numRows(numRows), numCols(numCols), nnz(nnz), blockRows(blockRows),
						blockCols(blockCols), rowOffsets(rowOffsets), colOffsets(colOffsets),
						deviceAssignments(deviceAssignments) {
		}

		// Constructs a MatrixDecompositionDescription for a square matrix given the
		// number of rows in the matrix and number of rows of blocks.
		MatrixDecompositionDescription(GlobalType numRows,
													GlobalType numBlockRows,
													GlobalType nnz,
													std::vector<int32_t> devices) :
				numRows(numRows),
						numCols(numRows),
						blockRows(numBlockRows),
						blockCols(numBlockRows),
						nnz(nnz) {
			// Tracking the current set device to change back
			int currentDevice;
			cudaGetDevice(&currentDevice);

			// Setting up the row and col offsets into equally sized chunks
			GlobalType remainder = numRows % blockRows;
			if (remainder != 0)
				offset = (numRows + blockRows - remainder) / blockRows;
			else
				offset = numRows / blockRows;

			rowOffsets.resize(blockRows + 1);
			colOffsets.resize(blockRows + 1);
			for (int i = 0; i < blockRows; i++) {
				rowOffsets[i] = i * offset;
				colOffsets[i] = i * offset;
			}
			rowOffsets.back() = blockRows * offset;
			colOffsets.back() = blockCols * offset;

			// Setting up the device assignments using the given device ids and also
			// setting up the stream associated with each block.
			deviceAssignments.resize(getNumBlocks());
			blockStreams.resize(getNumBlocks());
			for (int i = 0; i < getNumBlocks(); i++) {
				int device = devices[i % devices.size()];
				deviceAssignments[i] = device;
				cudaSetDevice(device);
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				blockStreams[i] = stream;
			}

			// Restoring to current device when called
			cudaSetDevice(currentDevice);
		}

		// Gets the row id for the block containing the given global row id
		int32_t getRowId(GlobalType val) const {
			return std::upper_bound(rowOffsets.begin(), rowOffsets.end(), val) - rowOffsets.begin() - 1;
		}

		// Gets the column id for the block containing the given global column id
		int32_t getColId(GlobalType val) const {
			return std::upper_bound(colOffsets.begin(), colOffsets.end(), val) - colOffsets.begin() - 1;
		}

		// Gets the number of blocks in the decomposition:
		int32_t getNumBlocks() const {
			return blockRows * blockCols;
		}

		// Getter for offset
		LocalType getOffset() const {
			return offset;
		}

		// Getter for deviceAssignments
		const std::vector<int32_t>& getDeviceAssignments() const {
			return deviceAssignments;
		}

		/**
		 * Getter for vector of streams for each block.
		 * @return Reference to vector of streams for each block
		 */
		const std::vector<cudaStream_t>& getBlockStreams() const {
			return blockStreams;
		}

		/**
		 * Getter for nnz
		 * @return The global number of non-zero elements
		 */
		GlobalType getNnz() const {
			return nnz;
		}

		/**
		 * Getter method for numRows
		 * @return The number of global rows in the matrix
		 */
		GlobalType getNumRows() const {
			return numRows;
		}

		/**
		 * Getter for BlockRows
		 * @return The number of blocks in a row in the decomposition.
		 */
		GlobalType getBlockRows() const {
			return blockRows;
		}

		/**
		 * Getter for BlockCols
		 * @return The number of blocks in a column in the decomposition.
		 */
		GlobalType getBlockCols() const {
			return blockCols;
		}

		/**
		 * Given a block id, returns the row which that block is in.
		 * @param bId The block ID
		 * @return The row number
		 */
		int32_t getBlockRow(int32_t bId) const {
			return bId / blockCols;
		}

		/**
		 * Given a block id, returns the column which that block is in.
		 * @param bId The block ID
		 * @return The column number
		 */
		int32_t getBlockCol(int32_t bId) const {
			return bId % blockCols;
		}

		/**
		 * Takes a COO global row and produces the COO local row and the block to which it belongs.
		 * @param globalRow The global row ID
		 * @param globalCol The global column ID
		 * @param localRow The block local row ID (return)
		 * @param localCol The block local column ID (return)
		 * @param blockId The block ID (return)
		 */
		void convertGlobaltoLocalRow(GlobalType globalRow,
												GlobalType globalCol,
												LocalType& localRow,
												LocalType& localCol,
												int32_t& blockId) const {
			int32_t rowId = getRowId(globalRow);
			int32_t colId = getColId(globalCol);
			blockId = rowId * blockCols + colId;
			localRow = globalRow - rowOffsets[rowId];
			localCol = globalCol - colOffsets[colId];
		}

		/**
		 * Takes in a row ID and column ID and returns the corresponding block ID
		 * @param rowId The row ID
		 * @param colId The column ID
		 * @return The ID of the corresponding block
		 */
		int32_t getBlockId(int32_t rowId, int32_t colId) const {
			return rowId * blockCols + colId;
		}

		/**
		 * Helper method to synchronize all streams after operations are issued.
		 */
		void syncAllStreams() const {
			int32_t numBlocks = getNumBlocks();
			int32_t current_device;
			cudaGetDevice(&current_device);
			for (int32_t i = 0; i < numBlocks; i++) {
				cudaSetDevice(deviceAssignments[i]);
				cudaStreamSynchronize(blockStreams[i]);
			}
			cudaSetDevice(current_device);
		}

		/**
		 * This method is only for testing and debugging use.
		 * @return A human readable string representation of the object
		 */
		std::string toString() const {
			std::stringstream ss;
			ss << "Global Info:\n\tnumRows: " << numRows << ", numCols: " << numCols << ", nnz: "
					<< nnz;
			ss << "\n";
			ss << "Block Info:\n\tblockRows: " << blockRows << ", blockCols: " << blockCols;
			ss << "\n";
			ss << "rowOffsets: [";
			for (int i = 0; i < (int) rowOffsets.size(); i++)
				ss << rowOffsets[i] << (i == (int) rowOffsets.size() - 1 ? "]\n" : ", ");
			ss << "colOffsets: [";
			for (int i = 0; i < (int) colOffsets.size(); i++)
				ss << colOffsets[i] << (i == (int) colOffsets.size() - 1 ? "]\n" : ", ");
			ss << "deviceAssignments: [";
			for (int i = 0; i < (int) deviceAssignments.size(); i++)
				ss << deviceAssignments[i] << (i == (int) deviceAssignments.size() - 1 ? "]\n" : ", ");
			return ss.str();
		}
	};

	template<typename GlobalType, typename LocalType, typename ValueType>
	class Matrix2d {
	protected:
		// Description of the matrix decomposition
		MatrixDecompositionDescription<GlobalType, LocalType> description;

		// Array of block matrices forming the decomposition
		std::vector<MultiValuedCsrGraph<LocalType, ValueType>*> blocks;
		public:
		Matrix2d() {
		}
		Matrix2d(MatrixDecompositionDescription<GlobalType, LocalType> descr,
					std::vector<MultiValuedCsrGraph<LocalType, ValueType>*> blocks) :
				description(descr), blocks(blocks) {
		}

		const MatrixDecompositionDescription<GlobalType, LocalType>& getMatrixDecompositionDescription() {
			return description;
		}

		MultiValuedCsrGraph<LocalType, ValueType>* getBlockMatrix(int32_t bId) {
			return blocks[bId];
		}

		std::string toString() {
			std::stringstream ss;
			ss << "MatrixDecompositionDescription:\n" << description.toString();
			for (int i = 0; i < (int) blocks.size(); i++) {
				ss << "Block " << i << ":\n";
				size_t numVerts = blocks[i]->get_num_vertices();
				size_t numEdges = blocks[i]->get_num_edges();
				size_t numValues = blocks[i]->getNumValues();
				ss << "numVerts: " << numVerts << ", numEdges: " << numEdges << "\n";
				LocalType* rowOffsets = (LocalType*) malloc((numVerts + 1) * sizeof(LocalType));
				LocalType* colIndices = (LocalType*) malloc(numEdges * sizeof(LocalType));
				ValueType* values = NULL;
				if (numValues > 0)
					values = (ValueType*) malloc(numEdges * sizeof(ValueType));
				cudaMemcpy(rowOffsets,
								blocks[i]->get_raw_row_offsets(),
								(numVerts + 1) * sizeof(LocalType),
								cudaMemcpyDefault);
				cudaMemcpy(colIndices,
								blocks[i]->get_raw_column_indices(),
								numEdges * sizeof(LocalType),
								cudaMemcpyDefault);
				if (values)
					cudaMemcpy(values,
									blocks[i]->get_raw_edge_dim(0),
									numEdges * sizeof(ValueType),
									cudaMemcpyDefault);
				int idxCount = numEdges >= (numVerts + 1) ? numEdges : (numVerts + 1);
				ss << "Idx\tOffset\tColInd\tValue\n";
				for (int j = 0; j < idxCount; j++) {
					if (j < (int) numVerts + 1 && j < (int) numEdges)
						ss << j << ":\t" << rowOffsets[j] << "\t" << colIndices[j] << "\t"
								<< (values ? values[j] : 0)
								<< "\n";
					else if (j < (int) numVerts + 1 && j >= (int) numEdges)
						ss << j << ":\t" << rowOffsets[j] << "\n";
					else if (j >= (int) numVerts + 1 && j < (int) numEdges)
						ss << j << ":\t" << "\t" << colIndices[j] << "\t" << (values ? values[j] : 0)
								<< "\n";
				}
				free(rowOffsets);
				free(colIndices);
				free(values);
			}
			return ss.str();
		}
	};

	template<typename GlobalType, typename LocalType, typename ValueType>
	class VertexData2D {
		const MatrixDecompositionDescription<GlobalType, LocalType>* description;
		int32_t n;
		std::vector<cub::DoubleBuffer<ValueType> > values;
		public:
		/**
		 * Creates a VertexData2D object given a pointer to a MatrixDecompositionDescription
		 * object which describes the matrix the data is attached to. Data buffers are
		 * allocated for each block using the offset from the description to size the
		 * buffers, and to locate the buffers on the same GPU as the matrix block.
		 */
		VertexData2D(const MatrixDecompositionDescription<GlobalType, LocalType>* descr) :
				description(descr) {
			// Resize the values array to be the same size as number of blocks
			values.resize(descr->getNumBlocks());

			// Grab the current device id to switch back after allocations are done
			int current_device;
			cudaGetDevice(&current_device);
			LocalType allocSize = descr->getOffset();
			n = allocSize;
			// Allocate the data for each block
			for (size_t i = 0; i < descr->getDeviceAssignments().size(); i++) {
				int device = descr->getDeviceAssignments()[i];
				cudaSetDevice(device);
				ValueType* d_current, *d_alternate;
				cudaMalloc(&d_current, sizeof(ValueType) * n);
				cudaMalloc(&d_alternate, sizeof(ValueType) * n);
				values[i].d_buffers[0] = d_current;
				values[i].d_buffers[1] = d_alternate;
			}

			// Set the device back to what it was initially
			cudaSetDevice(current_device);
		}

		/**
		 * Creates a VertexData2D object given a pointer to a MatrixDecompositionDescription
		 * object, which describes the matrix the data is attached to, and an integer which indicates
		 * how many data elements should be allocated for each block. Data buffers are allocated
		 * for each block using the offset from the description to size the buffers, and to locate
		 * the buffers on the same GPU as the matrix block.
		 */
		VertexData2D(const MatrixDecompositionDescription<GlobalType, LocalType>* descr, size_t _n) :
				description(descr) {
			// Resize the values array to be the same size as number of blocks
			values.resize(descr->getNumBlocks());

			// Grab the current device id to switch back after allocations are done
			int current_device;
			cudaGetDevice(&current_device);
			LocalType allocSize = _n;
			n = allocSize;
			// Allocate the data for each block
			for (size_t i = 0; i < descr->getDeviceAssignments().size(); i++) {
				int device = descr->getDeviceAssignments()[i];
				cudaSetDevice(device);
				ValueType* d_current, *d_alternate;
				cudaMalloc(&d_current, sizeof(ValueType) * n);
				cudaMalloc(&d_alternate, sizeof(ValueType) * n);
				values[i].d_buffers[0] = d_current;
				values[i].d_buffers[1] = d_alternate;
			}

			// Set the device back to what it was initially
			cudaSetDevice(current_device);
		}

		~VertexData2D() {
			for (size_t i = 0; i < values.size(); i++) {
				if (values[i].Current())
					cudaFree(values[i].Current());
				if (values[i].Alternate())
					cudaFree(values[i].Alternate());
			}
		}

		/**
		 * Getter for n the size of each block's allocation in elements.
		 * @return The value of n
		 */
		int32_t getN() {
			return n;
		}

		/**
		 * Getter for the MatrixDecompositionDescription associated with this VertexData2D
		 * @return Pointer to the MatrixDecompositionDescription for this VertexData2D
		 */
		const MatrixDecompositionDescription<GlobalType, LocalType>* getDescription() {
			return description;
		}

		/**
		 * Gets the current buffer corresponding to the given block ID
		 */
		ValueType* getCurrent(int bId) {
			return values[bId].Current();
		}

		/**
		 * Gets the alternate buffer corresponding to the given block ID
		 */
		ValueType* getAlternate(int bId) {
			return values[bId].Alternate();
		}

		/**
		 * Swaps the current and alternate buffers for all block IDs
		 */
		void swapBuffers() {
			for (size_t i = 0; i < values.size(); i++)
				values[i].selector ^= 1;
		}

		/**
		 * Sets an element in the global array, assuming that the data is currently
		 * valid and in the diagonal blocks. After calling this method either columnScatter
		 * or rowScatter should be called to propagate the change to all blocks.
		 */
		void setElement(GlobalType globalIndex, ValueType val) {
			LocalType blockId = globalIndex / n;
			LocalType blockOffset = globalIndex % n;
			int32_t bId = description->getBlockId(blockId, blockId);
			ValueType* copyTo = values[bId].Current() + blockOffset;
			cudaMemcpy(copyTo, &val, sizeof(ValueType), cudaMemcpyDefault);
		}

		/**
		 * Sets the elements of the global array, using the provided array of values. The values
		 * are set in the blocks of the diagonal, columnScatter or rowScatter should be called
		 * to propogate to all blocks.
		 * @param vals Pointer to an array with the values to be set.
		 */
		void setElements(ValueType* vals) {
			LocalType offset = description->getOffset();
			int32_t numRows = description->getBlockRows();
			for (int i = 0; i < numRows; i++) {
				int32_t id = description->getBlockId(i, i);
				cudaStream_t stream = description->getBlockStreams()[id];
				ValueType* copyFrom = vals + i * n;
				ValueType* copyTo = values[id].Current();
				cudaMemcpyAsync(copyTo, copyFrom, sizeof(ValueType) * n, cudaMemcpyDefault, stream);
			}
			description->syncAllStreams();
		}

		/**
		 * Fills the elements of the data array with the given value.
		 * The elements on the diagonal are filled with the given value. After filling,
		 * either rowScatter or columnScatter will copy the values across the blocks in
		 * either the rows or columns depending on the use.
		 * @param val The value to fill the array with
		 */
		void fillElements(ValueType val) {
			int current_device;
			cudaGetDevice(&current_device);
			int32_t numRows = description->getBlockRows();
			for (int32_t i = 0; i < numRows; i++) {
				int32_t blockId = description->getBlockId(i, i);
				ValueType* vals = getCurrent(blockId);
				int deviceId = description->getDeviceAssignments()[blockId];
				cudaStream_t stream = description->getBlockStreams()[blockId];
				cudaSetDevice(deviceId);
				thrust::fill(thrust::cuda::par.on(stream), vals, vals + n, val);
			}
			description->syncAllStreams();
			cudaSetDevice(current_device);
		}

		/**
		 * Copies the values of the diagonal blocks in this VertexData2D into the
		 * VertexData2D specified.
		 * @param other Pointer to the VertexData2D to copy into
		 */
		void copyTo(VertexData2D<GlobalType, LocalType, ValueType>* other) {
			const MatrixDecompositionDescription<GlobalType, LocalType>* otherDescr =
					other->getDescription();
			// Do a quick check that the sizes of both block arrays are the same.
			if (description->getBlockRows() == otherDescr->getBlockRows() && n == other->getN()) {
				// Issue asynchronous copies for each block's data
				for (int i = 0; i < description->getBlockRows(); i++) {
					int32_t bId = description->getBlockId(i, i);
					ValueType* copyFrom = getCurrent(bId);
					ValueType* copyTo = other->getCurrent(bId);
					cudaStream_t stream = description->getBlockStreams()[bId];
					cudaMemcpyAsync(copyTo, copyFrom, n * sizeof(ValueType), cudaMemcpyDefault, stream);
				}
				// Synchronize the streams after the copies are done
				for (int i = 0; i < description->getBlockRows(); i++) {
					int32_t bId = description->getBlockId(i, i);
					cudaStream_t stream = description->getBlockStreams()[bId];
					cudaStreamSynchronize(stream);
				}
			}
		}

		/**
		 * This method implements a row-wise reduction of each blocks data into a
		 * single array for each row. The block on the diagonal will have the result.
		 */
		template<typename Operator>
		void rowReduce() {
			int current_device;
			cudaGetDevice(&current_device);
			Operator op;

			// For each row in the decomposition:
			int32_t numRows = description->getBlockRows();
			std::vector<int32_t> blockIds;
			for (int32_t i = 0; i < numRows; i++) {
				// Put all the block ids for the row into a vector, with the ID of the diagonal block
				// at index 0.
				std::vector<int32_t> blockIds;
				blockIds.push_back(-1);
				for (int32_t j = 0; j < numRows; j++) {
					if (i == j) {
						blockIds[0] = description->getBlockId(i, j);
					}
					else {
						blockIds.push_back(description->getBlockId(i, j));
					}
				}

				// Do a binary tree reduction. At each step the primary buffer of the sender is
				// copied into the secondary buffer of the receiver. After the copy is done
				// each receiver performs the reduction operator and stores the result in it's
				// primary buffer.
				for (int32_t j = 2; (j / 2) < numRows; j *= 2) {
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the receiver
							int32_t receiverId = blockIds[id];

							// blockIds[id + j/2] is the sender
							int32_t senderId = blockIds[id + j / 2];

							// Get the stream associated with the receiver's block id
							cudaStream_t stream = description->getBlockStreams()[receiverId];

							// Copy from the sender to the receiver (use stream associated with receiver)
							cudaMemcpyAsync(values[receiverId].Alternate(),
													values[senderId].Current(),
													sizeof(ValueType) * n,
													cudaMemcpyDefault,
													stream);

							// Invoke the reduction operator on the receiver's GPU and values arrays.
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							ValueType* input1 = values[receiverId].Alternate();
							ValueType* input2 = values[receiverId].Current();
							thrust::transform(thrust::cuda::par.on(stream),
													input1,
													input1 + n,
													input2,
													input2,
													op);
						}
					}
					// Sync all active streams before next step
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the receiver
							int32_t receiverId = blockIds[id];

							// Set the device to the receiver and sync the stream
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							cudaStreamSynchronize(description->getBlockStreams()[receiverId]);
						}
					}
				}
			}

			cudaSetDevice(current_device);
		}

		/**
		 * This method implements a column-wise reduction of each blocks data into a
		 * single array for each column. The block on the diagonal will have the result.
		 */
		template<typename Operator>
		void columnReduce() {
			int current_device;
			cudaGetDevice(&current_device);
			Operator op;

			// For each column in the decomposition:
			int32_t numRows = description->getBlockRows();
			std::vector<int32_t> blockIds;
			for (int32_t i = 0; i < numRows; i++) {
				// Put all the block ids for the row into a vector, with the ID of the diagonal block
				// at index 0.
				std::vector<int32_t> blockIds;
				blockIds.push_back(-1);
				for (int32_t j = 0; j < numRows; j++) {
					if (i == j) {
						blockIds[0] = description->getBlockId(j, i);
					}
					else {
						blockIds.push_back(description->getBlockId(j, i));
					}
				}

				// Do a binary tree reduction. At each step the primary buffer of the sender is
				// copied into the secondary buffer of the receiver. After the copy is done
				// each receiver performs the reduction operator and stores the result in it's
				// primary buffer.
				for (int32_t j = 2; (j / 2) < numRows; j *= 2) {
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the receiver
							int32_t receiverId = blockIds[id];

							// blockIds[id + j/2] is the sender
							int32_t senderId = blockIds[id + j / 2];

							// Get the stream associated with the receiver's block id
							cudaStream_t stream = description->getBlockStreams()[receiverId];

							// Copy from the sender to the receiver (use stream associated with receiver)
							cudaMemcpyAsync(values[receiverId].Alternate(),
													values[senderId].Current(),
													sizeof(ValueType) * n,
													cudaMemcpyDefault,
													stream);

							// Invoke the reduction operator on the receiver's GPU and values arrays.
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							ValueType* input1 = values[receiverId].Alternate();
							ValueType* input2 = values[receiverId].Current();
							thrust::transform(thrust::cuda::par.on(stream),
													input1,
													input1 + n,
													input2,
													input2,
													op);
						}
					}
					// Sync all active streams before next step
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the receiver
							int32_t receiverId = blockIds[id];

							// Set the device to the receiver and sync the stream
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							cudaStreamSynchronize(description->getBlockStreams()[receiverId]);
						}
					}
				}
			}

			cudaSetDevice(current_device);
		}

		/**
		 * This implements a column-wise scatter of the global data from the corresponding
		 * row. i.e. The data reduced from row 1 is broadcast to all blocks in
		 * column 1. It is assumed that the data to broadcast is located in the block on
		 * the diagonal.
		 */
		void columnScatter() {
			int current_device;
			cudaGetDevice(&current_device);

			// For each column in the decomposition:
			int32_t numRows = description->getBlockRows();
			std::vector<int32_t> blockIds;
			for (int32_t i = 0; i < numRows; i++) {
				// Put all the block ids for the column into a vector, with the ID of the diagonal block
				// at index 0.
				std::vector<int32_t> blockIds;
				blockIds.push_back(-1);
				for (int32_t j = 0; j < numRows; j++) {
					if (i == j) {
						blockIds[0] = description->getBlockId(j, i);
					}
					else {
						blockIds.push_back(description->getBlockId(j, i));
					}
				}

				// Do a binary tree scatter. At each step the primary buffer of the sender is
				// copied into the primary buffer of the receiver.
				int32_t max2pow = 2;
				while (max2pow < numRows) {
					max2pow *= 2;
				}
				for (int32_t j = max2pow; j >= 2; j /= 2) {
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the sender
							int32_t senderId = blockIds[id];

							// blockIds[id + j/2] is the sender
							int32_t receiverId = blockIds[id + j / 2];

							// Get the stream associated with the receiver's block id
							cudaStream_t stream = description->getBlockStreams()[receiverId];

							// Copy from the sender to the receiver (use stream associated with receiver)
							cudaMemcpyAsync(values[receiverId].Current(),
													values[senderId].Current(),
													sizeof(ValueType) * n,
													cudaMemcpyDefault,
													stream);
						}
					}
					// Synchronize all the active streams before next step.
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id + j/2] is the sender
							int32_t receiverId = blockIds[id + j / 2];

							// Set device and sync receiver's stream
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							cudaStreamSynchronize(description->getBlockStreams()[receiverId]);
						}
					}
				}
			}

			cudaSetDevice(current_device);
		}

		/**
		 * This implements a row-wise scatter of the global data from the corresponding
		 * column. i.e. The data reduced from column 1 is broadcast to all blocks in
		 * row 1. It is assumed that the data to broadcast is located in the block on
		 * the diagonal.
		 */
		void rowScatter() {
			int current_device;
			cudaGetDevice(&current_device);

			// For each row in the decomposition:
			int32_t numRows = description->getBlockRows();
			std::vector<int32_t> blockIds;
			for (int32_t i = 0; i < numRows; i++) {
				// Put all the block ids for the column into a vector, with the ID of the diagonal block
				// at index 0.
				std::vector<int32_t> blockIds;
				blockIds.push_back(-1);
				for (int32_t j = 0; j < numRows; j++) {
					if (i == j) {
						blockIds[0] = description->getBlockId(i, j);
					}
					else {
						blockIds.push_back(description->getBlockId(i, j));
					}
				}

				// Do a binary tree scatter. At each step the primary buffer of the sender is
				// copied into the primary buffer of the receiver.
				int32_t max2pow = 2;
				while (max2pow < numRows) {
					max2pow *= 2;
				}
				for (int32_t j = max2pow; j >= 2; j /= 2) {
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the sender
							int32_t senderId = blockIds[id];

							// blockIds[id + j/2] is the receiver
							int32_t receiverId = blockIds[id + j / 2];

							// Get the stream associated with the receiver's block id
							cudaStream_t stream = description->getBlockStreams()[receiverId];

							// Copy from the sender to the receiver (use stream associated with receiver)
							cudaMemcpyAsync(values[receiverId].Current(),
													values[senderId].Current(),
													sizeof(ValueType) * n,
													cudaMemcpyDefault,
													stream);
						}
					}
					// Sync all the active streams before next step
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id + j/2] is the receiver
							int32_t receiverId = blockIds[id + j / 2];

							// Set device and sync receiver's stream
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							cudaStreamSynchronize(description->getBlockStreams()[receiverId]);
						}
					}
				}
			}

			cudaSetDevice(current_device);
		}

		/**
		 * Outputs a human readable string representation of this Vertex2d object. This is only
		 * intended to be used for de-bugging.
		 * @return Human readable string representation
		 */
		std::string toString() {
			std::stringstream ss;
			ValueType* c = (ValueType*) malloc(sizeof(ValueType) * n);
			ValueType* a = (ValueType*) malloc(sizeof(ValueType) * n);

			int32_t numBlocks = description->getNumBlocks();

			ss << "Vertex2d:\n";
			for (int32_t i = 0; i < numBlocks; i++) {
				ss << "Block " << i << ":\n";
				ss << "Idx\tCur\tAlt\n";
				cudaMemcpy(c, values[i].Current(), sizeof(ValueType) * n, cudaMemcpyDefault);
				cudaMemcpy(a, values[i].Alternate(), sizeof(ValueType) * n, cudaMemcpyDefault);
				for (int32_t j = 0; j < n; j++) {
					ss << j << ":\t" << c[j] << "\t" << a[j] << "\n";
				}
			}

			free(c);
			free(a);

			return ss.str();
		}
	};

	template<typename GlobalType, typename LocalType, typename ValueType>
	class VertexData2D_Unbuffered {
		const MatrixDecompositionDescription<GlobalType, LocalType>* description;
		int32_t n;
		std::vector<ValueType*> values;

	public:
		/**
		 * Sets up a VertexData2D_Unbuffered object with an element allocated for each vertex
		 * in each block.
		 * @param descr Pointer to a MatrixDecompositionDescription object describing the layout
		 * of the 2D blocks.
		 */
		VertexData2D_Unbuffered(const MatrixDecompositionDescription<GlobalType, LocalType>* descr) :
				description(descr) {
			// Resize the values array to be the same size as number of blocks
			values.resize(descr->getNumBlocks());

			// Grab the current device id to switch back after allocations are done
			int current_device;
			cudaGetDevice(&current_device);
			LocalType allocSize = descr->getOffset();
			n = allocSize;
			// Allocate the data for each block
			for (size_t i = 0; i < descr->getDeviceAssignments().size(); i++) {
				int device = descr->getDeviceAssignments()[i];
				cudaSetDevice(device);
				cudaMalloc(&(values[i]), sizeof(ValueType) * n);
			}

			// Set the device back to what it was initially
			cudaSetDevice(current_device);
		}

		/**
		 * Sets up a VertexData2D_Unbuffered object with _n elements allocated per block.
		 * @param descr Pointer to a MatrixDecompositionDescription object describing the layout
		 * of the 2D blocks.
		 * @param _n The number of elements to allocate per block.
		 */
		VertexData2D_Unbuffered(const MatrixDecompositionDescription<GlobalType, LocalType>* descr,
										size_t _n) :
				description(descr), n(_n) {
			// Resize the values array to be the same size as number of blocks
			values.resize(descr->getNumBlocks());

			// Grab the current device id to switch back after allocations are done
			int current_device;
			cudaGetDevice(&current_device);
			// Allocate the data for each block
			for (size_t i = 0; i < descr->getDeviceAssignments().size(); i++) {
				int device = descr->getDeviceAssignments()[i];
				cudaSetDevice(device);
				cudaMalloc(&(values[i]), sizeof(ValueType) * n);
			}

			// Set the device back to what it was initially
			cudaSetDevice(current_device);
		}

		/**
		 * Destructor. Frees all allocated memory.
		 */
		~VertexData2D_Unbuffered() {
			for (size_t i = 0; i < values.size(); i++) {
				if (values[i]) {
					cudaFree(values[i]);
				}
			}
		}

		/**
		 * Fills the elements of the data array with the given value.
		 * The elements on the diagonal are filled with the given value. After filling,
		 * either rowScatter or columnScatter will copy the values across the blocks in
		 * either the rows or columns depending on the use.
		 * @param val The value to fill the array with
		 */
		void fillElements(ValueType val) {
			int current_device;
			cudaGetDevice(&current_device);
			int32_t numRows = description->getBlockRows();
			for (int32_t i = 0; i < numRows; i++) {
				int32_t blockId = description->getBlockId(i, i);
				ValueType* vals = get(blockId);
				int deviceId = description->getDeviceAssignments()[blockId];
				cudaStream_t stream = description->getBlockStreams()[blockId];
				cudaSetDevice(deviceId);
				thrust::fill(thrust::cuda::par.on(stream), vals, vals + n, val);
			}
			description->syncAllStreams();
			cudaSetDevice(current_device);
		}

		/**
		 * This implements a column-wise scatter of the global data from the corresponding
		 * row. i.e. The data reduced from row 1 is broadcast to all blocks in
		 * column 1. It is assumed that the data to broadcast is located in the block on
		 * the diagonal.
		 */
		void columnScatter() {
			int current_device;
			cudaGetDevice(&current_device);

			// For each column in the decomposition:
			int32_t numRows = description->getBlockRows();
			std::vector<int32_t> blockIds;
			for (int32_t i = 0; i < numRows; i++) {
				// Put all the block ids for the column into a vector, with the ID of the diagonal block
				// at index 0.
				std::vector<int32_t> blockIds;
				blockIds.push_back(-1);
				for (int32_t j = 0; j < numRows; j++) {
					if (i == j) {
						blockIds[0] = description->getBlockId(j, i);
					}
					else {
						blockIds.push_back(description->getBlockId(j, i));
					}
				}

				// Do a binary tree scatter. At each step the primary buffer of the sender is
				// copied into the primary buffer of the receiver.
				int32_t max2pow = 2;
				while (max2pow < numRows) {
					max2pow *= 2;
				}
				for (int32_t j = max2pow; j >= 2; j /= 2) {
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the sender
							int32_t senderId = blockIds[id];

							// blockIds[id + j/2] is the sender
							int32_t receiverId = blockIds[id + j / 2];

							// Get the stream associated with the receiver's block id
							cudaStream_t stream = description->getBlockStreams()[receiverId];

							// Copy from the sender to the receiver (use stream associated with receiver)
							cudaMemcpyAsync(values[receiverId],
													values[senderId],
													sizeof(ValueType) * n,
													cudaMemcpyDefault,
													stream);
						}
					}
					// Synchronize all the active streams before next step.
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id + j/2] is the sender
							int32_t receiverId = blockIds[id + j / 2];

							// Set device and sync receiver's stream
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							cudaStreamSynchronize(description->getBlockStreams()[receiverId]);
						}
					}
				}
			}

			cudaSetDevice(current_device);
		}

		/**
		 * This implements a row-wise scatter of the global data from the corresponding
		 * column. i.e. The data reduced from column 1 is broadcast to all blocks in
		 * row 1. It is assumed that the data to broadcast is located in the block on
		 * the diagonal.
		 */
		void rowScatter() {
			int current_device;
			cudaGetDevice(&current_device);

			// For each row in the decomposition:
			int32_t numRows = description->getBlockRows();
			std::vector<int32_t> blockIds;
			for (int32_t i = 0; i < numRows; i++) {
				// Put all the block ids for the column into a vector, with the ID of the diagonal block
				// at index 0.
				std::vector<int32_t> blockIds;
				blockIds.push_back(-1);
				for (int32_t j = 0; j < numRows; j++) {
					if (i == j) {
						blockIds[0] = description->getBlockId(i, j);
					}
					else {
						blockIds.push_back(description->getBlockId(i, j));
					}
				}

				// Do a binary tree scatter. At each step the primary buffer of the sender is
				// copied into the primary buffer of the receiver.
				int32_t max2pow = 2;
				while (max2pow < numRows) {
					max2pow *= 2;
				}
				for (int32_t j = max2pow; j >= 2; j /= 2) {
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id] is the sender
							int32_t senderId = blockIds[id];

							// blockIds[id + j/2] is the receiver
							int32_t receiverId = blockIds[id + j / 2];

							// Get the stream associated with the receiver's block id
							cudaStream_t stream = description->getBlockStreams()[receiverId];

							// Copy from the sender to the receiver (use stream associated with receiver)
							cudaMemcpyAsync(values[receiverId],
													values[senderId],
													sizeof(ValueType) * n,
													cudaMemcpyDefault,
													stream);
						}
					}
					// Sync all the active streams before next step
					for (int32_t id = 0; id < numRows; id++) {
						if (id % j == 0 && id + j / 2 < numRows) {
							// blockIds[id + j/2] is the receiver
							int32_t receiverId = blockIds[id + j / 2];

							// Set device and sync receiver's stream
							cudaSetDevice(description->getDeviceAssignments()[receiverId]);
							cudaStreamSynchronize(description->getBlockStreams()[receiverId]);
						}
					}
				}
			}

			cudaSetDevice(current_device);
		}

		/**
		 * Getter for n
		 * @return The value of n
		 */
		int32_t getN() {
			return n;
		}

		/**
		 * Gets the pointer to the allocated memory for a specified block.
		 * @param bId The block id to get the memory for.
		 * @return A pointer to the allocated memory for the given block.
		 */
		ValueType* get(int32_t bId) {
			return values[bId];
		}
	};

	/**
	 * This method takes in COO format matrix data and a MatrixDecompositionDescription and
	 * returns a Matrix2d object containing the given data.
	 */
	template<typename GlobalType, typename LocalType, typename ValueType>
	Matrix2d<GlobalType, LocalType, ValueType> COOto2d(MatrixDecompositionDescription<GlobalType,
																				LocalType> descr,
																		GlobalType* rowIds,
																		GlobalType* colIds,
																		ValueType* values) {
		// Grab the current device id to switch back after allocations are done
		int current_device;
		cudaGetDevice(&current_device);

		int32_t blockCount = descr.getNumBlocks();

		// Allocate array of size global nnz to hold the block labels
		int32_t* blockLabels = (int32_t*) malloc(descr.getNnz() * sizeof(int32_t));

		// Allocate array to contain row counts for each block and initialize to zero
		// Allocate array to contain position offsets for writing each blocks data
		LocalType* blockCounts = (LocalType*) malloc(blockCount * sizeof(LocalType));
		LocalType* blockPos = (LocalType*) malloc(blockCount * sizeof(LocalType));
		for (int i = 0; i < blockCount; i++) {
			blockCounts[i] = 0;
			blockPos[i] = 0;
		}

		// For each edge mark in the array the id of the block to which it will belong
		int32_t blockId;
		LocalType localRow;
		LocalType localCol;
		for (int i = 0; i < descr.getNnz(); i++) {
			descr.convertGlobaltoLocalRow(rowIds[i], colIds[i], localRow, localCol, blockId);
			blockLabels[i] = blockId;
			blockCounts[blockId]++;
		}

		// Allocate arrays for putting each blocks data into
		LocalType** blockRowIds = (LocalType**) malloc(blockCount * sizeof(LocalType*));
		LocalType** blockColIds = (LocalType**) malloc(blockCount * sizeof(LocalType*));
		ValueType** blockValues = NULL;
		if (values)
			blockValues = (ValueType**) malloc(blockCount * sizeof(ValueType*));
		for (int i = 0; i < blockCount; i++) {
			blockRowIds[i] = (LocalType*) malloc(blockCounts[i] * sizeof(LocalType));
			blockColIds[i] = (LocalType*) malloc(blockCounts[i] * sizeof(LocalType));
			if (values)
				blockValues[i] = (ValueType*) malloc(blockCounts[i] * sizeof(ValueType));
		}

		// Convert each blocks global rows to local ids and copy into block arrays
		for (int i = 0; i < descr.getNnz(); i++) {
			descr.convertGlobaltoLocalRow(rowIds[i], colIds[i], localRow, localCol, blockId);
			blockRowIds[blockId][blockPos[blockId]] = localRow;
			blockColIds[blockId][blockPos[blockId]] = localCol;
			if (values)
				blockValues[blockId][blockPos[blockId]] = values[i];
			blockPos[blockId]++;
		}

		// Allocate the result blocks vector
		std::vector<MultiValuedCsrGraph<LocalType, ValueType>*> blockVector(blockCount);

		// Convert each blocks COO rows into CSR and create it's graph object.
		for (int i = 0; i < blockCount; i++) {
			// Set the device as indicated so the data ends up on the right GPU
			cudaSetDevice(descr.getDeviceAssignments()[i]);
			cudaStream_t stream = descr.getBlockStreams()[i];

			if (blockCounts[i] > 0) {
				CSR_Result_Weighted<LocalType, ValueType> result;
				ConvertCOOtoCSR_weighted(blockRowIds[i],
													blockColIds[i],
													values ? blockValues[i] : NULL,
													(int64_t) blockCounts[i],
													(descr.getOffset() - 1),
													result);
				MultiValuedCsrGraph<LocalType, ValueType>* csrGraph = new MultiValuedCsrGraph<LocalType,
						ValueType>((size_t) result.size, (size_t) result.nnz, stream);
				if (values)
					csrGraph->allocateEdgeData(1, NULL);
				cudaMemcpy(csrGraph->get_raw_row_offsets(),
								result.rowOffsets,
								(result.size + 1) * sizeof(LocalType),
								cudaMemcpyDefault);
				cudaMemcpy(csrGraph->get_raw_column_indices(),
								result.colIndices,
								result.nnz * sizeof(LocalType),
								cudaMemcpyDefault);
				if (values)
					cudaMemcpy(csrGraph->get_raw_edge_dim(0),
									result.edgeWeights,
									result.nnz * sizeof(LocalType),
									cudaMemcpyDefault);
				blockVector[i] = csrGraph;
				result.Destroy();
			}
			else {
				MultiValuedCsrGraph<LocalType, ValueType>* csrGraph = new MultiValuedCsrGraph<LocalType,
						ValueType>((size_t) descr.getOffset(), (size_t) 0, stream);
				cudaMemset(	csrGraph->get_raw_row_offsets(),
								0,
								sizeof(LocalType) * (descr.getOffset() + 1));
				blockVector[i] = csrGraph;
			}
		}

		// Free temporary memory
		for (int i = 0; i < blockCount; i++) {
			free(blockRowIds[i]);
			free(blockColIds[i]);
			if (values)
				free(blockValues[i]);
		}
		free(blockRowIds);
		free(blockColIds);
		if (values)
			free(blockValues);

		cudaSetDevice(current_device);

		// Put it all together into a Matrix2d object for return
		return Matrix2d<GlobalType, LocalType, ValueType>(descr, blockVector);
	}
}
