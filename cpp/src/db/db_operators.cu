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

#include <db/db_operators.cuh>
#include <cub/device/device_select.cuh>

namespace cugraph { 
  namespace db {
    template<typename IndexType>
    struct degree_iterator {
      IndexType* offsets;
      degree_iterator(IndexType* _offsets) :
          offsets(_offsets) {
      }

      __host__ __device__
      IndexType operator[](IndexType place) {
        return offsets[place + 1] - offsets[place];
      }
    };

    template<typename It, typename IndexType>
    struct deref_functor {
      It iterator;
      deref_functor(It it) :
          iterator(it) {
      }

      __host__ __device__
      IndexType operator()(IndexType in) {
        return iterator[in];
      }
    };

    template<typename idx_t, typename flag_t>
    struct notNegativeOne {
      __host__ __device__
      flag_t operator()(idx_t in) {
        return in != -1;
      }
    };

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
    __global__ void compute_bucket_offsets_kernel(const IndexType *frontier_degrees_exclusive_sum,
                                                  IndexType *bucket_offsets,
                                                  const IndexType frontier_size,
                                                  IndexType total_degree) {
      IndexType end = ((total_degree - 1 + FIND_MATCHES_BLOCK_SIZE) / FIND_MATCHES_BLOCK_SIZE);

      for (IndexType bid = blockIdx.x * blockDim.x + threadIdx.x;
          bid <= end;
          bid += gridDim.x * blockDim.x) {

        IndexType eid = min(bid * FIND_MATCHES_BLOCK_SIZE, total_degree - 1);

        bucket_offsets[bid] = binsearch_maxle(frontier_degrees_exclusive_sum,
                                              eid,
                                              (IndexType) 0,
                                              frontier_size - 1);

      }
    }

    template<typename idx_t>
    __global__ void findMatchesKernel(idx_t inputSize,
                                      idx_t outputSize,
                                      idx_t maxBlock,
                                      idx_t* offsets,
                                      idx_t* indirection,
                                      idx_t* blockStarts,
                                      idx_t* expandCounts,
                                      idx_t* frontier,
                                      idx_t* columnA,
                                      idx_t* columnB,
                                      idx_t* columnC,
                                      idx_t* outputA,
                                      idx_t* outputB,
                                      idx_t* outputC,
                                      idx_t* outputD,
                                      idx_t patternA,
                                      idx_t patternB,
                                      idx_t patternC) {
      __shared__ idx_t blockRange[2];
      __shared__ idx_t localExSum[FIND_MATCHES_BLOCK_SIZE * 2];
      __shared__ idx_t localFrontier[FIND_MATCHES_BLOCK_SIZE * 2];

      for (idx_t bid = blockIdx.x; bid < maxBlock; bid += gridDim.x) {
        // Copy in the block's section of the expand counts
        if (threadIdx.x == 0) {
          blockRange[0] = blockStarts[bid];
          blockRange[1] = blockStarts[bid + 1];
          if (blockRange[0] > 0) {
            blockRange[0] -= 1;
          }
        }
        __syncthreads();

        idx_t sectionSize = blockRange[1] - blockRange[0];
        for (int tid = threadIdx.x; tid <= sectionSize; tid += blockDim.x) {
          localExSum[tid] = expandCounts[blockRange[0] + tid];
          localFrontier[tid] = frontier[blockRange[0] + tid];
        }
        __syncthreads();

        // Do the work item for each thread of this virtual block:
        idx_t tid = bid * blockDim.x + threadIdx.x;
        if (tid < outputSize) {
          // Figure out which row this thread/iteration is working on
          idx_t sourceIdx = binsearch_maxle(localExSum, tid, (idx_t)0, (idx_t)sectionSize);
          idx_t source = localFrontier[sourceIdx];
          idx_t rank = tid - localExSum[sourceIdx];
          idx_t row_id = indirection[offsets[source] + rank];

          // Load in values from the row for A, B, and C columns
          idx_t valA = columnA[row_id];
          idx_t valB = columnB[row_id];
          idx_t valC = columnC[row_id];

          // Debugging output
//          printf("RowId: %d, valA: %d, valB: %d, valC: %d\n", row_id, valA, valB, valC);
//          printf("PatternA: %d, PatternB: %d, PatternC: %d\n", patternA, patternB, patternC);

          // Compare the row values with constants in the pattern
          bool matchA = outputA != nullptr ? true : patternA == valA;
          bool matchB = outputB != nullptr ? true : patternB == valB;
          bool matchC = outputC != nullptr ? true : patternC == valC;

          // If row doesn't match, set row values to -1 before writing out
          if (!(matchA && matchB && matchC)) {
            valA = -1;
            valB = -1;
            valC = -1;
            row_id = -1;
          }

          // Write out values to non-null outputs
          if (outputA != nullptr)
            outputA[tid] = valA;
          if (outputB != nullptr)
            outputB[tid] = valB;
          if (outputC != nullptr)
            outputC[tid] = valC;
          if (outputD != nullptr)
            outputD[tid] = row_id;
        }
      }
    }

    template<typename idx_t>
    db_result<idx_t> findMatches(db_pattern<idx_t>& pattern,
                        db_table<idx_t>& table,
                        gdf_column* frontier,
                        int indexPosition) {
      // Find out if the indexPosition is a variable or constant
      bool indexConstant = !pattern.getEntry(indexPosition).isVariable();

      db_column_index<idx_t>& theIndex = table.getIndex(indexPosition);

      // Check to see whether we are going to be saving out the row ids from matches
      bool saveRowIds = false;
      if (pattern.getSize() == 4)
        saveRowIds = true;

      // Check if we have a frontier to use, if we don't make one up
      bool givenInputFrontier = frontier != nullptr;
      idx_t frontierSize;
      idx_t* frontier_ptr = nullptr;
      if (givenInputFrontier) {
        frontier_ptr = (idx_t*)frontier->data;
        frontierSize = frontier->size;
      }
      else {
        if (indexConstant) {
          // Use a single value equal to the constant in the pattern
          idx_t constantValue = pattern.getEntry(indexPosition).getConstant();
          ALLOC_TRY(&frontier_ptr, sizeof(idx_t), nullptr);
          thrust::fill(rmm::exec_policy(nullptr)->on(nullptr),
                       frontier_ptr,
                       frontier_ptr + 1,
                       constantValue);
          frontierSize = 1;
        }
        else {
          // Making a sequence of values from zero to n where n is the highest ID present in the index.
          idx_t highestId = theIndex.getOffsets()->size - 2;
          ALLOC_TRY(&frontier_ptr, sizeof(idx_t) * (highestId + 1), nullptr);
          thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                           frontier_ptr,
                           frontier_ptr + highestId + 1);
          frontierSize = highestId + 1;
        }
      }

      // Collect all the pointers needed to run the main kernel
      idx_t* columnA = (idx_t*)table.getColumn(0)->data;
      idx_t* columnB = (idx_t*)table.getColumn(1)->data;
      idx_t* columnC = (idx_t*)table.getColumn(2)->data;
      idx_t* offsets = (idx_t*)theIndex.getOffsets()->data;
      idx_t* indirection = (idx_t*)theIndex.getIndirection()->data;

      // Load balance the input
      idx_t *exsum_degree = nullptr;
      ALLOC_TRY(&exsum_degree, sizeof(idx_t) * (frontierSize + 1), nullptr);
      degree_iterator<idx_t> deg_it(offsets);
      deref_functor<degree_iterator<idx_t>, idx_t> deref(deg_it);
      thrust::fill(rmm::exec_policy(nullptr)->on(nullptr), exsum_degree, exsum_degree + 1, 0);
      thrust::transform(rmm::exec_policy(nullptr)->on(nullptr),
                        frontier_ptr,
                        frontier_ptr + frontierSize,
                        exsum_degree + 1,
                        deref);
      thrust::inclusive_scan(rmm::exec_policy(nullptr)->on(nullptr),
                             exsum_degree + 1,
                             exsum_degree + frontierSize + 1,
                             exsum_degree + 1);
      idx_t output_size;
      cudaMemcpy(&output_size, &exsum_degree[frontierSize], sizeof(idx_t), cudaMemcpyDefault);

      // Debugging output
//      std::cout << "OutputSize = " << output_size << "\n";

      idx_t num_blocks = (output_size + FIND_MATCHES_BLOCK_SIZE - 1) / FIND_MATCHES_BLOCK_SIZE;
      idx_t *block_bucket_offsets = nullptr;
      ALLOC_TRY(&block_bucket_offsets, sizeof(idx_t) * (num_blocks + 1), nullptr);

      dim3 grid, block;
      block.x = 512;
      grid.x = min((idx_t) MAXBLOCKS, (num_blocks / 512) + 1);
      compute_bucket_offsets_kernel<<<grid, block, 0, nullptr>>>(exsum_degree,
                                                                 block_bucket_offsets,
                                                                 frontierSize,
                                                                 output_size);

      // Allocate space for the result
      idx_t *outputA = nullptr;
      idx_t *outputB = nullptr;
      idx_t *outputC = nullptr;
      idx_t *outputD = nullptr;
      if (pattern.getEntry(0).isVariable()) {
        ALLOC_TRY(&outputA, sizeof(idx_t) * output_size, nullptr);
      }
      if (pattern.getEntry(1).isVariable()) {
        ALLOC_TRY(&outputB, sizeof(idx_t) * output_size, nullptr);
      }
      if (pattern.getEntry(2).isVariable()) {
        ALLOC_TRY(&outputC, sizeof(idx_t) * output_size, nullptr);
      }
      if (saveRowIds) {
        ALLOC_TRY(&outputD, sizeof(idx_t) * output_size, nullptr);
      }

      // Get the constant pattern entries from the pattern to pass into the main kernel
      idx_t patternA = -1;
      idx_t patternB = -1;
      idx_t patternC = -1;
      if (!pattern.getEntry(0).isVariable()) {
        patternA = pattern.getEntry(0).getConstant();
      }
      if (!pattern.getEntry(1).isVariable()) {
        patternB = pattern.getEntry(1).getConstant();
      }
      if (!pattern.getEntry(2).isVariable()) {
        patternC = pattern.getEntry(2).getConstant();
      }

      // Call the main kernel
      block.x = FIND_MATCHES_BLOCK_SIZE;
      grid.x = min((idx_t) MAXBLOCKS,
                   (output_size + (idx_t) FIND_MATCHES_BLOCK_SIZE - 1)
                       / (idx_t) FIND_MATCHES_BLOCK_SIZE);
      findMatchesKernel<<<grid, block, 0, nullptr>>>(frontierSize,
                                                     output_size,
                                                     num_blocks,
                                                     offsets,
                                                     indirection,
                                                     block_bucket_offsets,
                                                     exsum_degree,
                                                     frontier_ptr,
                                                     columnA,
                                                     columnB,
                                                     columnC,
                                                     outputA,
                                                     outputB,
                                                     outputC,
                                                     outputD,
                                                     patternA,
                                                     patternB,
                                                     patternC);

      // Debugging output
//      if (outputA != nullptr) {
//        idx_t* outputA_h = (idx_t*)malloc(sizeof(idx_t) * output_size);
//        cudaMemcpy(outputA_h, outputA, sizeof(idx_t)*output_size, cudaMemcpyDefault);
//        std::cout << "OutputA: ";
//        for (int i = 0; i < output_size; i++)
//          std::cout << outputA_h[i] << " ";
//        std::cout << "\n";
//        free(outputA_h);
//      }
//      if (outputB != nullptr) {
//        idx_t* outputB_h = (idx_t*) malloc(sizeof(idx_t) * output_size);
//        cudaMemcpy(outputB_h, outputB, sizeof(idx_t) * output_size, cudaMemcpyDefault);
//        std::cout << "OutputB: ";
//        for (int i = 0; i < output_size; i++)
//          std::cout << outputB_h[i] << " ";
//        std::cout << "\n";
//        free(outputB_h);
//      }
//      if (outputC != nullptr) {
//        idx_t* outputC_h = (idx_t*) malloc(sizeof(idx_t) * output_size);
//        cudaMemcpy(outputC_h, outputC, sizeof(idx_t) * output_size, cudaMemcpyDefault);
//        std::cout << "OutputC: ";
//        for (int i = 0; i < output_size; i++)
//          std::cout << outputC_h[i] << " ";
//        std::cout << "\n";
//        free(outputC_h);
//      }
//      if (outputD != nullptr) {
//        idx_t* outputD_h = (idx_t*) malloc(sizeof(idx_t) * output_size);
//        cudaMemcpy(outputD_h, outputD, sizeof(idx_t) * output_size, cudaMemcpyDefault);
//        std::cout << "OutputD: ";
//        for (int i = 0; i < output_size; i++)
//          std::cout << outputD_h[i] << " ";
//        std::cout << "\n";
//        free(outputD_h);
//      }

      // Get the non-null output columns
      std::vector<idx_t*> columns;
      std::vector<std::string> names;
      if (outputA != nullptr) {
        columns.push_back(outputA);
        names.push_back(pattern.getEntry(0).getVariable());
      }
      if (outputB != nullptr) {
        columns.push_back(outputB);
        names.push_back(pattern.getEntry(1).getVariable());
      }
      if (outputC != nullptr) {
        columns.push_back(outputC);
        names.push_back(pattern.getEntry(2).getVariable());
      }
      if (outputD != nullptr) {
        columns.push_back(outputD);
        names.push_back(pattern.getEntry(3).getVariable());
      }

      // Remove non-matches from result
      int8_t* flags = nullptr;
      ALLOC_TRY(&flags, sizeof(int8_t) * output_size, nullptr);
      idx_t* col_ptr = columns[0];
      thrust::transform(rmm::exec_policy(nullptr)->on(nullptr),
                        col_ptr,
                        col_ptr + output_size,
                        flags,
                        notNegativeOne<idx_t, int8_t>());

      void* tempSpace = nullptr;
      size_t tempSpaceSize = 0;
      idx_t* compactSize_d = nullptr;
      ALLOC_TRY(&compactSize_d, sizeof(idx_t), nullptr);
      cub::DeviceSelect::Flagged(tempSpace,
                                 tempSpaceSize,
                                 col_ptr,
                                 flags,
                                 col_ptr,
                                 compactSize_d,
                                 output_size);
      ALLOC_TRY(&tempSpace, tempSpaceSize, nullptr);
      cub::DeviceSelect::Flagged(tempSpace,
                                 tempSpaceSize,
                                 col_ptr,
                                 flags,
                                 col_ptr,
                                 compactSize_d,
                                 output_size);
      idx_t compactSize_h;
      cudaMemcpy(&compactSize_h, compactSize_d, sizeof(idx_t), cudaMemcpyDefault);
      for (size_t i = 1; i < columns.size(); i++) {
        col_ptr = columns[i];
        cub::DeviceSelect::Flagged(tempSpace,
                                   tempSpaceSize,
                                   col_ptr,
                                   flags,
                                   col_ptr,
                                   compactSize_d,
                                   output_size);
      }

      // Put together the result to return
      db_result<idx_t> result;
      for (size_t i = 0; i < names.size(); i++) {
        result.addColumn(names[i]);
      }
      result.allocateColumns(compactSize_h);
      for (size_t i = 0; i < columns.size(); i++) {
        idx_t* outputPtr = result.getData(names[i]);
        idx_t* inputPtr = columns[i];
        cudaMemcpy(outputPtr, inputPtr, sizeof(idx_t) * compactSize_h, cudaMemcpyDefault);
      }

      // Clean up allocations
      if (!givenInputFrontier)
        ALLOC_FREE_TRY(frontier_ptr, nullptr);
      ALLOC_FREE_TRY(exsum_degree, nullptr);
      ALLOC_FREE_TRY(block_bucket_offsets, nullptr);
      ALLOC_FREE_TRY(tempSpace, nullptr);
      ALLOC_FREE_TRY(compactSize_d, nullptr);
      ALLOC_FREE_TRY(flags, nullptr);
      if (outputA != nullptr)
        ALLOC_FREE_TRY(outputA, nullptr);
      if (outputB != nullptr)
        ALLOC_FREE_TRY(outputB, nullptr);
      if (outputC != nullptr)
        ALLOC_FREE_TRY(outputC, nullptr);
      if (outputD != nullptr)
        ALLOC_FREE_TRY(outputD, nullptr);

      // Return the result
      return result;
    }

    template db_result<int32_t> findMatches(db_pattern<int32_t>& pattern,
                                            db_table<int32_t>& table,
                                            gdf_column* frontier,
                                            int indexPosition);
    template db_result<int64_t> findMatches(db_pattern<int64_t>& pattern,
                                            db_table<int64_t>& table,
                                            gdf_column* frontier,
                                            int indexPosition);
} } //namespace
