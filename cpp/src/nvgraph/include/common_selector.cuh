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
//#pragma once

namespace nvlouvain{

template <typename T_ELEM> __inline__ __device__ T_ELEM __cachingLoad(const T_ELEM *addr) {
#if __CUDA_ARCH__ < 350
  return *addr;
#else
  return __ldg(addr);
#endif
}
__device__
inline float random_weight(int i, int j, int n)
{
#define RAND_MULTIPLIER   1145637293
  int i_min = (min(i, j) * RAND_MULTIPLIER) % n;
  int i_max = (max(i, j) * RAND_MULTIPLIER) % n;
  return ((float)i_max / n) * i_min;
}

/* WARNING: notice that based on the hexadecimal number in the last line 
   in the hash function the resulting floating point value is very likely 
   on the order of 0.5. */
__host__ __device__ inline unsigned int hash_val(unsigned int a, unsigned int seed)
{
  a ^= seed;
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) + (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a ^ 0xd3a2646c) + (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) + (a >> 16);
  return a;
}

/* return 1e-5 for float [sizeof(float)=4] and 1e-12 for double [sizeof(double)=8] types */
template<typename WeightType>
__host__ __device__ WeightType scaling_factor(){
    return (sizeof(WeightType) == 4) ? 1e-5f : 1e-12; 
}

// Kernel to compute the weight of the edges
// original version from AmgX.
template <typename IndexType, typename ValueType, typename WeightType>
__global__
void computeEdgeWeightsBlockDiaCsr_V2( const IndexType* row_offsets, const IndexType *row_indices, const IndexType *column_indices, 
                            const IndexType *dia_values, const ValueType* nonzero_values, const IndexType num_nonzero_blocks, 
          WeightType *str_edge_weights, WeightType *rand_edge_weights, int num_owned, int bsize, int component, int weight_formula)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  int i,j,kmin,kmax;
  int bsize_sq = bsize*bsize;
  WeightType den;

  int matrix_weight_entry = component*bsize+component;

  while (tid < num_nonzero_blocks)
  {
      i = row_indices[tid];
      j = column_indices[tid];

      if ((i != j) && (j < num_owned)) // skip diagonal and across-boundary edges
      {
        den = (WeightType) max(fabs(__cachingLoad(&nonzero_values[dia_values[i]*bsize_sq+matrix_weight_entry])),fabs(__cachingLoad(&nonzero_values[dia_values[j]*bsize_sq+matrix_weight_entry])));

        kmin = __cachingLoad(&row_offsets[j]); //kmin = row_offsets[j];
        kmax = __cachingLoad(&row_offsets[j+1]); //kmax = row_offsets[j+1];

        WeightType kvalue = 0.0;
        bool foundk = false; 
        for (int k=kmin;k<kmax;k++)
        {
          if ((column_indices[k] == i) /* && (column_indices[k] < num_owned) */) 
          {
            kvalue = __cachingLoad(&nonzero_values[k*bsize_sq+matrix_weight_entry]); //kvalue = nonzero_values[k*bsize_sq+matrix_weight_entry];
            foundk = true;
            break;
          }
        }

        // handles both symmetric & non-symmetric matrices
        WeightType ed_weight=0;
        if( foundk )
        {
            if( weight_formula == 0 )
                ed_weight =  0.5*(fabs(__cachingLoad(&nonzero_values[tid*bsize_sq+matrix_weight_entry])) + fabs(kvalue)) / den; // 0.5*(aij+aji)/max(a_ii,a_jj)
            else
                ed_weight = -0.5 * ( __cachingLoad(&nonzero_values[tid*bsize_sq+matrix_weight_entry]) / __cachingLoad(&nonzero_values[dia_values[i]*bsize_sq+matrix_weight_entry])  + // -0.5 * ( a_ij/a_ii +
                                    kvalue / __cachingLoad(&nonzero_values[dia_values[j]*bsize_sq+matrix_weight_entry]) );                                                            //          a_ji/a_jj )
        }


        // 05/09/13: Perturb the edge weights slightly to handle cases where edge weights are uniform
        WeightType small_fraction = scaling_factor<WeightType>()*hash_val(min(i,j),max(i,j))/UINT_MAX;
        ed_weight += small_fraction*ed_weight;
        str_edge_weights[tid] = ed_weight;

        // fill up random unique weights 
        if( rand_edge_weights != NULL )
          rand_edge_weights[tid] = random_weight(i, j, num_owned);
      }
      tid += gridDim.x*blockDim.x;
  }
}

// Kernel to compute the weight of the edges
// simple version modified for nvgraph
template <typename IndexType, typename ValueType, typename WeightType>
__global__
void computeEdgeWeights_simple( const IndexType* row_offsets, const IndexType *row_indices, const IndexType *column_indices, 
                            const ValueType *row_sum, const ValueType* nonzero_values, const IndexType num_nonzero_blocks, 
          WeightType *str_edge_weights, WeightType *rand_edge_weights, int n, int weight_formula)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  int i,j,kmin,kmax;
  WeightType den;

  while (tid < num_nonzero_blocks)
  {
      i = row_indices[tid];
      j = column_indices[tid];

      if ((i != j) && (j < n)) // skip diagonal and across-boundary edges
      {
        den = (WeightType) max(fabs(__cachingLoad(&row_sum[i])),fabs(__cachingLoad(&row_sum[j])));

        kmin = __cachingLoad(&row_offsets[j]); //kmin = row_offsets[j];
        kmax = __cachingLoad(&row_offsets[j+1]); //kmax = row_offsets[j+1];

        WeightType kvalue = 0.0;
        bool foundk = false; 
        for (int k=kmin;k<kmax;k++)
        {
          if ((column_indices[k] == i) /* && (column_indices[k] < n) */) 
          {
            kvalue = __cachingLoad(&nonzero_values[k]); //kvalue = nonzero_values[k];
            foundk = true;
            break;
          }
        }

        // handles both symmetric & non-symmetric matrices
        WeightType ed_weight=0;
        if( foundk )
        {
            if( weight_formula == 0 )
                ed_weight =  0.5*(fabs(__cachingLoad(&nonzero_values[tid])) + fabs(kvalue)) / den; // 0.5*(aij+aji)/max(a_ii,a_jj)
            else
                ed_weight = -0.5 * ( __cachingLoad(&nonzero_values[tid]) / __cachingLoad(&row_sum[i])  + // -0.5 * ( a_ij/a_ii +
                                    kvalue / __cachingLoad(&row_sum[j]) );                                                            //          a_ji/a_jj )
        }

        // 05/09/13: Perturb the edge weights slightly to handle cases where edge weights are uniform
        WeightType small_fraction = scaling_factor<WeightType>()*hash_val(min(i,j),max(i,j))/UINT_MAX;
        ed_weight += small_fraction*ed_weight;
        str_edge_weights[tid] = ed_weight;

        // fill up random unique weights 
        if( rand_edge_weights != NULL )
          rand_edge_weights[tid] = random_weight(i, j, n);
      }
      tid += gridDim.x*blockDim.x;
  }
}

// Kernel to compute the weight of the edges using geometry distance between edges
template <typename IndexType, typename ValueType>
__global__
void computeEdgeWeightsDistance3d( const int* row_offsets, const IndexType *column_indices, 
                                   const ValueType* gx, const ValueType* gy, const ValueType* gz, float *str_edge_weights, int num_rows)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  float lx, ly, lz;  
  float px, py, pz;
  int kmin, kmax;
  int col_id;

  while (tid < num_rows)
  {
      lx = gx[tid];
      ly = gy[tid];
      lz = gz[tid];
      kmin = row_offsets[tid];
      kmax = row_offsets[tid+1];

      for (int k=kmin;k<kmax;k++)
      {
        col_id = column_indices[k];
        if (col_id != tid)      // skip diagonal
        {
          px = gx[col_id];
          py = gy[col_id];
          pz = gz[col_id];

          str_edge_weights[k] =  1.0 / sqrt((px - lx)*(px - lx) + (py - ly)*(py - ly) + (pz - lz)*(pz - lz));
        }
      }
      tid += gridDim.x*blockDim.x;
  }
}

// Kernel that checks if perfect matchs exist
template <typename IndexType>
__global__
void matchEdges(const IndexType num_rows, IndexType *partner_index, IndexType *aggregates, const IndexType *strongest_neighbour)
{
  int potential_match, potential_match_neighbour;

  for (int tid= threadIdx.x + blockDim.x*blockIdx.x; tid < num_rows; tid += gridDim.x*blockDim.x)
  {
    if (partner_index[tid] == -1) // Unaggregated row
    {
      potential_match = strongest_neighbour[tid];
      if (potential_match!=-1)
      {
          potential_match_neighbour = strongest_neighbour[potential_match];

          if ( potential_match_neighbour == tid ) // we have a match
          {
            partner_index[tid] = potential_match;
            aggregates[tid] = ( potential_match > tid) ? tid : potential_match;
          }
      }
    }
  }
}

template <typename IndexType>
__global__
void joinExistingAggregates(IndexType num_rows, IndexType *aggregates, IndexType *aggregated, const IndexType *aggregates_candidate)
{
   int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  while (tid < num_rows)
  {
    if (aggregated[tid] == -1 && aggregates_candidate[tid] != -1) // Unaggregated row
    {
      aggregates[tid] = aggregates_candidate[tid];
      aggregated[tid] = 1;
    }

  tid += gridDim.x*blockDim.x;
  }
}


template<typename IndexType>
__global__
void aggregateSingletons( IndexType* aggregates, IndexType numRows )
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    while( tid < numRows )
    {
        if( aggregates[tid] == -1 ) //still unaggregated!
            aggregates[tid] = tid; //then become a singleton

        tid += gridDim.x*blockDim.x;
    }
}

__device__
inline float random_weight2(int i, int j)
{
#define RAND_MULTIPLIER         1145637293
  unsigned long i_min = (min(i, j) * RAND_MULTIPLIER);
  unsigned long i_max = (max(i, j) * RAND_MULTIPLIER);
  return ((float)i_min / i_max);
}


// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType>
__global__
void findStrongestNeighbourBlockDiaCsr_V2(const IndexType *row_offsets, const IndexType *column_indices,
                                        const float *edge_weights, IndexType n, IndexType *aggregates, 
          IndexType *strongest_neighbour_1phase, IndexType *strongest_neighbour,
          const size_t bsize, int phase, bool merge_singletons)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  
  float   weight;
  int jcol;

  while (tid < n)
  {
    int strongest_unaggregated = -1;
    int strongest_aggregated = -1;
    float   max_weight_unaggregated = 0.;
    float   max_weight_aggregated = 0.;
    if (aggregates[tid] == -1) // Unaggregated row
    {
      for (int j=row_offsets[tid]; j<row_offsets[tid+1]; j++)
      {
        //TODO: check if aggregated before computing the weight
        jcol = column_indices[j];
//        if (phase == 1) weight = edge_weights[j];
//        else weight = random_weight2(tid, jcol);
        weight = edge_weights[j];
//        printf("j: %d weight %f\n", j, weight);

        if (tid == jcol || jcol >= n) continue;  // skip diagonal and halo
        if (phase == 2 && strongest_neighbour_1phase[jcol] != tid) continue; // if 2nd phase only accept those who gave a hand on the 1st phase

        // Identify strongest aggregated and unaggregated neighbours
        if (aggregates[jcol] == -1 && (weight > max_weight_unaggregated || (weight==max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
        {
          max_weight_unaggregated= weight;
          strongest_unaggregated= jcol;
          // find the smallestt index with weight = max_weight
        }
        else if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight==max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
        {
           max_weight_aggregated = weight; 
           strongest_aggregated = jcol;
        }
      }
//      printf("-- phase: %d tid: %d strongest_neighbour: %d %f\n", phase, tid, strongest_neighbour[tid], max_weight_unaggregated);

      if (strongest_unaggregated == -1 && strongest_aggregated != -1) // All neighbours are aggregated
      {
        if( merge_singletons ){
            // Put in same aggregate as strongest neighbour
            aggregates[tid] = aggregates[strongest_aggregated];
        }
        else{
            aggregates[tid] = tid;
        }
      }
      else if (strongest_unaggregated != -1) {

        if (phase == 2) {
            float rand_w1 = random_weight2(tid, strongest_neighbour_1phase[tid]); 
          strongest_neighbour[tid] = max_weight_unaggregated > rand_w1 ? strongest_unaggregated : strongest_neighbour_1phase[tid];
        }
        else strongest_neighbour_1phase[tid] = strongest_unaggregated;
      
        //strongest_neighbour_1phase[tid] = strongest_unaggregated;
      }

      else {
        if (phase == 2) strongest_neighbour[tid] = strongest_neighbour_1phase[tid];
        else strongest_neighbour_1phase[tid] = tid;
      }
    }
/*
    if(tid<16)
      printf("++ phase: %d tid: %d strongest_neighbour: %d %f\n", phase, tid, strongest_neighbour[tid], max_weight_unaggregated);
 */
    tid += gridDim.x*blockDim.x;
 }
}

// Kernel that checks if perfect matchs exist
template <typename IndexType>
__global__
void matchEdges(const IndexType num_rows, IndexType *aggregates, const int *strongest_neighbour)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  int potential_match, potential_match_neighbour;

  while (tid < num_rows)
  {
    if (aggregates[tid] == -1) // Unaggregated row
    {
      potential_match = strongest_neighbour[tid];
      potential_match_neighbour = strongest_neighbour[potential_match];

      if (potential_match != -1 && potential_match_neighbour == tid) // we have a match
        aggregates[tid] = ( potential_match > tid ) ? tid : potential_match;
      /*
      if (potential_match != -1){
          potential_match_neighbour = strongest_neighbour[potential_match];

          if (potential_match_neighbour == tid) // we have a match
              aggregates[tid] = ( potential_match > tid ) ? tid : potential_match;
      }
      */
    }
  tid += gridDim.x*blockDim.x;
  }
}

template <typename IndexType, int block_size>
__global__
void countAggregates(const IndexType num_rows, const IndexType *aggregates, int *num_unaggregated)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x; 
  int c = 0;
  int i = tid;
  while( i < num_rows ) {
    c += ( aggregates[i] == -1 );   
    i += gridDim.x * blockDim.x;
  }
  __shared__ volatile int smem[block_size];
  smem[threadIdx.x] = c;  
  __syncthreads();

  for( int off = blockDim.x / 2; off >= 32; off = off / 2 ) {
    if( threadIdx.x < off ) 
      smem[threadIdx.x] += smem[threadIdx.x + off];
    __syncthreads();
  }

  // warp reduce
  if( threadIdx.x < 32 ) {
    smem[threadIdx.x] += smem[threadIdx.x+16];
    smem[threadIdx.x] += smem[threadIdx.x+8];
    smem[threadIdx.x] += smem[threadIdx.x+4];
    smem[threadIdx.x] += smem[threadIdx.x+2];
    smem[threadIdx.x] += smem[threadIdx.x+1];
  }

  if( threadIdx.x == 0 )
    atomicAdd(num_unaggregated, smem[0]);
}


template <typename IndexType>
__global__
void joinExistingAggregates(IndexType num_rows, IndexType *aggregates, const IndexType *aggregates_candidate)
{
   int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  while (tid < num_rows)
  {
    if (aggregates[tid] == -1 && aggregates_candidate[tid] != -1) // Unaggregated row
      aggregates[tid] = aggregates_candidate[tid];

    tid+=gridDim.x*blockDim.x;
  }
}



// Kernel that merges unaggregated vertices its strongest aggregated neighbour
// Weights are read from edge_weights array
// For block_dia_csr_matrix_format
template <typename IndexType>
__global__
void mergeWithExistingAggregatesBlockDiaCsr_V2(const IndexType *row_offsets, const IndexType *column_indices, const float *edge_weights,
                                            const int n, IndexType *aggregates, int bsize, const int deterministic, IndexType *aggregates_candidate)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  int jcol;
  float weight;
  
   
  while (tid < n)
  {
    float max_weight_aggregated = 0.;
    int strongest_aggregated = -1;
    if (aggregates[tid] == -1) // Unaggregated row
    {
      for (int j=row_offsets[tid]; j<row_offsets[tid+1]; j++)
      {
        // Compute edge weight
        weight = edge_weights[j];
        jcol = column_indices[j];

        if (jcol == tid || jcol >= n) continue;  // skip diagonal

        // Identify strongest aggregated neighbour
        if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight==max_weight_aggregated && jcol > strongest_aggregated))) // 
        {
           max_weight_aggregated = weight; 
           strongest_aggregated = jcol;
        }
      }

      if (strongest_aggregated != -1) // Found a neighbour to aggregate to
      {
        if (deterministic) {
          aggregates_candidate[tid] = aggregates[strongest_aggregated];
        }
        else {
          // Put in same aggregate as strongest neighbour
          aggregates[tid] = aggregates[strongest_aggregated];
        }
      }
      else // All neighbours are unaggregated, leave alone
      {
        if (deterministic)
          aggregates_candidate[tid] = tid;
        else
          aggregates[tid] = tid; 
      }


    }
  tid += gridDim.x*blockDim.x;
  }
}



template <typename INDEX_TYPE>
__global__ void computeDiagonalKernelCSR(INDEX_TYPE num_rows, const INDEX_TYPE *row_offsets, const INDEX_TYPE *col_indices, INDEX_TYPE *diag) {

  INDEX_TYPE row=(blockIdx.x*blockDim.x+threadIdx.x);

  while(row<num_rows) {
    int nz=row_offsets[row];
    int last_nz=row_offsets[row+1];
    //diag[row] = null_index;
    while(nz<last_nz) {
      int col=col_indices[nz];
      if(row==col) {
        diag[row]=nz;
        //diag_end_offsets[row]=nz+1;
        break;
      }
      nz++;
    }
    row+=blockDim.x*gridDim.x;
  }
}

template <typename T1, typename T2>
__global__ void convert_type(int n, const T1 *src, T2 *dest) {

  int tid=(blockIdx.x*blockDim.x+threadIdx.x);
  while(tid<n) 
  {
    dest[tid] = static_cast<T2>(src[tid]);
    tid += gridDim.x*blockDim.x;    
  }
}
}//nvlouvain

/*

// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType>
__global__
void agreeOnProposal(const IndexType *row_offsets, const IndexType *column_indices,
                                        IndexType num_block_rows, IndexType *aggregated, int *strongest_neighbour, float *weight_strongest_neighbour, IndexType *partner_index, int *aggregates)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  int partner;

  while(tid < num_block_rows)
  {
    if (aggregated[tid] == -1)
    {
      partner = partner_index[tid];
      float my_weight = weight_strongest_neighbour[tid];
      float partners_weight = -1;
      if (partner != -1) partners_weight = weight_strongest_neighbour[partner];

        if (my_weight < 0. && partners_weight < 0.) { // All neighbours are aggregated, leave in current aggregate
        //if (deterministic!=1)
        //{
          aggregated[tid] = 1; 
          strongest_neighbour[tid] = -1;
          partner_index[tid+num_block_rows] = tid;
          partner_index[tid+2*num_block_rows] = tid;
        //}
        }
        // if my weight is smaller than my partner's weight, change my strongest neighbour
        else if (my_weight < partners_weight)
          strongest_neighbour[tid] = strongest_neighbour[partner];

    }
  tid += gridDim.x*blockDim.x;
  }
}

// Kernel that checks if perfect matchs exist
template <typename IndexType>
__global__
void matchAggregates(IndexType *aggregates, IndexType *aggregated, IndexType *strongest_neighbour, const IndexType num_rows)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  int potential_match, potential_match_neighbour, my_aggregate;

  while (tid < num_rows)
  {
    if (aggregated[tid] == -1) // Unaggregated row
    {

      potential_match = strongest_neighbour[tid];
      if (potential_match!=-1)
      {
          potential_match_neighbour = strongest_neighbour[potential_match];

          my_aggregate = aggregates[tid];

          if (potential_match_neighbour == my_aggregate) // we have a match
          {
            aggregated[tid] = 1;
            aggregates[tid] = ( potential_match > my_aggregate) ? my_aggregate: potential_match;
          }
      }
    }
  tid += gridDim.x*blockDim.x;
  }
}

// Kernel that checks if perfect matchs exist
template <typename IndexType>
__global__
void assignUnassignedVertices(IndexType *partner_index, const IndexType num_rows)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;

  while (tid < num_rows)
  {
    if (partner_index[tid] == -1) // Unaggregated row
    {
      partner_index[tid] = tid;
    }
  tid += gridDim.x*blockDim.x;
  }
}

// Kernel that merges unaggregated vertices its strongest aggregated neighbour
// Edge weights are computed on the fly
// For block_dia_csr_matrix_format
template <typename IndexType, typename ValueType>
__global__
void mergeWithExistingAggregatesBlockDiaCsr(const IndexType *row_offsets, const IndexType *column_indices, const ValueType *dia_values, const ValueType *nonzero_values,
                                            const int n, IndexType *aggregates, int bsize, int deterministic, IndexType *aggregates_candidate)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  int jcol;
  ValueType weight;
  int bsize_sq = bsize*bsize;

  while (tid < n)
  {
    int strongest_aggregated = -1;
    ValueType max_weight_aggregated = 0.;
    if (aggregates[tid] == -1) // Unaggregated row
    {
      for (int j=row_offsets[tid]; j<row_offsets[tid+1]; j++)
      {
        jcol = column_indices[j];
        if (jcol >= n) continue;
        // Compute edge weight
        weight = fabs(nonzero_values[j*bsize_sq])/max( fabs(dia_values[tid*bsize_sq]),fabs(dia_values[jcol*bsize_sq]));

        // Identify strongest aggregated neighbour
        if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight==max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
        {
           max_weight_aggregated = weight; 
           strongest_aggregated = jcol;
        }
      }

      if (strongest_aggregated != -1) // Found a neighbour to aggregate to
      {
        if (deterministic) {
          aggregates_candidate[tid] = aggregates[strongest_aggregated];
        }
        else {
          // Put in same aggregate as strongest neighbour
          aggregates[tid] = aggregates[strongest_aggregated];
        }
      }
      else // All neighbours are unaggregated, leave alone
      {
        if (deterministic)
          aggregates_candidate[tid] = tid;
        else
          aggregates[tid] = tid; 
      }
    }
  tid += gridDim.x*blockDim.x;
  }
}

// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType>
__global__
void findStrongestNeighbourBlockDiaCsr_NoMerge(const IndexType *row_offsets, const IndexType *column_indices,
                                        float *edge_weights, const IndexType num_block_rows, IndexType* partner_index, int *strongest_neighbour, int deterministic)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  int jmin,jmax; 
  float weight;

  int jcol;

  while (tid < num_block_rows)
  {
    float max_weight_unaggregated = 0.;
    int strongest_unaggregated = -1;

    if (partner_index[tid] == -1) // Unaggregated row
    {
      jmin = row_offsets[tid];
      jmax = row_offsets[tid+1];

      for (int j=jmin; j<jmax; j++)
      {
        jcol = column_indices[j];
        if (tid == jcol || jcol >= num_block_rows) continue; // Skip diagonal and boundary edges.
        weight = edge_weights[j];
        // Identify strongest unaggregated neighbours
        if (partner_index[jcol] == -1 && (weight > max_weight_unaggregated || (weight==max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
        {
          max_weight_unaggregated= weight;
          strongest_unaggregated= jcol;
        }
      }

      if (strongest_unaggregated == -1) // All neighbours are aggregated
      {
        // Put in its own aggregate
        if (!deterministic)
          partner_index[tid] = tid;
      }
      else
      {
        strongest_neighbour[tid] = strongest_unaggregated;
      }

      //if (strongest_unaggregated != -1) // All neighbours are aggregated
      //  strongest_neighbour[tid] = strongest_unaggregated;
        // Put in its own aggregate
      //  partner_index[tid] = tid;
      //else


    }

  tid += gridDim.x*blockDim.x;
  }
}

// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType>
__global__
void findStrongestNeighbourBlockDiaCsr_StoreWeight(const IndexType *row_offsets, const IndexType *column_indices,
                                        const float *edge_weights, const IndexType num_block_rows, IndexType *aggregated, IndexType *aggregates, int *strongest_neighbour, IndexType *partner_index, float *weight_strongest_neighbour, int deterministic)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  float weight;

  int jcol,jmin,jmax;
  int agg_jcol;

  while (tid < num_block_rows)
  {
    float max_weight_unaggregated = 0.;
    float max_weight_aggregated = 0.;
    int strongest_unaggregated = -1;
    int strongest_aggregated = -1;
    int partner = -1;
    if (aggregated[tid] == -1) // Unaggregated row
    {
      partner = partner_index[tid];
      jmin = row_offsets[tid];
      jmax = row_offsets[tid+1];

      for (int j=jmin; j<jmax; j++)
      {
        jcol = column_indices[j];
        if (tid == jcol || jcol >= num_block_rows) continue; // Skip diagonal and boundary edges.
        weight = edge_weights[j];

        agg_jcol = aggregated[jcol];

        if (agg_jcol == -1 && jcol != partner && (weight > max_weight_unaggregated || (weight==max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
        {
            max_weight_unaggregated= weight;
            strongest_unaggregated= jcol;
        }
        else if (agg_jcol != -1 && jcol != partner && (weight > max_weight_aggregated || (weight==max_weight_aggregated && jcol > strongest_aggregated))) // unaggregated
        {
           max_weight_aggregated = weight; 
           strongest_aggregated = jcol;
        }
      }

      if (strongest_unaggregated== -1) // All neighbours are aggregated
      {
        if (!deterministic)
        {
          if (strongest_aggregated != -1) {
            aggregates[tid] = aggregates[strongest_aggregated];
            aggregated[tid] = 1;
            if (partner != -1) {
              aggregates[partner] = aggregates[strongest_aggregated];
              aggregated[partner] = 1;
      }
          }
          else {// leave in its own aggregate
            if (partner != -1) 
        aggregated[partner] = 1; 
            aggregated[tid] = 1; 
          }
        }

      }
      else // Found an unaggregated aggregate
      {
        weight_strongest_neighbour[tid] = max_weight_unaggregated;
        strongest_neighbour[tid] = aggregates[strongest_unaggregated];
      }
    }
  tid += gridDim.x*blockDim.x;
  }
}

// findStrongestNeighbour kernel for block_dia_csr_matrix format
// computes weight on the fly
template <typename IndexType, typename ValueType>
__global__
void findStrongestNeighbourBlockDiaCsr(const IndexType *row_offsets, const IndexType *column_indices, 
                            const ValueType *dia_values, const ValueType *nonzero_values, const IndexType n, IndexType *aggregates, int *strongest_neighbour, int bsize)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  ValueType weight;

  int jcol;
  int bsize_sq = bsize*bsize;

  while (tid < n)
  {
    ValueType max_weight_unaggregated = 0.;
    ValueType max_weight_aggregated = 0.;
    int strongest_unaggregated = -1;
    int strongest_aggregated = -1;
    if (aggregates[tid] == -1) // Unaggregated row
    {
      for (int j=row_offsets[tid]; j<row_offsets[tid+1]; j++)
      {
        jcol = column_indices[j];
        if (jcol >= n) continue;

        // Compute edge weight
        for (int k=row_offsets[jcol];k<row_offsets[jcol+1];k++)
        {
          if (column_indices[k] == tid)
          {
            weight = 0.5*(fabs(nonzero_values[j*bsize_sq]) + fabs(nonzero_values[k*bsize_sq]))
                     / max( fabs(dia_values[tid*bsize_sq]),fabs(dia_values[jcol*bsize_sq]));
            break;
          }
        }

        // Identify strongest aggregated and unaggregated neighbours
        if (aggregates[jcol] == -1 && (weight > max_weight_unaggregated || (weight==max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
        {
          max_weight_unaggregated= weight;
          strongest_unaggregated= jcol;
        }
        else if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight==max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
        {
           max_weight_aggregated = weight; 
           strongest_aggregated = jcol;
        }
      }
      if (strongest_unaggregated == -1 && strongest_aggregated != -1) // All neighbours are aggregated
        // Put in same aggregate as strongest neighbour
        aggregates[tid] = aggregates[strongest_aggregated];
      else if (strongest_unaggregated != -1)
        strongest_neighbour[tid] = strongest_unaggregated;
      else
        strongest_neighbour[tid] = tid;
    }
  tid += gridDim.x*blockDim.x;
  }
}

// Kernel that merges unaggregated vertices its strongest aggregated neighbour
// Weights are read from edge_weights array
// For block_dia_csr_matrix_format
template <typename IndexType>
__global__
void mergeWithExistingAggregatesBlockDiaCsr(const IndexType *row_offsets, const IndexType *column_indices, const float *edge_weights,
                                            const int num_block_rows, IndexType *aggregates, IndexType *aggregated, int deterministic, IndexType *aggregates_candidate, bool allow_singletons = true)
{
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  
  int jcol;

  float weight;

  while (tid < num_block_rows)
  {
    float max_weight_aggregated = 0.;
    int strongest_aggregated = -1;
    if (aggregated[tid] == -1) // Unaggregated row
    {
      for (int j=row_offsets[tid]; j<row_offsets[tid+1]; j++)
      {
        jcol = column_indices[j];
        if (tid == jcol || jcol >= num_block_rows) continue; // Skip diagonal and boundary edges.
        // Identify strongest aggregated neighbour
        if (aggregated[jcol] != -1) {

          weight = edge_weights[j];
          if (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated)) {
           max_weight_aggregated = weight; 
           strongest_aggregated = jcol;
          }

        }
      }

      if (strongest_aggregated != -1) {
        if (deterministic)
        {
          aggregates_candidate[tid] = aggregates[strongest_aggregated];
        }
        else
        {
          // Put in same aggregate as strongest neighbour
          aggregates[tid] = aggregates[strongest_aggregated];
          aggregated[tid] = 1;
        }
      }
      else // All neighbours are unaggregated, leave alone
      {
        if (deterministic) {
          if (allow_singletons) aggregates_candidate[tid] = tid;
        }
        else
          aggregates[tid] = tid; 
      }

    }
  tid += gridDim.x*blockDim.x;
  }
}

// Kernel to extract diagonal for csr_matrix format
template <typename IndexType, typename ValueType>
__global__
void getDiagonalKernel(const IndexType *offsets, const IndexType *column_indices,
                       const ValueType *values, const IndexType numRows, ValueType *diagonal)
{
  int tIdx = threadIdx.x + blockDim.x*blockIdx.x;

  while (tIdx < numRows)
  {
    const int offset = offsets[tIdx];
    const int numj = offsets[tIdx+1]-offset;

    for (int j=offset; j < offset+numj; j++)
    {
      int jcol = column_indices[j];
      if (tIdx == jcol)
      {
        diagonal[tIdx] = values[j];
      }
    }
  tIdx += gridDim.x*blockDim.x;
  }
}

template <typename INDEX_TYPE>
__global__ void computeDiagonalKernelCOO(INDEX_TYPE num_nz, INDEX_TYPE *row_indices, INDEX_TYPE *col_indices, INDEX_TYPE *diag) {
  //BLOCKY*BLOCKX threads per nz
  INDEX_TYPE nz=(blockIdx.x*blockDim.x+threadIdx.x);

  while(nz<num_nz) {
    INDEX_TYPE row=row_indices[nz];
    INDEX_TYPE col=col_indices[nz];

    if(row==col) {
      //copy block to diag
      diag[row]=nz;
      //diag_end_offsets[row]=nz+1;
    }

    nz+=blockDim.x*gridDim.x;
  }
}

// Kernel to extract diagonal for csr_matrix format
template <typename IndexType, typename ValueType>
__global__
void getDiagonalKernelNoDiaProp(const IndexType *dia_idx, const ValueType *values, const IndexType numRows, ValueType *diagonal)
{
  int tIdx = threadIdx.x + blockDim.x*blockIdx.x;

  while (tIdx < numRows)
  {
    diagonal[tIdx] = values[dia_idx[tIdx]];
    tIdx += gridDim.x*blockDim.x;
  }
}



*/
