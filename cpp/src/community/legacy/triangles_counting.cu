/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>
#include <raft/cudart_utils.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/iterator/counting_iterator.h>

#include "cub/cub.cuh"

#define TH_CENT_K_LOCLEN (34)
#define WP_LEN_TH1       (24)
#define WP_LEN_TH2       (2)

#if WP_LEN_TH1 > 32
#error WP_LEN_TH1 must be <= 32!
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define THREADS      (128)
#define DIV_UP(a, b) (((a) + ((b)-1)) / (b))
#define BITSOF(x)    (sizeof(*x) * 8)

#define BLK_BWL0 (128)

#define DEG_THR1 (3.5)
#define DEG_THR2 (38.0)

namespace cugraph {
namespace triangle {

namespace {  // anonym.

template <typename T>
struct type_utils;

template <>
struct type_utils<int> {
  typedef int LOCINT;
};

template <>
struct type_utils<int64_t> {
  typedef uint64_t LOCINT;
};

template <typename T>
struct spmat_t {
  T N;
  T nnz;
  T nrows;
  const T* roff_d;
  const T* rows_d;
  const T* cols_d;
  bool is_lower_triangular;
};

template <typename T>
size_t bitmap_roundup(size_t n)
{
  size_t size = DIV_UP(n, 8 * sizeof(T));
  size        = size_t{8} * DIV_UP(size * sizeof(T), 8);
  size /= sizeof(T);
  return size;
}

template <typename InputIteratorT, typename OutputIteratorT>
static inline void cubSum(InputIteratorT d_in,
                          OutputIteratorT d_out,
                          int num_items,
                          cudaStream_t stream    = 0,
                          bool debug_synchronous = false)
{
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
  RAFT_CHECK_CUDA(stream);

  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  cub::DeviceReduce::Sum(
    d_temp_storage.data(), temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
  RAFT_CHECK_CUDA(stream);

  return;
}

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename NumSelectedIteratorT,
          typename SelectOp>
static inline void cubIf(InputIteratorT d_in,
                         OutputIteratorT d_out,
                         NumSelectedIteratorT d_num_selected_out,
                         int num_items,
                         SelectOp select_op,
                         cudaStream_t stream    = 0,
                         bool debug_synchronous = false)
{
  size_t temp_storage_bytes = 0;

  cub::DeviceSelect::If(nullptr,
                        temp_storage_bytes,
                        d_in,
                        d_out,
                        d_num_selected_out,
                        num_items,
                        select_op,
                        stream,
                        debug_synchronous);
  RAFT_CHECK_CUDA(stream);

  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  cub::DeviceSelect::If(d_temp_storage.data(),
                        temp_storage_bytes,
                        d_in,
                        d_out,
                        d_num_selected_out,
                        num_items,
                        select_op,
                        stream,
                        debug_synchronous);
  RAFT_CHECK_CUDA(stream);

  return;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ T __block_bcast(const T v, const int x)
{
  __shared__ T shv;

  __syncthreads();
  if (threadIdx.x == x) shv = v;
  __syncthreads();

  return shv;
}

template <int BDIM_X, int BDIM_Y, int WSIZE, typename T>
__device__ __forceinline__ T block_sum(T v)
{
  __shared__ T sh[BDIM_X * BDIM_Y / WSIZE];

  const int lid = threadIdx.x % 32;
  const int wid = threadIdx.x / 32 + ((BDIM_Y > 1) ? threadIdx.y * (BDIM_X / 32) : 0);

#pragma unroll
  for (int i = WSIZE / 2; i; i >>= 1) {
    v += __shfl_down_sync(raft::warp_full_mask(), v, i);
  }
  if (lid == 0) sh[wid] = v;

  __syncthreads();
  if (wid == 0) {
    v = (lid < (BDIM_X * BDIM_Y / WSIZE)) ? sh[lid] : 0;

#pragma unroll
    for (int i = (BDIM_X * BDIM_Y / WSIZE) / 2; i; i >>= 1) {
      v += __shfl_down_sync(raft::warp_full_mask(), v, i);
    }
  }
  return v;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <int BDIM,
          int WSIZE,
          int BWL0,
          typename ROW_T,
          typename OFF_T,
          typename CNT_T,
          typename MAP_T>
__global__ void tricnt_b2b_k(const ROW_T ner,
                             const ROW_T* __restrict__ rows,
                             const OFF_T* __restrict__ roff,
                             const ROW_T* __restrict__ cols,
                             CNT_T* __restrict__ ocnt,
                             MAP_T* __restrict__ bmapL0,
                             const size_t bmldL0,
                             MAP_T* __restrict__ bmapL1,
                             const size_t bmldL1)
{
  CNT_T __cnt = 0;

  bmapL1 += bmldL1 * blockIdx.x;
  bmapL0 += bmldL0 * blockIdx.x;
  for (ROW_T bid = blockIdx.x; bid < ner; bid += gridDim.x) {
    const OFF_T rbeg = roff[rows[bid]];
    const OFF_T rend = roff[rows[bid] + 1];

    ROW_T firstcol = 0;
    ROW_T lastcol  = 0;

    for (OFF_T i = rbeg; i < rend; i += BDIM) {
      const ROW_T c = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;

      __syncthreads();
      if (c > -1) {
        atomicOr(bmapL1 + c / BITSOF(bmapL1), ((MAP_T)1) << (c % BITSOF(bmapL1)));
        atomicOr(bmapL0 + c / BWL0 / BITSOF(bmapL0), ((MAP_T)1) << ((c / BWL0) % BITSOF(bmapL0)));
      }
      __syncthreads();

#pragma unroll
      for (int j = 0; j < BDIM; j++) {
        const ROW_T curc = __block_bcast(c, j);
        if (curc == -1) break;

        lastcol = curc;
        if ((i == rbeg) && !j) {
          firstcol = curc;
          continue;
        }
        const OFF_T soff = roff[curc];
        const OFF_T eoff = roff[curc + 1];

        for (OFF_T k = eoff - 1; k >= soff; k -= BDIM) {
          if (k - (int)threadIdx.x < soff) break;

          const ROW_T cc = __ldg(cols + k - threadIdx.x);
          if (cc < firstcol) break;

          MAP_T mm = ((MAP_T)1) << ((cc / BWL0) % BITSOF(bmapL0));
          if (0 == (bmapL0[cc / BWL0 / BITSOF(bmapL0)] & mm)) continue;

          mm = ((MAP_T)1) << (cc % BITSOF(bmapL1));
          if (bmapL1[cc / BITSOF(bmapL1)] & mm) { __cnt++; }
        }
      }
    }

    lastcol /= 64;
    firstcol /= 64;

    __syncthreads();
    for (int i = rbeg; i < rend; i += BDIM) {
      if (i + threadIdx.x < rend) {
        ROW_T c                           = cols[i + threadIdx.x];
        bmapL1[c / BITSOF(bmapL1)]        = 0;
        bmapL0[c / BWL0 / BITSOF(bmapL0)] = 0;
      }
    }
    __syncthreads();
  }

  __cnt = block_sum<BDIM, 1, WSIZE>(__cnt);
  if (threadIdx.x == 0) ocnt[blockIdx.x] = __cnt;

  return;
}

template <typename T>
void tricnt_b2b(T nblock,
                spmat_t<T>* m,
                uint64_t* ocnt_d,
                unsigned int* bmapL0_d,
                size_t bmldL0,
                unsigned int* bmapL1_d,
                size_t bmldL1,
                cudaStream_t stream)
{
  // still best overall (with no psum)
  tricnt_b2b_k<THREADS, 32, BLK_BWL0><<<nblock, THREADS, 0, stream>>>(
    m->nrows, m->rows_d, m->roff_d, m->cols_d, ocnt_d, bmapL0_d, bmldL0, bmapL1_d, bmldL1);
  RAFT_CHECK_CUDA(stream);
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <int BDIM_X, int BDIM_Y, int WSIZE, typename T>
__device__ __forceinline__ T block_sum_sh(T v, T* sh)
{
  const int lid = threadIdx.x % 32;
  const int wid = threadIdx.x / 32 + ((BDIM_Y > 1) ? threadIdx.y * (BDIM_X / 32) : 0);

#pragma unroll
  for (int i = WSIZE / 2; i; i >>= 1) {
    v += __shfl_down_sync(raft::warp_full_mask(), v, i);
  }
  if (lid == 0) sh[wid] = v;

  __syncthreads();
  if (wid == 0) {
    v = (lid < (BDIM_X * BDIM_Y / WSIZE)) ? sh[lid] : 0;

#pragma unroll
    for (int i = (BDIM_X * BDIM_Y / WSIZE) / 2; i; i >>= 1) {
      v += __shfl_down_sync(raft::warp_full_mask(), v, i);
    }
  }
  return v;
}

template <int BDIM, int WSIZE, typename ROW_T, typename OFF_T, typename CNT_T>
__global__ void tricnt_bsh_k(const ROW_T ner,
                             const ROW_T* __restrict__ rows,
                             const OFF_T* __restrict__ roff,
                             const ROW_T* __restrict__ cols,
                             CNT_T* __restrict__ ocnt,
                             const size_t bmld)
{
  CNT_T __cnt = 0;
  extern __shared__ unsigned int shm[];

  for (int i = 0; i < bmld; i += BDIM) {
    if (i + threadIdx.x < bmld) { shm[i + threadIdx.x] = 0; }
  }

  for (ROW_T bid = blockIdx.x; bid < ner; bid += gridDim.x) {
    const OFF_T rbeg = roff[rows[bid]];
    const OFF_T rend = roff[rows[bid] + 1];

    ROW_T firstcol = 0;
    ROW_T lastcol  = 0;

    for (OFF_T i = rbeg; i < rend; i += BDIM) {
      const ROW_T c = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;

      __syncthreads();
      if (c > -1) atomicOr(shm + c / BITSOF(shm), 1u << (c % BITSOF(shm)));
      __syncthreads();

#pragma unroll
      for (int j = 0; j < BDIM; j++) {
        const ROW_T curc = __block_bcast(c, j);
        if (curc == -1) break;

        lastcol = curc;
        if ((i == rbeg) && !j) {
          firstcol = curc;
          continue;
        }

        const OFF_T soff = roff[curc];
        const OFF_T eoff = roff[curc + 1];
        for (OFF_T k = eoff - 1; k >= soff; k -= BDIM) {
          if (k - (int)threadIdx.x < soff) break;

          const ROW_T cc = __ldg(cols + k - threadIdx.x);
          if (cc < firstcol) break;

          const unsigned int mm = 1u << (cc % BITSOF(shm));
          if (shm[cc / BITSOF(shm)] & mm) { __cnt++; }
        }
      }
    }
    lastcol /= 64;
    firstcol /= 64;

    __syncthreads();
    if (lastcol - firstcol < rend - rbeg) {
      for (int i = firstcol; i <= lastcol; i += BDIM) {
        if (i + threadIdx.x <= lastcol) { ((unsigned long long*)shm)[i + threadIdx.x] = 0ull; }
      }
    } else {
      for (int i = rbeg; i < rend; i += BDIM) {
        if (i + threadIdx.x < rend) { shm[cols[i + threadIdx.x] / BITSOF(shm)] = 0; }
      }
    }
    __syncthreads();
  }
  __cnt = block_sum_sh<BDIM, 1, WSIZE>(__cnt, (uint64_t*)shm);
  if (threadIdx.x == 0) ocnt[blockIdx.x] = __cnt;

  return;
}

template <typename T>
void tricnt_bsh(T nblock, spmat_t<T>* m, uint64_t* ocnt_d, size_t bmld, cudaStream_t stream)
{
  tricnt_bsh_k<THREADS, 32><<<nblock, THREADS, sizeof(unsigned int) * bmld, stream>>>(
    m->nrows, m->rows_d, m->roff_d, m->cols_d, ocnt_d, bmld);
  RAFT_CHECK_CUDA(stream);
  return;
}

////////////////////////////////////////////////////////////////////////////////////////
template <int WSIZE,
          int NWARP,
          int RLEN_THR1,
          int RLEN_THR2,
          typename ROW_T,
          typename OFF_T,
          typename CNT_T,
          typename MAP_T>
__global__ void tricnt_wrp_ps_k(const ROW_T ner,
                                const ROW_T* __restrict__ rows,
                                const OFF_T* __restrict__ roff,
                                const ROW_T* __restrict__ cols,
                                CNT_T* __restrict__ ocnt,
                                MAP_T* __restrict__ bmap,
                                const size_t bmld)
{
  __shared__ OFF_T sho[NWARP][WSIZE];
  __shared__ ROW_T shs[NWARP][WSIZE];
  __shared__ ROW_T shc[NWARP][WSIZE];

  CNT_T __cnt = 0;
  ROW_T wid   = blockIdx.x * blockDim.y + threadIdx.y;

  bmap += bmld * wid;
  for (; wid < ner; wid += gridDim.x * blockDim.y) {
    const OFF_T rbeg = roff[rows[wid]];
    const OFF_T rend = roff[rows[wid] + 1];

    // RLEN_THR1 <= 32
    if (rend - rbeg <= RLEN_THR1) {
      const int nloc = rend - rbeg;

      OFF_T soff;
      OFF_T eoff;
      if (threadIdx.x < nloc) {
        const ROW_T c                 = cols[rbeg + threadIdx.x];
        shc[threadIdx.y][threadIdx.x] = c;
        soff                          = roff[c];
        eoff                          = roff[c + 1];
      }

      int mysm = -1;

#pragma unroll
      for (int i = 1; i < RLEN_THR1; i++) {
        if (i == nloc) break;

        const OFF_T csoff = __shfl_sync(raft::warp_full_mask(), soff, i);
        const OFF_T ceoff = __shfl_sync(raft::warp_full_mask(), eoff, i);

        if (ceoff - csoff < RLEN_THR2) {
          if (threadIdx.x == i) mysm = i;
          continue;
        }
        for (OFF_T k = ceoff - 1; k >= csoff; k -= WSIZE) {
          if (k - (int)threadIdx.x < csoff) break;

          const ROW_T cc = cols[k - threadIdx.x];
          if (cc < shc[threadIdx.y][0]) break;
          for (int j = i - 1; j >= 0; j--) {
            if (cc == shc[threadIdx.y][j]) { __cnt++; }
          }
        }
      }
      if (mysm > -1) {
        for (OFF_T k = eoff - 1; k >= soff; k--) {
          const ROW_T cc = cols[k];
          if (cc < shc[threadIdx.y][0]) break;
          for (int j = mysm - 1; j >= 0; j--) {
            if (cc == shc[threadIdx.y][j]) { __cnt++; }
          }
        }
      }
    } else {
      ROW_T firstcol = cols[rbeg];
      ROW_T lastcol  = cols[rend - 1];
      for (OFF_T i = rbeg; i < rend; i += 32) {
        const ROW_T c = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;

        if (c > -1) atomicOr(bmap + c / BITSOF(bmap), ((MAP_T)1) << (c % BITSOF(bmap)));
      }

      for (OFF_T i = rbeg; i < rend; i += 32) {
        const ROW_T c                 = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;
        sho[threadIdx.y][threadIdx.x] = (c > -1) ? roff[c] : 0;
        shc[threadIdx.y][threadIdx.x] = c;

        ROW_T len    = (c > -1) ? roff[c + 1] - sho[threadIdx.y][threadIdx.x] : 0;
        ROW_T lensum = len;

#pragma unroll
        for (int j = 1; j < 32; j <<= 1) {
          lensum += (threadIdx.x >= j) * (__shfl_up_sync(raft::warp_full_mask(), lensum, j));
        }
        shs[threadIdx.y][threadIdx.x] = lensum - len;

        lensum = __shfl_sync(raft::warp_full_mask(), lensum, 31);

        int k = WSIZE - 1;
        for (int j = lensum - 1; j >= 0; j -= WSIZE) {
          if (j < threadIdx.x) break;

          // bisect-right
          for (; k >= 0; k--) {
            if (shs[threadIdx.y][k] <= j - threadIdx.x) break;
          }

          const ROW_T cc =
            __ldg(cols + (sho[threadIdx.y][k] + j - threadIdx.x - shs[threadIdx.y][k]));

          if (cc < shc[threadIdx.y][k]) continue;

          const MAP_T mm = ((MAP_T)1) << (cc % BITSOF(bmap));
          if (bmap[cc / BITSOF(bmap)] & mm) { __cnt++; }
        }
      }
      lastcol /= 64;
      firstcol /= 64;

      if (lastcol - firstcol < rend - rbeg) {
        for (int i = firstcol; i <= lastcol; i += WSIZE) {
          if (i + threadIdx.x <= lastcol) { ((unsigned long long*)bmap)[i + threadIdx.x] = 0ull; }
        }
      } else {
        for (int i = rbeg; i < rend; i += WSIZE) {
          if (i + threadIdx.x < rend) { bmap[cols[i + threadIdx.x] / BITSOF(bmap)] = 0; }
        }
      }
    }
  }
  __syncthreads();
  __cnt = block_sum<WSIZE, NWARP, WSIZE>(__cnt);
  if (threadIdx.x == 0 && threadIdx.y == 0) { ocnt[blockIdx.x] = __cnt; }
  return;
}

template <typename T>
void tricnt_wrp(
  T nblock, spmat_t<T>* m, uint64_t* ocnt_d, unsigned int* bmap_d, size_t bmld, cudaStream_t stream)
{
  dim3 block(32, THREADS / 32);
  tricnt_wrp_ps_k<32, THREADS / 32, WP_LEN_TH1, WP_LEN_TH2>
    <<<nblock, block, 0, stream>>>(m->nrows, m->rows_d, m->roff_d, m->cols_d, ocnt_d, bmap_d, bmld);
  RAFT_CHECK_CUDA(stream);
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <int BDIM, int LOCLEN, typename ROW_T, typename OFF_T, typename CNT_T>
__global__ void tricnt_thr_k(const ROW_T ner,
                             const ROW_T* __restrict__ rows,
                             const OFF_T* __restrict__ roff,
                             const ROW_T* __restrict__ cols,
                             CNT_T* __restrict__ ocnt)
{
  CNT_T __cnt     = 0;
  const ROW_T tid = blockIdx.x * BDIM + threadIdx.x;

  for (ROW_T rid = tid; rid < ner; rid += gridDim.x * BDIM) {
    const ROW_T r = rows[rid];

    const OFF_T rbeg = roff[r];
    const OFF_T rend = roff[r + 1];
    const ROW_T rlen = rend - rbeg;

    if (!rlen) continue;
    if (rlen <= LOCLEN) {
      int nloc = 0;
      ROW_T loc[LOCLEN];

#pragma unroll
      for (nloc = 0; nloc < LOCLEN; nloc++) {
        if (rbeg + nloc >= rend) break;
        loc[nloc] = __ldg(cols + rbeg + nloc);
      }

#pragma unroll
      for (int i = 1; i < LOCLEN; i++) {
        if (i == nloc) break;

        const ROW_T c    = loc[i];
        const OFF_T soff = roff[c];
        const OFF_T eoff = roff[c + 1];

        for (OFF_T k = eoff - 1; k >= soff; k--) {
          const ROW_T cc = __ldg(cols + k);
          if (cc < loc[0]) break;

          for (int j = i - 1; j >= 0; j--) {
            if (cc == loc[j]) __cnt++;
          }
        }
      }
    } else {
      const ROW_T minc = cols[rbeg];
      for (int i = 1; i < rlen; i++) {
        const ROW_T c    = __ldg(cols + rbeg + i);
        const OFF_T soff = roff[c];
        const OFF_T eoff = roff[c + 1];

        for (OFF_T k = eoff - 1; k >= soff; k--) {
          const ROW_T cc = __ldg(cols + k);
          if (cc < minc) break;

          for (int j = i - 1; j >= 0; j--) {
            if (cc == __ldg(cols + rbeg + j)) __cnt++;
          }
        }
      }
    }
  }

  __syncthreads();
  __cnt = block_sum<BDIM, 1, 32>(__cnt);
  if (threadIdx.x == 0) ocnt[blockIdx.x] = __cnt;

  return;
}

template <typename T>
void tricnt_thr(T nblock, spmat_t<T>* m, uint64_t* ocnt_d, cudaStream_t stream)
{
  cudaFuncSetCacheConfig(tricnt_thr_k<THREADS,
                                      TH_CENT_K_LOCLEN,
                                      typename type_utils<T>::LOCINT,
                                      typename type_utils<T>::LOCINT,
                                      uint64_t>,
                         cudaFuncCachePreferL1);

  tricnt_thr_k<THREADS, TH_CENT_K_LOCLEN>
    <<<nblock, THREADS, 0, stream>>>(m->nrows, m->rows_d, m->roff_d, m->cols_d, ocnt_d);
  RAFT_CHECK_CUDA(stream);
  return;
}

/////////////////////////////////////////////////////////////////
template <typename IndexType>
struct NonEmptyRow {
  const IndexType* p_roff;
  __host__ __device__ NonEmptyRow(const IndexType* roff) : p_roff(roff) {}
  __host__ __device__ __forceinline__ bool operator()(const IndexType& a) const
  {
    return (p_roff[a] < p_roff[a + 1]);
  }
};

template <typename T>
void create_nondangling_vector(
  const T* roff, T* p_nonempty, T* n_nonempty, size_t n, cudaStream_t stream)
{
  if (n <= 0) return;
  thrust::counting_iterator<T> it(0);
  NonEmptyRow<T> temp_func(roff);
  rmm::device_vector<T> out_num(*n_nonempty);

  cubIf(it, p_nonempty, out_num.data().get(), n, temp_func, stream);
  cudaMemcpy(n_nonempty, out_num.data().get(), sizeof(*n_nonempty), cudaMemcpyDeviceToHost);
  RAFT_CHECK_CUDA(stream);
}

template <typename T>
uint64_t reduce(uint64_t* v_d, T n, cudaStream_t stream)
{
  rmm::device_vector<uint64_t> tmp(1);

  cubSum(v_d, tmp.data().get(), n, stream);
  RAFT_CHECK_CUDA(stream);

  return tmp[0];
}

template <typename IndexType>
class TrianglesCount {
 private:
  uint64_t m_triangles_number;
  spmat_t<IndexType> m_mat;
  int m_shared_mem_per_block{};
  int m_multi_processor_count{};
  int m_max_threads_per_multi_processor{};

  rmm::device_vector<IndexType> m_seq;

  cudaStream_t m_stream;

  bool m_done;

  void tcount_bsh();
  void tcount_b2b();
  void tcount_wrp();
  void tcount_thr();

 public:
  // Simple constructor
  TrianglesCount(IndexType num_vertices,
                 IndexType num_edges,
                 IndexType const* row_offsets,
                 IndexType const* col_indices,
                 cudaStream_t stream = NULL);

  void count();
  inline uint64_t get_triangles_count() const { return m_triangles_number; }
};

template <typename IndexType>
TrianglesCount<IndexType>::TrianglesCount(IndexType num_vertices,
                                          IndexType num_edges,
                                          IndexType const* row_offsets,
                                          IndexType const* col_indices,
                                          cudaStream_t stream)
  : m_mat{num_vertices, num_edges, num_vertices, row_offsets, nullptr, col_indices},
    m_stream{stream},
    m_done{true}
{
  int device_id;
  cudaGetDevice(&device_id);

  cudaDeviceGetAttribute(&m_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
  RAFT_CHECK_CUDA(m_stream);
  cudaDeviceGetAttribute(&m_multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
  RAFT_CHECK_CUDA(m_stream);
  cudaDeviceGetAttribute(
    &m_max_threads_per_multi_processor, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
  RAFT_CHECK_CUDA(m_stream);

  m_seq.resize(m_mat.N, IndexType{0});
  create_nondangling_vector(m_mat.roff_d, m_seq.data().get(), &(m_mat.nrows), m_mat.N, m_stream);
  m_mat.rows_d = m_seq.data().get();
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_bsh()
{
  CUGRAPH_EXPECTS(not(m_shared_mem_per_block * 8 < m_mat.nrows),
                  "Number of vertices too high for TrainglesCount.");
  /// if (m_shared_mem_per_block * 8 < (size_t)m_mat.nrows) {
  ///  FatalError("Number of vertices too high to use this kernel!", NVGRAPH_ERR_BAD_PARAMETERS);
  ///}

  size_t bmld = bitmap_roundup<uint32_t>(m_mat.N);
  int nblock  = m_mat.nrows;

  rmm::device_vector<uint64_t> ocnt_d(nblock, uint64_t{0});

  tricnt_bsh(nblock, &m_mat, ocnt_d.data().get(), bmld, m_stream);
  m_triangles_number = reduce(ocnt_d.data().get(), nblock, m_stream);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_b2b()
{
  // allocate a big enough array for output

  rmm::device_vector<uint64_t> ocnt_d(m_mat.nrows, uint64_t{0});

  size_t bmldL1 = bitmap_roundup<uint32_t>(m_mat.N);

  size_t free_bytes, total_bytes;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  RAFT_CHECK_CUDA(m_stream);

  size_t nblock_available = (free_bytes * 95 / 100) / (sizeof(uint32_t) * bmldL1);

  int nblock = static_cast<int>(MIN(nblock_available, static_cast<size_t>(m_mat.nrows)));

  // allocate level 1 bitmap
  rmm::device_vector<uint32_t> bmapL1_d(bmldL1 * nblock, uint32_t{0});

  // allocate level 0 bitmap
  size_t bmldL0 = bitmap_roundup<uint32_t>(DIV_UP(m_mat.N, BLK_BWL0));
  rmm::device_vector<uint32_t> bmapL0_d(nblock * bmldL0, uint32_t{0});

  tricnt_b2b(nblock,
             &m_mat,
             ocnt_d.data().get(),
             bmapL0_d.data().get(),
             bmldL0,
             bmapL1_d.data().get(),
             bmldL1,
             m_stream);
  m_triangles_number = reduce(ocnt_d.data().get(), nblock, m_stream);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_wrp()
{
  // allocate a big enough array for output
  rmm::device_vector<uint64_t> ocnt_d(DIV_UP(m_mat.nrows, (THREADS / 32)), uint64_t{0});

  size_t bmld = bitmap_roundup<uint32_t>(m_mat.N);

  // number of blocks limited by birmap size
  size_t free_bytes, total_bytes;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  RAFT_CHECK_CUDA(m_stream);

  size_t nblock_available = (free_bytes * 95 / 100) / (sizeof(uint32_t) * bmld * (THREADS / 32));

  int nblock = static_cast<int>(
    MIN(nblock_available, static_cast<size_t>(DIV_UP(m_mat.nrows, (THREADS / 32)))));

  size_t bmap_sz = bmld * nblock * (THREADS / 32);

  rmm::device_vector<uint32_t> bmap_d(bmap_sz, uint32_t{0});

  tricnt_wrp(nblock, &m_mat, ocnt_d.data().get(), bmap_d.data().get(), bmld, m_stream);
  m_triangles_number = reduce(ocnt_d.data().get(), nblock, m_stream);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_thr()
{
  int maxblocks = m_multi_processor_count * m_max_threads_per_multi_processor / THREADS;

  int nblock = MIN(maxblocks, DIV_UP(m_mat.nrows, THREADS));

  rmm::device_vector<uint64_t> ocnt_d(nblock, uint64_t{0});

  tricnt_thr(nblock, &m_mat, ocnt_d.data().get(), m_stream);
  m_triangles_number = reduce(ocnt_d.data().get(), nblock, m_stream);
}

template <typename IndexType>
void TrianglesCount<IndexType>::count()
{
  double mean_deg = (double)m_mat.nnz / m_mat.nrows;
  if (mean_deg < DEG_THR1)
    tcount_thr();
  else if (mean_deg < DEG_THR2)
    tcount_wrp();
  else {
    const int shMinBlkXSM = 6;
    if (static_cast<size_t>(m_shared_mem_per_block * 8 / shMinBlkXSM) <
        static_cast<size_t>(m_mat.N))
      tcount_b2b();
    else
      tcount_bsh();
  }
}

}  // namespace

template <typename VT, typename ET, typename WT>
uint64_t triangle_count(legacy::GraphCSRView<VT, ET, WT> const& graph)
{
  TrianglesCount<VT> counter(
    graph.number_of_vertices, graph.number_of_edges, graph.offsets, graph.indices);

  counter.count();
  return counter.get_triangles_count();
}

template uint64_t triangle_count<int32_t, int32_t, float>(
  legacy::GraphCSRView<int32_t, int32_t, float> const&);

}  // namespace triangle
}  // namespace cugraph
