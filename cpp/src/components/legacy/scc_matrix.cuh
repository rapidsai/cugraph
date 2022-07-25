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
#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <type_traits>
//

// Convergence check logic;
//
/**
 * @brief Provide convergence check logic for GEMM SCC via a device pointer
 */
struct CStableChecker {
  explicit CStableChecker(int flag) : d_flag_(1, flag) {}

  // hopefully might be cheaper than copying the value from device to host:
  //
  bool is_set(void) const
  {
    auto pos = thrust::find(d_flag_.begin(), d_flag_.end(), 1);
    return (pos != d_flag_.end());
  }

  void set(int flag)
  {
    thrust::for_each(d_flag_.begin(), d_flag_.end(), [flag] __device__(int& val) { val = flag; });
  }

  int* get_ptr(void) { return d_flag_.data().get(); }

 private:
  thrust::device_vector<int> d_flag_;
};

/**
 * @brief SCC Algorithm
 * (Adapted from John Gilbert's "Graph Algorithms in the Language of Linear Algebra")
 *
 * C = I | A | A^2 |...| A^k
 * where matrix multiplication is through
 * (combine, reduce) == (&, |) (bitwise ops)
 * Then: X = C & transpose(C);
 * apply get_labels(X);
 */
template <typename ByteT, typename IndexT = int>
struct SCC_Data {
  SCC_Data(size_t nrows,
           const IndexT* p_d_r_o,  // row_offsets
           const IndexT* p_d_c_i)
    :  // column indices
      nrows_(nrows),
      p_d_r_o_(p_d_r_o),
      p_d_c_i_(p_d_c_i),
      d_C(nrows * nrows, 0),
      d_Cprev(nrows * nrows, 0),
      p_d_C_(d_C.data().get())
  {
    init();
  }

  ByteT const* get_Cptr(void) const { return p_d_C_; }

  size_t nrows(void) const { return nrows_; }

  const IndexT* r_o(void) const { return p_d_r_o_; }

  const IndexT* c_i(void) const { return p_d_c_i_; }

  // protected: cannot have device lambda inside protected memf
  void init(void)
  {
    // init d_Cprev to identity:
    //
    auto* p_d_Cprev = d_Cprev.data().get();
    size_t n = nrows_;  // for lambda capture, since I cannot capture `this` (host), or `nrows_`
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(nrows_),
      [p_d_Cprev, n] __device__(size_t indx) { p_d_Cprev[indx * n + indx] = ByteT{1}; });
  }

  void get_labels(IndexT* d_labels) const
  {
    size_t n = nrows_;  // for lambda capture, since I cannot capture `this` (host), or `nrows_`
    thrust::transform(thrust::device,
                      thrust::make_counting_iterator<IndexT>(0),
                      thrust::make_counting_iterator<IndexT>(nrows_),
                      d_labels,
                      [n, p_d_C = p_d_C_] __device__(IndexT k) {
                        auto begin = p_d_C + k * n;
                        auto end   = begin + n;
                        ByteT one{1};

                        auto pos = thrust::find_if(
                          thrust::seq, begin, end, [one](IndexT entry) { return (entry == one); });

                        // if( pos != end ) // always the case, because C starts as I + A
                        return IndexT(pos - begin);
                      });
  }

  size_t run_scc(IndexT* d_labels)
  {
    size_t nrows = nrows_;
    size_t count = 0;

    ByteT* p_d_Cprev = get_Cprev().data().get();

    size_t n2            = nrows * nrows;
    const IndexT* p_d_ro = r_o();
    const IndexT* p_d_ci = c_i();

    CStableChecker flag(0);
    int* p_d_flag = flag.get_ptr();
    do {
      flag.set(0);

      thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(n2),
        [nrows, p_d_C = p_d_C_, p_d_Cprev, p_d_flag, p_d_ro, p_d_ci] __device__(size_t indx) {
          ByteT one{1};

          auto i = indx / nrows;
          auto j = indx % nrows;

          if ((i == j) || (p_d_Cprev[indx] == one)) {
            p_d_C[indx] = one;
          } else {
            // this ammounts to A (^,v) B
            // (where A = adjacency matrix defined by (p_ro, p_ci),
            //  B := p_d_Cprev; (^,v) := (*,+) semiring);
            // Here's why:
            // (A (^,v) B)[i][j] := A[i][.] (^,v) B[j][.]
            // (where X[i][.] := i-th row of X;
            //        X[.][j] := j-th column of X);
            // which is:
            // 1, iff A[i][.] and B[j][.] have a 1 in the same location,
            // 0, otherwise;
            //
            // i.e., corresponfing entry in p_d_C is 1
            // if B[k][j] == 1 for any column k in A's i-th row;
            // hence, for each column k of row A[i][.],
            // which is the set:
            // k \in {p_ci + p_ro[i], ..., p_ci + p_ro[i+1] - 1},
            // check if (B[k][j] == 1),
            // i.e., p_d_Cprev[k*nrows + j]) == 1:
            //
            auto begin = p_d_ci + p_d_ro[i];
            auto end   = p_d_ci + p_d_ro[i + 1];
            auto pos   = thrust::find_if(
              thrust::seq, begin, end, [one, j, nrows, p_d_Cprev, p_d_ci](IndexT k) {
                return (p_d_Cprev[k * nrows + j] == one);
              });

            if (pos != end) p_d_C[indx] = one;
          }

          if (p_d_C[indx] != p_d_Cprev[indx])
            *p_d_flag = 1;  // race-condition: harmless,
                            // worst case many threads
                            // write the _same_ value
        });
      ++count;
      cudaDeviceSynchronize();

      std::swap(p_d_C_, p_d_Cprev);  // Note 1: this swap makes `p_d_Cprev` the
                                     // most recently updated matrix pointer
                                     // at the end of this loop
                                     // (see `Note 2` why this matters);
    } while (flag.is_set());

    // C & Ct:
    // This is the actual reason we need both C and Cprev:
    // to avoid race condition on C1 = C0 & transpose(C0):
    //
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(n2),
                     [nrows, p_d_C = p_d_C_, p_d_Cprev] __device__(size_t indx) {
                       auto i     = indx / nrows;
                       auto j     = indx % nrows;
                       auto tindx = j * nrows + i;

                       // Note 2: per Note 1, p_d_Cprev is latest:
                       //
                       p_d_C[indx] = (p_d_Cprev[indx]) & (p_d_Cprev[tindx]);
                     });

    get_labels(d_labels);

    return count;
  }

 private:
  size_t nrows_;
  const IndexT* p_d_r_o_;  // row_offsets
  const IndexT* p_d_c_i_;  // column indices
  thrust::device_vector<ByteT> d_C;
  thrust::device_vector<ByteT> d_Cprev;
  ByteT* p_d_C_{nullptr};  // holds the most recent update,
  // which can have storage in any of d_C or d_Cprev,
  // because the pointers get swapped!

  thrust::device_vector<ByteT>& get_Cprev(void) { return d_Cprev; }
};
