/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#ifndef __DIA_H__
#define __DIA_H__

#include <algorithm>


/*
 * Compute Y += A*X for DIA matrix A and dense vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_row            - number of rows in A
 *   I  n_col            - number of columns in A
 *   I  n_diags          - number of diagonals
 *   I  L                - length of each diagonal
 *   I  offsets[n_diags] - diagonal offsets 
 *   T  diags[n_diags,L] - nonzeros 
 *   T  Xx[n_col]        - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row]        - output vector 
 *
 * Note:
 *   Output array Yx must be preallocated
 *   Negative offsets correspond to lower diagonals
 *   Positive offsets correspond to upper diagonals
 *
 */
template <class I, class T>
void dia_matvec(const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
	            const I offsets[], 
	            const T diags[], 
	            const T Xx[],
	                  T Yx[])
{
    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max(0,-k);
        const I j_start = std::max(0, k);
        const I j_end   = std::min(std::min(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T * diag = diags + i*L + j_start;
        const T * x = Xx + j_start;
              T * y = Yx + i_start;

        for(I n = 0; n < N; n++){
            y[n] += diag[n] * x[n]; 
        }
    }
}


#endif
