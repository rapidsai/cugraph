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


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// Turn on to see stats for each level
//#define ENABLE_LOG true
#include "nvlouvain.cuh"



/* Louvain Clustering Sample

Social network example: Zachary Karate Club 
W. Zachary, “An information flow model for conflict and fission in small groups,” Journal of Anthropological Research, vol. 33, pp. 452–473, 1977
https://en.wikipedia.org/wiki/Zachary's_karate_club
--------------------------------------------------------------------
V = 34
E = 78 bidirectional, 156 directed edges

Bidirectional edges list:
[2 1] [3 1] [3 2] [4 1] [4 2] [4 3] [5 1] [6 1] [7 1] [7 5] [7 6] [8 1] [8 2] [8 3] [8 4] [9 1] [9 3] [10 3] [11 1] [11 5] [11 6] [12 1] [13 1] [13 4] [14 1] [14 2] [14 3] [14 4] [17 6] [17 7] 
[18 1] [18 2] [20 1] [20 2] [22 1] [22 2] [26 24] [26 25] [28 3] [28 24] [28 25] [29 3] [30 24] [30 27] [31 2] [31 9] [32 1] [32 25] [32 26] [32 29] [33 3] [33 9] [33 15] [33 16] 
[33 19] [33 21] [33 23] [33 24] [33 30] [33 31] [33 32] [34 9] [34 10] [34 14] [34 15] [34 16] [34 19] [34 20] [34 21] [34 23] [34 24] [34 27] [34 28] [34 29] [34 30] [34 31] 
[34 32] [34 33]

CSR representation (directed):
csrRowPtrA_h {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156}
csrColIndA_h {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 6, 10, 16, 0, 
4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 
24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 
26, 27, 28, 29, 30, 31, 32}
csrValA_h {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

--------------------------------------------------------------------

Operation: Louvain Clustering  default parameters in modularity maximization

--------------------------------------------------------------------

Expected output: 
This sample prints the modlarity score and compare against the python reference (https://python-louvain.readthedocs.io/en/latest/api.html)


*/

using namespace nvlouvain;

void check_status(nvlouvainStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %s\n",nvlouvainStatusGetString(status));
        exit(0);
    }
}

int main(int argc, char **argv)
{
    // Hard-coded Zachary Karate Club network input
    int csrRowPtrA_input [] = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
        139, 156};
    int csrColIndA_input [] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
        6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
        25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
        18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
    float csrValA_input [] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int ref_clustering [] = {0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 2};
    int *csrRowPtrA_h = &csrRowPtrA_input[0];
    int *csrColIndA_h = &csrColIndA_input[0];
    float *csrValA_h = &csrValA_input[0];
    
    // Variables
    const size_t  n = 34, nnz = 156;
    bool weighted = false;
    bool has_init_cluster = false;    
    int *clustering_h, *init_cluster_ptr = nullptr;;
    int num_levels = 0, hits =0;
    float final_modulartiy = 0; 
    // Allocate host data for nvgraphSpectralClustering output
    clustering_h = (int*)malloc(n*sizeof(int));
     
    //Solve clustering with modularity maximization algorithm
    check_status(louvain<int,float>(csrRowPtrA_h, csrColIndA_h, csrValA_h, n, nnz, weighted, has_init_cluster, init_cluster_ptr, final_modulartiy, clustering_h, num_levels));

    //Print quality (modualrity)
    printf("Modularity_score: %f\n", final_modulartiy);
    printf("num levels: %d\n", num_levels);
    for (int i = 0; i < (int)n; i++)
        if (clustering_h[i] == ref_clustering[i])
            hits++;
    printf("Hit rate : %f%% (%d hits)\n", (hits*100.0)/n, hits);
    // Print the clustering vector in csv format
    //for (int i = 0; i < (int)(n-1); i++)
    //    printf("%d,",clustering_h[i]);
    //printf("%d,\n",clustering_h[n-1]);
    free(clustering_h);
    printf("Done!\n");

    return EXIT_SUCCESS;
}

