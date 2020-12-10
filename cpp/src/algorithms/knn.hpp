/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

# pragma once

inline float dist_xy( int i, int j, float **x, float **y){
   float xi = (*x)[i];
   float yi = (*y)[i];
   float xj = (*x)[j];
   float yj = (*y)[j];
   return sqrtf((xi-xj)*(xi-xj) +(yi-yj)*(yi-yj));
}

inline bool inQuadRelative( int quad, int orig, int p, float **x, float **y) {
/* Filter a point p depending on whether it is in 2D quadrant quad relative to orig
   Quad 1: p.x >= orig.x and p.y >= orig.y
   Quad 2: p.x <= orig.x and p.y >= orig.y
   Quad 3: p.x <= orig.x and p.y <= orig.y
   Quad 4: p.x >= orig.x and p.y <= orig.y
   Yes, points on the boundaries count as being in *both* quadrants
   Returns true is p is in quad relative to orig
*/
   bool non_neg_x = ( (*x)[p] >= (*x)[orig]);
   bool non_neg_y = ( (*y)[p] >= (*y)[orig]);
   bool non_pos_x = ( (*x)[p] <= (*x)[orig]);
   bool non_pos_y = ( (*y)[p] <= (*y)[orig]);
   if (quad == 1 && (non_neg_x && non_neg_y)) return true;
   if (quad == 2 && (non_pos_x && non_neg_y)) return true;
   if (quad == 3 && (non_pos_x && non_pos_y)) return true;
   if (quad == 4 && (non_neg_x && non_pos_y)) return true;
   return false;
}

inline void topK( int i, int j, int k, int quad, float dist, float *nearest, int **neighbors) {
// insertion sort into the topk list of nearest indices and Knn list
// will be called for each vertex, for each neighbor bin in the Knn search
// Fast return if new value can't beat the kth-best value
  if (quad ==0) {
    if( dist > nearest[k-1]) return;
    for (int ik = 0; ik < k; ++ik) {
       int idx = k*i +ik;
       //printf("slot %d, comp %d @%f to %d @%f\n",ik,j,dist,(*neighbors)[idx],nearest[ik]);
       if( (*neighbors)[idx] == j && dist == nearest[ik] ) break;
       if( dist < nearest[ik] && nearest[ik] == 1e6) {
         //printf("%d %d dist = %f goes in slot %d\n",i,j,dist,ik);
         (*neighbors)[idx] = j;
         nearest[ik] = dist;
         break;
       }
       if( dist < nearest[ik] && nearest[ik] < 1e6 ) {
          //printf("found improvement over %d to %f, now %f\n",ik, nearest[ik],dist);
          for (int jk = k-1; jk > ik; --jk) {
              if (dist < nearest[jk-1] && nearest[jk-1]<1e6) {
                 //printf(" moving %d to %d ", jk-1, jk);
                 nearest[jk] = nearest[jk-1];
                 (*neighbors)[k*i +jk] = (*neighbors)[k*i+jk-1];
              }
          }
          //printf("%d %d dist = %f goes in slot %d\n",i,j,dist,ik);
          (*neighbors)[k*i + ik] = j;
          nearest[ik] = dist;
          break;
       }
    }
 }
 else {
    if( dist > nearest[quad]) return;
    if( dist < nearest[quad]) {
      nearest[quad] = dist;
      (*neighbors)[k*i + quad] = j;
    }
 }
}


/*  Nearest neighbor 2D search
    Break up the domain into boxes, then for each point locate it's box and search all
    points in the local box and box neighbors (9 total boxes need to be searched)
    Uses topK routine to maintain list of k-nearest for each point
    Now also tries to sample each quadrant to find possibly far but important neighbors.
*/
void findKneighbors( int numPackages, int k,
                     float **x, float **y,
                     int **neighbors, bool useQuadrants){

  int avg_pkgs_per_bin = (k +4 -1)/4; //target to scale bin size so we get same points/bin

  int num_bins = (numPackages+avg_pkgs_per_bin-1)/avg_pkgs_per_bin; //divs*divs;
  int divs = ceil(sqrt(num_bins)); // square grid, row major order idx = divs*rowid + colid
  num_bins = divs*divs;

  int *assignments = (int *)malloc( numPackages*sizeof(int));
  int *bins        = (int *)malloc( numPackages*sizeof(int));
  int *bincounts   = (int *)malloc(  num_bins * sizeof(int));
  int *binoffsets  = (int *)malloc( (num_bins +1)* sizeof(int));
  float *nearest   = (float *)malloc( k * sizeof(float));
  int bin_width = (1024+divs -1)/divs;

  printf(" in findKneighbors k=%d, using %d wide bins, %d bins %d divs, useQuad= %d\n",
           k, bin_width, num_bins, divs, useQuadrants);
  for (int i= 0; i< (num_bins); ++i)  bincounts[i] = 0;
  for (int i= 0; i< numPackages; ++i) {
     int ix = floor((*x)[i] / bin_width);
     int jy = floor((*y)[i] / bin_width);
     //printf(" package %d x=%f y=%f, ix=%d jy=%d \n",i, (*x)[i], (*y)[i], ix,jy);
     int idx = divs*ix + jy;
     bincounts[idx] += 1;
     assignments[i] = idx;
     for (int j =0; j< k; j++) (*neighbors)[k*i + j] = numPackages;
     //printf(" package %d in bin %d\n", i, idx);
     //fflush(stdout);
  }
  binoffsets[0] = 0;
  binoffsets[num_bins] = numPackages;
  for (int i= 0; i < num_bins; ++i){
     binoffsets[i+1] = bincounts[i]+binoffsets[i];
     //printf("idx %d count=%d binoffset=%d  \n",i,bincounts[i], binoffsets[i]);
  }
  for (int i= 0; i< numPackages; ++i) {
     //printf("assigning package %d, in bin %d\n",i, assignments[i]);
     int mybin = assignments[i];
     bincounts[mybin] -= 1;
     int loc = binoffsets[mybin] + bincounts[mybin];
     //printf("  package %d goes in location %d\n",i, loc);
     bins[loc] = i;
     //printf(" package %d in bin %d at loc=%d, %d left\n",i, assignments[i], loc, bincounts[assignments[i]]);
     //fflush(stdout);
  }

#define binloop( idx ) candidates += ( binoffsets[idx+1] - binoffsets[idx]) ; \
                       for (int p=0; p< (binoffsets[idx+1]-binoffsets[idx]); p++) { \
                            int j = bins[binoffsets[idx] +p]; \
                            /*printf(" p = %d, j = %d\n", p, j);*/ \
                            if (i==j) continue;   \
                            if (useQuadrants && !inQuadRelative( quad, i, j, x, y) ) continue; \
                            topK( i, j, k, quad, dist_xy( i, j, x, y), nearest, neighbors); \
                            }  idx = assignments[i];

  for (int i= 0; i< numPackages; ++i) {
     int ix = floor((*x)[i] / bin_width);
     int jy = floor((*y)[i] / bin_width);
     //int idx = divs*ix + jy;
     int idx = assignments[i];
     float xi= (*x)[i];
     float yi= (*y)[i];
     for (int j=0; j<k; j++) {
        nearest[j] = 1e6;
     }
     //printf(" package %d x=%f y=%f ix=%d jy=%d \n",i, xi, yi, ix,jy);
     int candidates =0;
     int quad_max = ( useQuadrants ? 4 : 1);

     for (int quad = 0; quad < quad_max; quad++) {
     // local bin case
     //printf(" local bin = %d\n",idx);
     binloop( idx )

     if( ix >0 && ( quad ==0 || quad == 2 || quad == 3) ) {
         idx -= divs;
         //printf(" looking in left bin = %d\n",idx);
         binloop( idx )
     }
     if( ix < divs-1 && ( quad ==0 || quad == 1 || quad == 4) ) {
         idx += divs;
         //printf(" looking in right bin = %d\n",idx);
         binloop( idx )
     }
     if( jy >0 && ( quad ==0 || quad == 3 || quad == 4) ) {
         idx -= 1;
         //printf(" looking in down bin = %d\n",idx);
         binloop( idx )
     }
     if( jy < divs-1 && ( quad ==0 || quad == 1 || quad == 2) ) {
         idx += 1;
         //printf(" looking in up bin = %d\n",idx);
         binloop( idx )
     }
     if( (ix >0) && (jy>0) && (quad ==0 || quad == 3)) {
         idx += -divs -1;
         //printf(" looking in left-down bin = %d\n",idx);
         binloop( idx )
     }
     if( (ix >0) && (jy<7) && (quad ==0 || quad == 2)) {
         idx += -divs +1;
         //printf(" looking in left-up bin = %d\n",idx);
         binloop( idx )
     }
     if( (ix <divs-1) && (jy>0) && (quad ==0 || quad ==4)) {
         idx += +divs -1;
         //printf(" looking in right-down bin = %d\n",idx);
         binloop( idx )
     }
     if( (ix <divs-1) && (jy<divs-1) &&( quad ==0 || quad ==1)) {
         idx += +divs +1;
         //printf(" looking in right-up bin = %d\n",idx);
         binloop( idx )
     }
     //printf("%d examined %d candidates\n",i,candidates);
     //fflush(stdout);
     }

  }

  free(assignments);
  free(bins);
  free(binoffsets);
  free(bincounts);
  free(nearest);
}
