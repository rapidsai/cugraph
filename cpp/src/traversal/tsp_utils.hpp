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

/* Affine map routine

   Transforms a vector of numbers by the affine map y = A*x + b
   Operates in-place and overwrite x
   Inverse operations happen in reverse order
*/
void affineTrans( int n, bool forward, float *x, float A, float b) {
   if (forward)
   for (int i = 0; i< n; i++){
      x[i] += b;
      x[i] *= A;
   }
   else
   for (int i = 0; i< n; i++){
      x[i] *= A;
      x[i] += b;
   }
}
