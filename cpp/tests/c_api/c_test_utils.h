/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <time.h>


/*
 * Runs the function pointed to by "test" and returns the return code.  Also
 * prints reporting info (using "test_name"): pass/fail and run time, to stdout.
 *
 * Intended to be used by the RUN_TEST macro.
 */
int run_test(int (*test)(), const char* test_name) {
   int ret_val = 0;
   time_t start_time, end_time;

   printf("RUNNING: %s...", test_name);
   fflush(stdout);

   time(&start_time);
   ret_val = test();
   time(&end_time);

   printf("done (%f seconds).", difftime(end_time, start_time));
   if(ret_val == 0) {
      printf(" - passed\n");
   } else {
      printf(" - FAILED\n");
   }
   fflush(stdout);

   return ret_val;
}

#define RUN_TEST(test_name) run_test(test_name, #test_name)
