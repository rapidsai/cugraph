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
// A wrapper of clock_gettime.
// Michael A. Frumkin (mfrumkin@nvidia.com)
#pragma once

#include <iostream>
#include <string>
#include <time.h>

class HighResClock {
 public:
  HighResClock() {
    clock_gettime(CLOCK_REALTIME, &_start_time);
    clock_gettime(CLOCK_REALTIME, &_stop_time);
  }
  ~HighResClock() { }

  void start() { clock_gettime(CLOCK_REALTIME, &_start_time); }

  std::string stop() {
    clock_gettime(CLOCK_REALTIME, &_stop_time);
    char buffer[64];
    long long int start_time =
        _start_time.tv_sec * 1e9 + _start_time.tv_nsec;
    long long int stop_time =
        _stop_time.tv_sec * 1e9 + _stop_time.tv_nsec;

    sprintf(buffer, "%lld us", 
            (stop_time - start_time) / 1000);
    std::string str(buffer);
    return str;
  }

  void stop(double* elapsed_time) {  // returns time in us
    clock_gettime(CLOCK_REALTIME, &_stop_time);
    long long int start_time =
        _start_time.tv_sec * 1e9 + _start_time.tv_nsec;
    long long int stop_time =
        _stop_time.tv_sec * 1e9 + _stop_time.tv_nsec;
    *elapsed_time = (stop_time - start_time) / 1000;
  }

 private: 
  timespec _start_time;
  timespec _stop_time;   
};
