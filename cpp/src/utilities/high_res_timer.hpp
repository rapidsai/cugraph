/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <ctime>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

//#define TIMING

class HighResTimer {
 public:
  HighResTimer() : timers() {}
  ~HighResTimer() {}

  void start(std::string label)
  {
    auto it = timers.find(label);
    if (it == timers.end()) {
      it = timers.insert(std::make_pair(label, std::make_pair<int, int64_t>(int{0}, int64_t{0})))
             .first;
    }

    timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    it->second.second -= start_time.tv_sec * 1000000000 + start_time.tv_nsec;

    open_label = label;
  }

  void stop()
  {
    timespec stop_time;
    clock_gettime(CLOCK_REALTIME, &stop_time);

    auto it = timers.find(open_label);
    it->second.first++;
    it->second.second += stop_time.tv_sec * 1000000000 + stop_time.tv_nsec;
  }

  double get_average_runtime(std::string const& label)
  {
    auto it = timers.find(label);
    if (it != timers.end()) {
      return (static_cast<double>(it->second.second) / (1000000.0 * it->second.first));
    } else {
      std::stringstream ss;
      ss << "ERROR: timing label: " << label << "not found.";

      throw std::runtime_error(ss.str());
    }
  }

  //
  //  Add display functions... specific label or entire structure
  //
  void display(std::ostream& os)
  {
    os << "Timer Results (in ms):" << std::endl;
    for (auto it = timers.begin(); it != timers.end(); ++it) {
      os << "   " << it->first << " called " << it->second.first
         << " times, average time: " << (it->second.second / (1000000.0 * it->second.first))
         << std::endl;
    }
  }

  void display(std::ostream& os, std::string label)
  {
    auto it = timers.find(label);
    os << it->first << " called " << it->second.first
       << " times, average time: " << (it->second.second / (1000000.0 * it->second.first))
       << std::endl;
  }

  void display_and_clear(std::ostream& os)
  {
    os << "Timer Results (in ms):" << std::endl;
    for (auto it = timers.begin(); it != timers.end(); ++it) {
      os << "   " << it->first << " called " << it->second.first
         << " times, average time: " << (it->second.second / (1000000.0 * it->second.first))
         << std::endl;
    }

    timers.clear();
  }

 private:
  std::map<std::string, std::pair<int, int64_t>> timers;
  std::string open_label;  // should probably be a stack...
};
