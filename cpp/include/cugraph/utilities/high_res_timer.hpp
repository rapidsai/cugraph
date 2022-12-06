/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/utilities/error.hpp>

#include <chrono>
#include <iostream>
#include <map>
#include <ostream>
#include <stack>
#include <string>
#include <tuple>

class HighResTimer {
 public:
  void start(std::string label)
  {
    start_stack.push(std::make_tuple(label, std::chrono::steady_clock::now()));
  }

  double stop()
  {
    auto stop_time           = std::chrono::steady_clock::now();
    auto [label, start_time] = start_stack.top();
    start_stack.pop();

    auto it = labeled_timers.find(label);
    if (it == labeled_timers.end()) {
      labeled_timers[label] = std::make_tuple(size_t{0}, double{0});
      it                    = labeled_timers.find(label);
    }

    std::chrono::duration<double> diff = stop_time - start_time;

    auto& timer = it->second;
    std::get<0>(timer) += 1;
    std::get<1>(timer) += diff.count();

    return diff.count();
  }

  void display(std::ostream& os)
  {
    for (auto it = labeled_timers.begin(); it != labeled_timers.end(); ++it) {
      auto [count, duration] = it->second;
      os << it->first << " called " << count
         << " times, average time: " << duration / static_cast<double>(count) << " s." << std::endl;
    }
  }

  void display(std::ostream& os, std::string label)
  {
    auto it = labeled_timers.find(label);
    CUGRAPH_EXPECTS(it != labeled_timers.end(), "Invalid input argument: invalid label.");
    auto [count, duration] = it->second;
    os << it->first << " called " << count
       << " times, average time: " << duration / static_cast<double>(count) << " s." << std::endl;
  }

  void display_and_clear(std::ostream& os)
  {
    display(os);
    labeled_timers.clear();
  }

 private:
  std::map<std::string, std::tuple<size_t, double> /* # calls, aggregate duration in seconds */>
    labeled_timers{};
  std::stack<std::tuple<std::string, std::chrono::time_point<std::chrono::steady_clock>>>
    start_stack{};
};
