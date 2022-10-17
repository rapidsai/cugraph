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

#include <cmath>
#include <list>
#include <map>
#include <set>
#include <vector>

struct point {
  point() {}
  point(std::vector<double> vec) { attributes.assign(vec.begin(), vec.end()); }
  point(std::list<double>& data) : attributes(data) {}
  std::list<double> attributes;
  double distance;
  int index;
};

struct point_compare {
  bool operator()(const point& x, const point& y) const
  {
    if (x.distance != y.distance) return x.distance < y.distance;
    return x.attributes.front() < y.attributes.front();
  }
};

double sq_euclid_dist(const point& x, const point& y)
{
  double total = 0;
  auto i       = x.attributes.begin();
  auto j       = y.attributes.begin();
  for (; i != x.attributes.end() && j != y.attributes.end(); ++i, ++j)
    total += pow(*i - *j, 2);
  return total;
}

std::vector<int> knn_classify(std::list<point>& dataframe, const point& c, const int k)
{
  std::set<point, point_compare> distances;
  auto i    = dataframe.begin();
  int index = 0;
  for (; i != dataframe.end(); ++i) {
    i->distance = sq_euclid_dist(c, *i);
    i->index    = index++;
    distances.insert(*i);
  }

  std::vector<int> res;
  auto count = 0;
  auto j     = distances.begin();
  ++j;
  for (; j != distances.end() && count < k; ++j, ++count)
    res.push_back(j->index);
  return res;
}
