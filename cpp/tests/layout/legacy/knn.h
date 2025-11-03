/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
