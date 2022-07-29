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

#include "knn.h"

#include <algorithm>
#include <iostream>
#include <vector>

double euclidian_dist(const std::vector<int>& x, const std::vector<int>& y)
{
  double total = 0;
  auto i       = x.begin();
  auto j       = y.begin();
  for (; i != x.end() && j != y.end(); ++i, ++j)
    total += pow(*i, 2) - 2 * *i * *j + pow(*j, 2);
  return sqrt(total);
}

std::vector<std::vector<double>> pairwise_distances(const std::vector<std::vector<int>>& X)
{
  std::vector<std::vector<double>> distance_matrix(X.size(), std::vector<double>(X[0].size()));
  for (size_t i = 0; i < X.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      const float val       = euclidian_dist(X[i], X[j]);
      distance_matrix[i][j] = val;
      distance_matrix[j][i] = val;
    }
  }
  return distance_matrix;
}

template <typename Iter, typename Compare>
std::vector<int> argsort(Iter begin, Iter end, Compare comp)
{
  std::vector<std::pair<int, Iter>> pairList;
  std::vector<int> ret;

  int i = 0;
  for (auto it = begin; it < end; it++) {
    std::pair<int, Iter> pair(i, it);
    pairList.push_back(pair);
    i++;
  }

  std::stable_sort(pairList.begin(),
                   pairList.end(),
                   [comp](std::pair<int, Iter> prev, std::pair<int, Iter> next) -> bool {
                     return comp(*prev.second, *next.second);
                   });

  for (auto i : pairList)
    ret.push_back(i.first);

  return ret;
}

void fill_diag(std::vector<std::vector<double>>& X)
{
  for (size_t i = 0; i < X.size(); ++i) {
    for (size_t j = 0; j < X[i].size(); ++j) {
      if (i == j) X[i][j] = INFINITY;
    }
  }
}

std::vector<std::vector<int>> get_knn_indices(const std::vector<std::vector<double>>& X,
                                              const int k)
{
  std::list<point> X_list;
  for (size_t i = 0; i < X.size(); ++i) {
    point p(X[i]);
    X_list.push_back(p);
  }

  std::vector<std::vector<int>> ind_X_embedded;
  for (auto i = X_list.begin(); i != X_list.end(); ++i) {
    auto temp = knn_classify(X_list, *i, k);
    ind_X_embedded.push_back(temp);
  }
  return ind_X_embedded;
}

double compute_rank(const std::vector<std::vector<int>>& ind_X,
                    std::vector<std::vector<int>>& ind_X_embedded,
                    const int k)
{
  const auto n = ind_X.size();

  auto rank = 0;
  for (size_t i = 0; i < n; ++i) {
    std::vector<int> ranks(k, 0);
    for (auto j = 0; j < k; ++j) {
      auto it = std::find(ind_X[i].begin(), ind_X[i].end(), ind_X_embedded[i][j]);
      if (it != ind_X[i].end()) {
        auto idx = std::distance(ind_X[i].begin(), it);
        ranks[j] = idx;
      }
    }
    for (auto& val : ranks)
      val -= k;

    for (const auto& val : ranks)
      if (val > 0) rank += val;
  }
  return rank;
}

template <typename T>
void print_matrix(const std::vector<std::vector<T>>& matrix)
{
  for (size_t i = 0; i < matrix.size(); ++i) {
    std::cout << "[ ";
    for (size_t j = 0; j < matrix[i].size(); ++j) {
      std::cout << matrix[i][j] << ' ';
    }
    std::cout << "]\n";
  }
}

double trustworthiness_score(const std::vector<std::vector<int>>& X,
                             const std::vector<std::vector<double>>& Y,
                             int n,
                             int d,
                             int k)
{
  auto dist_X = pairwise_distances(X);
  fill_diag(dist_X);

  std::vector<std::vector<int>> ind_X;
  for (size_t i = 0; i < dist_X.size(); ++i) {
    auto tmp = argsort(dist_X[i].begin(), dist_X[i].end(), std::less<double>());
    ind_X.push_back(tmp);
  }

  auto ind_X_embedded = get_knn_indices(Y, k);

  double t = compute_rank(ind_X, ind_X_embedded, k);
  t        = 1.0 - t * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)));
  return t;
}
