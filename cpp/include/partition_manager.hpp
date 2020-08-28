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

#pragma once

#include <raft/comms/comms.hpp>
#include <raft/handle.hpp>

#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cugraph {
namespace partition_2d {
template <typename from_t>
std::string to_string(from_t const& value)
{
  std::stringstream ss;
  ss << value;
  return ss.str();
}

// default key-naming mechanism:
//
struct key_naming_t {
  key_naming_t(int row_indx,
               int col_indx,
               std::string const& col_suffix = std::string("_col"),
               std::string const& row_suffix = std::string("_row"),
               std::string const& prefix     = std::string("partition"))
    : col_suffix_(col_suffix),
      row_suffix_(row_suffix),
      prefix_(prefix),
      name_(prefix_ + "_" + to_string(row_indx) + "_" + to_string(col_indx))
  {
  }

  std::string col_name(void) const { return name_ + col_suffix_; }

  std::string row_name(void) const { return name_ + row_suffix_; }

 private:
  std::string const col_suffix_;
  std::string const row_suffix_;
  std::string const prefix_;
  std::string name_;
};

template <typename value_t>
using matrix_t = std::vector<std::vector<value_t>>;

using pair_comms_t =
  std::pair<std::shared_ptr<raft::comms::comms_t>, std::shared_ptr<raft::comms::comms_t>>;

enum class colors_2d_t : int { ROW = 0, COL = 1 };

// class responsible for creating 2D partition sub-comms:
//
template <typename key_name_policy_t = key_naming_t, typename size_type = int>
class partition_manager_t {
 public:
  partition_manager_t(raft::handle_t& handle, size_type p_row_size, size_type p_col_size)
    : handle_(handle), p_row_size_(p_row_size), p_col_size_(p_col_size)
  {
    init_comms();
  }

  partition_manager_t(raft::handle_t const& handle, size_type p_size) : handle_(handle)
  {
    partition2d(p_size);
    init_comms();
  }

  virtual ~partition_manager_t(void) {}

  matrix_t<pair_comms_t> const& comms_matrix(void) const { return comms_set_; }

 protected:
  virtual void partition2d(size_type p_size)
  {
    auto sqr = static_cast<size_type>(std::sqrt(p_size));

    // find divisor of p_size
    // nearest to sqr;
    //
    p_row_size_ = nearest_divisor(sqr, p_size);
    p_col_size_ = p_size / p_row_size_;

    assert(p_row_size_ > 1 && p_col_size_ > 1);
  }

  void init_comms(void)
  {
    std::vector<pair_comms_t> empty_row(p_col_size_, std::make_pair(nullptr, nullptr));
    comms_set_.assign(p_row_size_, empty_row);

    raft::comms::comms_t const& communicator = handle_.get_comms();

    for (size_type row = 0; row < p_row_size_; ++row)
      for (size_type col = 0; col < p_col_size_; ++col) {
        key_name_policy_t key{row, col};

        // comm_slpit() on same key=linear_key,
        // but different colors for row and column
        //
        // TODO: check if this assummed
        // functionality is correct
        //
        size_type linear_key{p_col_size_ * row + col};

        auto shared_row_comm = std::make_shared<raft::comms::comms_t>(
          communicator.comm_split(static_cast<int>(colors_2d_t::ROW), linear_key));
        handle_.set_subcomm(key.row_name(), shared_row_comm);

        auto shared_col_comm = std::make_shared<raft::comms::comms_t>(
          communicator.comm_split(static_cast<int>(colors_2d_t::COL), linear_key));
        handle_.set_subcomm(key.col_name(), shared_col_comm);

        // Also store in a matrix of comms_t;
        // this may be redundant, but useful;
        // TODO: check if this is okay...
        //
        comms_set_[row][col] = std::make_pair(shared_row_comm, shared_col_comm);
      }
  }

 private:
  raft::handle_t& handle_;
  size_type p_row_size_;
  size_type p_col_size_;
  matrix_t<pair_comms_t> comms_set_;

  static decltype(auto) nearest_divisor(size_type sqr, size_type p_size)
  {
    assert(sqr > 0);

    for (size_type div = sqr; div > 0; --div) {
      auto p_div = p_size % div;
      if (p_div == 0) return div;
    }
  }
};
}  // namespace partition_2d
}  // namespace cugraph
