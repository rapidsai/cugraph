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
  // simplified key (one per all row subcomms / one per all column sub-comms):
  //
  key_naming_t(std::string const& row_suffix = std::string("_p_row"),
               std::string const& col_suffix = std::string("_p_col"),
               std::string const& prefix     = std::string("comm"))
    : row_suffix_(row_suffix), col_suffix_(col_suffix), prefix_(prefix), name_(prefix_)
  {
  }

  // more involved key naming, using row/col indices:
  //
  key_naming_t(int row_indx,
               int col_indx,
               std::string const& row_suffix = std::string("_p_row"),
               std::string const& col_suffix = std::string("_p_col"),
               std::string const& prefix     = std::string("comm"))
    : row_suffix_(row_suffix),
      col_suffix_(col_suffix),
      prefix_(prefix),
      name_(prefix_ + "_" + to_string(row_indx) + "_" + to_string(col_indx))
  {
  }

  std::string col_name(void) const { return name_ + col_suffix_; }

  std::string row_name(void) const { return name_ + row_suffix_; }

 private:
  std::string const row_suffix_;
  std::string const col_suffix_;
  std::string const prefix_;
  std::string name_;
};

using pair_comms_t =
  std::pair<std::shared_ptr<raft::comms::comms_t>, std::shared_ptr<raft::comms::comms_t>>;

// class responsible for creating 2D partition sub-comms:
// this is instantiated by each worker (processing element, PE)
// for the row/column it belongs to;
//
// naming policy defaults to simplified naming:
// one key per row subcomms, one per column subcomms;
//
template <typename name_policy_t = key_naming_t, typename size_type = int>
class subcomm_factory_t {
 public:
  subcomm_factory_t(raft::handle_t& handle, size_type row_size)
    : handle_(handle), row_size_(row_size)
  {
    init_row_col_comms();
  }
  virtual ~subcomm_factory_t(void) {}

  pair_comms_t const& row_col_comms(void) const { return row_col_subcomms_; }

 protected:
  virtual void init_row_col_comms(void)
  {
    name_policy_t key;
    raft::comms::comms_t const& communicator = handle_.get_comms();

    int const rank = communicator.get_rank();
    int row_index  = rank / row_size_;
    int col_index  = rank % row_size_;

    auto row_comm =
      std::make_shared<raft::comms::comms_t>(communicator.comm_split(row_index, col_index));
    handle_.set_subcomm(key.row_name(), row_comm);

    auto col_comm =
      std::make_shared<raft::comms::comms_t>(communicator.comm_split(col_index, row_index));
    handle_.set_subcomm(key.col_name(), col_comm);

    row_col_subcomms_.first  = row_comm;
    row_col_subcomms_.second = col_comm;
  }

 private:
  raft::handle_t& handle_;
  size_type row_size_;
  pair_comms_t row_col_subcomms_;
};
}  // namespace partition_2d
}  // namespace cugraph
