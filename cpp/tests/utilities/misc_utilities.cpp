/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utilities/misc_utilities.hpp"

#include <cstddef>
#include <string>

namespace cugraph {
namespace test {

std::string getFileName(const std::string& s)
{
  char sep = '/';
#ifdef _WIN32
  sep = '\\';
#endif
  size_t i = s.rfind(sep, s.length());
  if (i != std::string::npos) { return (s.substr(i + 1, s.length() - i)); }
  return ("");
}

}  // namespace test
}  // namespace cugraph
