/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

namespace cugraph {
namespace test {

std::string getFileName(const std::string& s);

// Define RAPIDS_DATASET_ROOT_DIR using a preprocessor variable to
// allow for a build to override the default. This is useful for
// having different builds for specific default dataset locations.
#ifndef RAPIDS_DATASET_ROOT_DIR
#define RAPIDS_DATASET_ROOT_DIR "/datasets"
#endif

static const std::string& get_rapids_dataset_root_dir()
{
  static std::string rdrd("");
  // Env var always overrides the value of RAPIDS_DATASET_ROOT_DIR
  if (rdrd == "") {
    const char* envVar = std::getenv("RAPIDS_DATASET_ROOT_DIR");
    rdrd               = (envVar != NULL) ? envVar : RAPIDS_DATASET_ROOT_DIR;
  }
  return rdrd;
}

}  // namespace test
}  // namespace cugraph
