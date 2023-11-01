# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib.resources

# Read VERSION file from the module that is symlinked to VERSION file
# in the root of the repo at build time or copied to the moudle at
# installation. VERSION is a separate file that allows CI build-time scripts
# to update version info (including commit hashes) without modifying
# source files.
__version__ = (
    importlib.resources.files("nx_cugraph").joinpath("VERSION").read_text().strip()
)
__git_commit__ = ""
