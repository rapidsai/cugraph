# Copyright (c) 2024, NVIDIA CORPORATION.
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

# This file is simply used to make librmm a real package rather than a namespace
# package to work around https://github.com/scikit-build/scikit-build-core/issues/682.
# Since we have it, we may as well also set up some helpful metadata.
from libcugraph._version import __git_commit__, __version__
from libcugraph.load import load_library
