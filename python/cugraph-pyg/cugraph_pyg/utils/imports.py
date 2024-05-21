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

from packaging.requirements import Requirement
from importlib import import_module


def package_available(requirement: str) -> bool:
    """Check if a package is installed and meets the version requirement."""
    req = Requirement(requirement)
    try:
        pkg = import_module(req.name)
    except ImportError:
        return False

    if len(req.specifier) > 0:
        if hasattr(pkg, "__version__"):
            return pkg.__version__ in req.specifier
        else:
            return False

    return True
