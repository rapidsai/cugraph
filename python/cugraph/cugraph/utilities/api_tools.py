# Copyright (c) 2022, NVIDIA CORPORATION.
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

import pylibcugraph.utilities.api_tools as api_tools

experimental_prefix = "EXPERIMENTAL"


def experimental_warning_wrapper(obj):
    """
    Wrap obj in a function or class that prints a warning about it being
    "experimental" (ie. it is in the public API but subject to change or
    removal), prior to calling obj and returning its value.

    The object's name used in the warning message also has any leading __
    and/or EXPERIMENTAL string are removed from the name used in warning
    messages. This allows an object to be named with a "private" name in the
    public API so it can remain hidden while it is still experimental, but
    have a public name within the experimental namespace so it can be easily
    discovered and used.
    """
    api_tools.experimental_warning_wrapper(obj)


def promoted_experimental_warning_wrapper(obj):
    api_tools.promoted_experimental_warning_wrapper(obj)
