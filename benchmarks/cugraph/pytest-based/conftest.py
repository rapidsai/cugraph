# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

def pytest_addoption(parser):
    parser.addoption("--allow-rmm-reinit",
                     action="store_true",
                     default=False,
                     help="Allow RMM to be reinitialized, possibly multiple times within "
                     "the same process, in order to run benchmarks with different managed "
                     "memory and pool allocator options. This is not the default behavior "
                     "since it does not represent a typical use case, and support for "
                     "this may be limited. Instead, consider multiple pytest runs that "
                     "use a fixed set of RMM settings.")


def pytest_sessionstart(session):
    # if the --allow-rmm-reinit option is not given, set (or add to) the CLI
    # "mark expression" (-m) the markers for no managedmem and
    # poolallocator. This will result in the RMM reinit() function to be called
    # only once in the running process (the typical use case).
    if session.config.getoption("allow_rmm_reinit") is False:
        currentMarkexpr = session.config.getoption("markexpr")

        if ("managedmem" in currentMarkexpr) or \
           ("poolallocator" in currentMarkexpr):
            raise RuntimeError("managedmem and poolallocator markers cannot "
                               "be used without --allow-rmm-reinit.")

        newMarkexpr = "managedmem_off and poolallocator_on"
        if currentMarkexpr:
            newMarkexpr = f"({currentMarkexpr}) and ({newMarkexpr})"

        session.config.option.markexpr = newMarkexpr
