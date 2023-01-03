# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
    parser.addoption("--no-rmm-reinit", action="store_true", default=False,
                     help="Do not reinit RMM to run benchmarks with different"
                          " managed memory and pool allocator options.")


def pytest_sessionstart(session):
    # if the --no-rmm-reinit option is given, set (or add to) the CLI "mark
    # expression" (-m) the markers for no managedmem and no poolallocator. This
    # will cause the RMM reinit() function to not be called.
    if session.config.getoption("no_rmm_reinit"):
        newMarkexpr = "managedmem_off and poolallocator_off"
        currentMarkexpr = session.config.getoption("markexpr")

        if ("managedmem" in currentMarkexpr) or \
           ("poolallocator" in currentMarkexpr):
            raise RuntimeError("managedmem and poolallocator markers cannot "
                               "be used with --no-rmm-reinit")

        if currentMarkexpr:
            newMarkexpr = f"({currentMarkexpr}) and ({newMarkexpr})"

        session.config.option.markexpr = newMarkexpr
