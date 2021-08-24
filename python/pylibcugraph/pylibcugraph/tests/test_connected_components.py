# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pytest


@pytest.fixture
def plc():
    import pylibcugraph
    return pylibcugraph


###############################################################################
def test_import():
    """
    Ensure pylibcugraph is importable.
    """
    import pylibcugraph


def test_scc(plc):
    """
    FIXME: rewrite once SCC is implemented.
    """
    with pytest.raises(NotImplementedError):
        plc.strongly_connected_components(None, None, None, None, None, None)


def test_wcc(plc):
    """
    FIXME: rewrite once WCC is implemented.
    """
    with pytest.raises(NotImplementedError):
        plc.weakly_connected_components(None, None, None, None, None, None)
