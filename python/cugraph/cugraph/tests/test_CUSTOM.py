# Copyright (c) 2020-2025, NVIDIA CORPORATION.:
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

# import gc
# import random

import pytest

# import networkx as nx

# import cudf
# import cugraph
# from cudf.testing import assert_series_equal
# from cugraph.utilities import ensure_cugraph_obj_for_nx
from cugraph.testing import SMALL_DATASETS, DEFAULT_DATASETS


# =============================================================================
# Parameters
# =============================================================================

DATASETS = [pytest.param(d) for d in DEFAULT_DATASETS]
SMALL_DATASETS = [pytest.param(d) for d in SMALL_DATASETS]

# TESTS


@pytest.mark.sg
@pytest.mark.boop
@pytest.mark.parametrize("dataset", DATASETS)
def test_download_path(dataset):
    from pprint import pprint

    print(f"Got dataset: {dataset.metadata['name']}")

    pprint(dataset.metadata)

    print(f"This path is set to: {dataset.get_path()}")

    assert 1 == 2
