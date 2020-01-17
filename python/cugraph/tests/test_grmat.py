# Copyright (c) 2019, NVIDIA CORPORATION.
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

import gc
import pytest
# from itertools import product # flake8 required

import cugraph
import rmm


# Test all combinations of default/managed and pooled/non-pooled allocation
# TODO: when GRMAT is back uncomment the 2 lines below:
# @pytest.mark.parametrize('managed, pool',
#                         list(product([False, True], [False, True])))
# ...and (TODO): remove this line below:
@pytest.mark.skip(reason="GRMAT undergoing changes in Gunrock")
def test_grmat_gen(managed, pool):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    vertices, edges, sources, destinations = cugraph.grmat_gen(
        'grmat --rmat_scale=2 --rmat_edgefactor=2 --device=0 --normalized'
        ' --quiet')
