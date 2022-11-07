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

from cugraph.structure.graph_implementation import (
    simpleDistributedGraphImpl,
    simpleGraphImpl,
)


def call_cugraph_algorithm(name, graph, *args, **kwargs):
    # TODO check using graph property in a future PR
    if isinstance(graph._Impl, simpleDistributedGraphImpl):
        import cugraph.dask

        return getattr(cugraph.dask, name)(graph, *args, **kwargs)

    # TODO check using graph property in a future PR
    elif isinstance(graph._Impl, simpleGraphImpl):
        import cugraph

        return getattr(cugraph, name)(graph, *args, **kwargs)

    # TODO Properly dispatch for cugraph-service.
