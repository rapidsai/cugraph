# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.dask.community import louvain_wrapper


def louvain(input_graph, max_iter=100, resolution=1.):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method on multiple GPUs

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> Comms.initialize()
    >>> chunksize = dcg.get_chunksize(input_data_path)
    >>> ddf = dask_cudf.read_csv('datasets/karate.csv', chunksize=chunksize,
                                 delimiter=' ',
                                 names=['src', 'dst', 'value'],
                                 dtype=['int32', 'int32', 'float32'])
    >>> dg = cugraph.Graph()
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
                                   edge_attr='value')
    >>> parts, modularity_score = dcg.louvain(dg)
    """
    # FIXME: import here to prevent circular import: cugraph->louvain
    # wrapper->cugraph/structure->cugraph/dask->dask/louvain->cugraph/structure
    # from cugraph.structure.graph import Graph

    # FIXME: dask methods to populate graphs from edgelists are only present on
    # DiGraph classes. Disable the Graph check for now and assume inputs are
    # symmetric DiGraphs.
    # if type(input_graph) is not Graph:
    #     raise Exception("input graph must be undirected")

    parts, modularity_score = louvain_wrapper.louvain(
        input_graph, max_iter, resolution
    )

    if input_graph.renumbered:
        # MG renumbering is lazy, but it's safe to assume it's been called at
        # this point if renumbered=True
        parts = input_graph.unrenumber(parts, "vertex")

    return parts, modularity_score
