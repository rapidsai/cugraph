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

# this file is pure python and no need to be a cython file. Once cugraph's
# issue #146 is addressed, this file's extension should be changed from .pyx to
# .py and should be located outside the python/cugraph/bindings directory.

from cugraph.structure.graph import DiGraph, Graph


def from_cudf_edgelist(df, source='source', destination='destination',
                       edge_attr=None, create_using=Graph, renumber=True):
    """
    Return a new graph created from the edge list representaion. This function
    is added for NetworkX compatibility (this function is a RAPIDS version of
    NetworkX's from_pandas_edge_list()).  This function does not support multiple
    source or destination columns.  But does support renumbering

    Parameters
    ----------
    df : cudf.DataFrame
        This cudf.DataFrame contains columns storing edge source vertices,
        destination (or target following NetworkX's terminology) vertices, and
        (optional) weights.
    source : string or integer
        This is used to index the source column.
    destination : string or integer
        This is used to index the destination (or target following NetworkX's
        terminology) column.
    edge_attr : string or integer, optional
        This pointer can be ``None``. If not, this is used to index the weight
        column.
    create_using : cuGraph.Graph
        Specify the type of Graph to create.  Default is cugraph.Graph
    renumber : bool
        If source and destination indices are not in range 0 to V where V
        is number of vertices, renumber argument should be True.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G = cugraph.from_cudf_edgelist(M, source='0', target='1', weight='2')

    """
    if create_using is Graph:
        G = Graph()
    elif create_using is DiGraph:
        G = DiGraph()
    else:
        raise Exception("create_using supports Graph and DiGraph")

    G.from_cudf_edgelist(df, source=source, destination=destination,
                         edge_attr=edge_attr, renumber=renumber)

    return G
