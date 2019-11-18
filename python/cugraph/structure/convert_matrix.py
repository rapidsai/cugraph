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

# this file is pure python and no need to be a cython file. Once cugraph's
# issue #146 is addressed, this file's extension should be changed from .pyx to
# .py and should be located outside the python/cugraph/bindings directory.

from cugraph.structure.graph import DiGraph


def from_cudf_edgelist(df, source='source', target='target', weight=None):
    """
    Return a new graph created from the edge list representaion. This function
    is added for NetworkX compatibility (this function is a RAPIDS version of
    NetworkX's from_pandas_edge_list()).

    Parameters
    ----------
    df : cudf.DataFrame
        This cudf.DataFrame contains columns storing edge source vertices,
        destination (or target following NetworkX's terminology) vertices, and
        (optional) weights.
    source : string or integer
        This is used to index the source column.
    target : string or integer
        This is used to index the destination (or target following NetworkX's
        terminology) column.
    weight : string or integer, optional
        This pointer can be ``None``. If not, this is used to index the weight
        column.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G = cugraph.from_cudf_edgelist(M, source='0', target='1', weight='2')
    """

    G = DiGraph()

    if weight is None:
        G.from_cudf_edgelist(df, source=source, target=target)
    else:
        G.from_cudf_edgelist(df, source=source, target=target,
                             edge_attr=weight)

    return G
