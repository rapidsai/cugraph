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

from cugraph.structure import graph_wrapper
from cugraph.structure import graph as csg


def renumber(source_col, dest_col):
    """
    Take a (potentially sparse) set of source and destination vertex ids and
    renumber the vertices to create a dense set of vertex ids using all values
    contiguously from 0 to the number of unique vertices - 1.

    Input columns can be either int64 or int32.  The output will be mapped to
    int32, since many of the cugraph functions are limited to int32. If the
    number of unique values in source_col and dest_col > 2^31-1 then this
    function will return an error.

    Return from this call will be three cudf Series - the renumbered
    source_col, the renumbered dest_col and a numbering map that maps the new
    ids to the original ids.

    Parameters
    ----------
    source_col : cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the source index for each edge.
        Source indices must be an integer type.
    dest_col : cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the destination index for each edge.
        Destination indices must be an integer type.
    numbering_map : cudf.Series
        This cudf.Series wraps a gdf column of size V (V: number of vertices).
        The gdf column contains a numbering map that mpas the new ids to the
        original ids.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> source_col, dest_col, numbering_map = cugraph.renumber(sources,
    >>>                                                        destinations)
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(source_col, dest_col, None)
    """
    csg.null_check(source_col)
    csg.null_check(dest_col)

    source_col, dest_col, numbering_map = graph_wrapper.renumber(source_col,
                                                                 dest_col)

    return source_col, dest_col, numbering_map
