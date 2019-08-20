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

from cugraph.structure.graph import null_check
import cudf


def symmetrize(source_col, dest_col):
    """
    Take a COO set of src/dest pairs assumed to represent a directed graph
    and create a new COO set of src/dest pairs where all edges exist in
    both directions.

    Return from this call will be a COO stored as two cudf Series - the
    symmetrized source column and the symmetrized dest column.

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

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> source_col, dest_col, cugraph.symmetrize(sources, destinations)
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(source_col, dest_col, None)
    """
    null_check(source_col)
    null_check(dest_col)

    append_gdf = cudf.DataFrame()
    append_gdf['src'] = source_col.append(dest_col, ignore_index=True)
    append_gdf['dst'] = dest_col.append(source_col, ignore_index=True)

    final_gdf = append_gdf.drop_duplicates()
    return final_gdf['src'], final_gdf['dst']
