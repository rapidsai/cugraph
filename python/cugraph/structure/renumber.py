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

import numpy as np
import cudf
from collections import OrderedDict
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


def renumber_from_cudf(_df, source_cols_names, dest_cols_names):
    """
    Take a set, collection (lists) of source and destination columns, and
    renumber the vertices to create a dense set of contiguously vertex ids
    from 0 to the number of unique vertices - 1.

    Input columns can be any data type.

    The output will be mapped to int32, since many of the cugraph functions
    are limited to int32. If the number of unique values is > 2^31-1 then
    this function will return an error.

    NOTICE
    ---------
    - The number of source and destination columns must be the same
    - The source and destination column names cannot be the same or overlap.
    - The data type order needs to be the same between source and destination
        columns. This is due to the two sets being merged to create a single
        list of all possible values    

    Input Parameters
    ----------
    df : cudf.DataFrame
        The dataframe containing the source and destination columans
    source_cols_names : List
        This is a list of source column names
    dest_cols_names : List
        This is a list of destination column names

    Returns
    ---------
    src_ids : cudf.Series
        The new source vertex IDs
    dst_ids : cudf.Series
        The new destination vertex IDs
    numbering_df : cudf.DataFrame (
        a dataframe that maps a vertex ID to the unique


    Examples
    --------
    >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)

    >>> source_col, dest_col, numbering_map =
    >>>    cugraph.renumber_from_cudf(gdf, "0", "1")
    >>>
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(source_col, dest_col, None)
    """
    if len(source_cols_names) == 0:
        raise ValueError('Source column list is empty')

    if len(dest_cols_names) == 0:
        raise ValueError('Destination column list is empty')

    if len(source_cols_names) != len(dest_cols_names):
        raise ValueError(
            'Source and Destination column lists are not the same size')

    # ---------------------------------------------------
    # get the source column names and map to indexes
    _src_map = OrderedDict()
    for i in range(len(source_cols_names)):
        _src_map.update({source_cols_names[i]: str(i)})

    _tmp_df_src = _df[source_cols_names].rename(_src_map)

    # --------------------------------------------------------
    # get the destination column names and map to indexes
    _dst_map = OrderedDict()
    for i in range(len(dest_cols_names)):
        _dst_map.update({dest_cols_names[i]: str(i)})

    _tmp_df_dst = _df[dest_cols_names].rename(_dst_map)

    # ------------------------------------
    _s = _tmp_df_src.drop_duplicates()
    _d = _tmp_df_dst.drop_duplicates()

    _tmp_df = cudf.concat([_s, _d])
    _tmp_df = _tmp_df.drop_duplicates().reset_index().drop('index')

    if len(_tmp_df) > np.info(np.int32).max():
        raise ValueError('dataset is larger than int32')

    _tmp_df['id'] = _tmp_df.index.astype(np.int32)

    del _s
    del _d

    _src_ids = _tmp_df_src.merge(_tmp_df, on=_src_map.values())
    _dst_ids = _tmp_df_src.merge(_tmp_df, on=_dst_map.values())

    return _src_ids['id'], _dst_ids['id'], _tmp_df
