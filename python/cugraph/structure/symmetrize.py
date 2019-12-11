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

from cugraph.structure import graph as csg
import cudf


def symmetrize_df(df, src_name, dst_name):
    """
    Take a COO stored in a DataFrame, along with the column names of
    the source and destination columns and create a new data frame
    using the same column names that symmetrize the graph so that all
    edges appear in both directions.

    Note that if other columns exist in the data frame (e.g. edge weights)
    the other columns will also be replicated.  That is, if (u,v,data)
    represents the source value (u), destination value (v) and some
    set of other columns (data) in the input data, then the output
    data will contain both (u,v,data) and (v,u,data) with matching
    data.

    If (u,v,data1) and (v,u,data2) exist in the input data where data1
    != data2 then this code will arbitrarily pick the smaller data
    element to keep, if this is not desired then the caller should
    should correct the data prior to calling symmetrize.

    Parameters
    ----------
    df : cudf.DataFrame
        Input data frame containing COO.  Columns should contain source
        ids, destination ids and any properties associated with the
        edges.
    src_name : string
        Name of the column in the data frame containing the source ids
    dst_name : string
        Name of the column in the data frame containing the destination ids

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sym_df = cugraph.symmetrize(M, '0', '1')
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sym_df['0]', sym_df['1'], sym_df['2'])
    """
    gdf = cudf.DataFrame()

    #
    #  NOTE: if there are values then we can't use drop_duplicates - in
    #        case the values are different in different directions.  To
    #        address this, we will use groupby if there are values.  If
    #        there are no values then groupby won't eliminate the duplicate
    #        keys.  Believe this is a bug, see
    #        https://github.com/rapidsai/cudf/issues/2730.  Once this
    #        is resolved we should be able to just use groupby.
    #
    #  We will use drop_duplicates if there are no non-key fields
    #
    use_groupby = False

    #
    #  Now append the columns.  We add sources to the end of destinations,
    #  and destinations to the end of sources.  Otherwise we append a
    #  column onto itself.
    #
    for idx, name in enumerate(df.columns):
        if (name == src_name):
            gdf[src_name] = df[src_name].append(df[dst_name],
                                                ignore_index=True)
        elif (name == dst_name):
            gdf[dst_name] = df[dst_name].append(df[src_name],
                                                ignore_index=True)
        else:
            gdf[name] = df[name].append(df[name], ignore_index=True)
            use_groupby = True

    if use_groupby:
        return gdf.groupby(by=[src_name, dst_name], as_index=False).min()
    else:
        return gdf.drop_duplicates(subset=[src_name, dst_name], keep='first')


def symmetrize(source_col, dest_col, value_col=None):
    """
    Take a COO set of source destination pairs along with associated values and
    create a new COO set of source destination pairs along with values where
    all edges exist in both directions.

    Return from this call will be a COO stored as two cudf Series - the
    symmetrized source column and the symmetrized dest column, along with
    an optional cudf Series containing the associated values (only if the
    values are passed in).

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
    value_col : cudf.Series (optional)
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains values associated with this edge.
        For this function the values can be any type, they are not
        examined, just copied.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> values = cudf.Series(M['2'])
    >>> src, dst, val = cugraph.symmetrize(sources, destinations, values)
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(src, dst, val)
    """
    csg.null_check(source_col)
    csg.null_check(dest_col)

    input_df = cudf.DataFrame({'source': source_col,
                               'destination': dest_col})

    if value_col is not None:
        csg.null_check(value_col)
        input_df.add_column('value', value_col)

    output_df = symmetrize_df(input_df, 'source', 'destination')

    if value_col is not None:
        return (output_df['source'],
                output_df['destination'],
                output_df['value'])

    return output_df['source'], output_df['destination']
