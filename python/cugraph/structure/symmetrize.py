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


def symmetrize_df(df, src_name, dst_name):
    """
    Take a COO stored in a dataframe, along with the column names of
    the source and destination columns assumed to represent a directed graph
    and create a new data frame using the same column names that
    symmetrize the graph so that all edges appear in both directions.

    Note that if other columns exist in the data frame (e.g. edge weights)
    the other columns will also be replicated.  That is, if (u,v,data)
    represents the source value (u), destination value (v) and some
    set of other columns (data) in the input data, then the output
    data will contain both (u,v,data) and (v,u,data) with matching
    data.

    If (u,v,data1) and (v,u,data2) exist in the input data where data1
    != data2 then this code will arbitrarily pick a data element to keep,
    not necessarily the same element for each direction, so the caller
    should avoid or tolerate this behavior.

    Parameters
    ----------
    df : cdf.DataFrame
        Input data frame containing COO.  Columns should contain source
        ids, destination ids and any properties associated with the
        edges.
    src_name : string
        Name of the column in the data frame containing the source ids
    dst_name : string
        Name of the column in the data frame containing the dest ids

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sym_df = cugraph.symmetrize(M, '0', '1')
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(M['0]', M['1'], M['2'])
    """
    gdf = cudf.DataFrame()

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

    return gdf.drop_duplicates(subset=[src_name, dst_name], keep='first')


def symmetrize(source_col, dest_col, weights=None):
    """
    Take a COO set of src/dest pairs assumed to represent a directed graph
    along with associated weights and create a new COO set of src/dest
    pairs along with weights where all edges exist in both directions.

    Return from this call will be a COO stored as two cudf Series - the
    symmetrized source column and the symmetrized dest column, along with
    a new cudf Series containing the associated weights.

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
    weights : cudf.Series (optional)
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains weights associated with this edge.
        For this function the weights can be any type, they are not
        examined, just copied.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> weights = cudf.Series(M['2'])
    >>> src, dst, wt = cugraph.symmetrize(sources, destinations, weights)
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(src, dst, wt)
    """
    null_check(source_col)
    null_check(dest_col)

    input_df = cudf.DataFrame([('src', source_col), ('dst', dest_col)])

    if weights is not None:
        null_check(weights)
        input_df.add_column('weight', weights)

    output_df = symmetrize_df(input_df, 'src', 'dst')

    if weights is not None:
        return output_df['src'], output_df['dst'], output_df['weight']

    return output_df['src'], output_df['dst']
