# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cudf


def get_traversed_path(df, id):
    """
    Take the DataFrame result from a BFS or SSSP function call and extract
    the path to a specified vertex.

    Input Parameters
    ----------
    df : cudf.DataFrame
        The dataframe containing the results of a BFS or SSSP call
    id : Int
        The vertex ID

    Returns
    ---------
    src_ids : cudf.Series
        The new source vertex IDs
    dst_ids : cudf.Series
        The new destination vertex IDs
    numbering_df : cudf.DataFrame
        a dataframe that maps a vertex ID to the unique values


    Examples
    --------
    >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>>
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> sssp_df = cugraph.sssp(G, 1)
    >>> path = cugraph.utils.get_traversed_path(sssp_df, 32)
    """

    if 'vertex' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'vertex' column missing")
    if 'distance' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'distance' column missing")
    if 'predecessor' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'predecessor' column missing")
    if type(id) != int:
        raise ValueError("The vertex 'id' needs to be an integer")

    # There is no guarantee that the dataframe has not been filtered
    # or edited.  Therefore we cannot assume that using the vertex ID
    # as an index will work

    ddf = df.loc[df['vertex'] == id]
    if len(ddf) == 0:
        raise ValueError("The vertex (", id, " is not in the result set")

    pred = ddf['predecessor'][0]

    answer = []
    answer.append(ddf)

    while pred != -1:
        ddf = df.loc[df['vertex'] == pred]
        pred = ddf['predecessor'][0]
        answer.append(ddf)

    return cudf.concat(answer)
