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


import cudf
from collections import OrderedDict


def bfs_df_pregel(_df, start, src_col='src', dst_col='dst', copy_data=True):
    """
    This function executes an unwieghted Breadth-First-Search (BFS) traversal
    to find the distances and predecessors from a specified starting vertex

    NOTE: Only reachable vertices are returned
    NOTE: data is not sorted

    Parameters
    ----------
    _df : cudf.dataframe
        a dataframe containing the source and destination edge list

    start : same type as 'src' and 'dst'
        The index of the graph vertex from which the traversal begins

    src : string
        the source column name

    dst : string
        the destination column name

    copy_data : Bool
        whether we can manipulate the dataframe or if a copy should be made


    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex
        df['distance'][i] gives the path distance for the i'th vertex
            from the starting vertex
        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal

    Examples
    --------
    >>> data_df =
          cudf.read_csv('datasets/karate.csv', delimiter=' ', header=None)
    >>> df = cugraph.pregel_bfs(data_df, 1, '0', '1')

    """

    # extract the src and dst into a dataframe that can be modified
    if copy_data:
        coo_data = _df[[src_col, dst_col]]
    else:
        coo_data = _df

    coo_data.rename(columns={src_col: 'src', dst_col: 'dst'}, inplace=True)

    # convert the "start" vertex into a series
    frontier = cudf.Series(start).to_frame('dst')

    # create the answer DF
    answer = cudf.DataFrame()
    answer['vertex'] = start
    answer['distance'] = 0
    answer['predecessor'] = -1

    # init some variables
    distance = 0
    done = False

    while not done:

        # ---------------------------------
        # update the distance and add it to the dataframe
        distance = distance + 1
        frontier['distance'] = distance

        # -----------------------------------
        # Removed all instances of the frontier vertices from 'dst' side
        # we do not want to hop to a vertex that has already been seen
        coo_data = coo_data.merge(frontier, on=['dst'], how='left')
        coo_data = coo_data[coo_data.distance.isnull()]
        coo_data.drop_column('distance')

        # now update column names for finding source vertices
        frontier.rename(columns={'dst': 'src'}, inplace=True)

        # ---------------------------------
        # merge the list of vertices and distances with the COO list
        # there are two sets of results that we get from the "hop_df" merge
        # (A) the set of edges that start with a vertice in the frontier set
        #     - this goes into the answer set
        #     - this also forms the next frontier set
        # (B) the set of edges that did not start with a frontier vertex
        #     - this form the new set of coo_data
        hop_df = coo_data.merge(frontier, on=['src'], how='left')

        # ---------------------------------
        # (A) get the data where the 'src' was in the frontier list
        # create a new dataframe of vertices to hop out from (the 'dst')
        one_hop = hop_df.query("distance == @distance")
        frontier = one_hop['dst'].to_frame('dst')

        # ---------------------------------
        # (B) get all the edges that where not touched
        coo_data = hop_df[hop_df.distance.isnull()]
        coo_data.drop_column('distance')

        # ---------------------------------
        # update the answer
        one_hop.rename(
            columns={'dst': 'vertex', 'src': 'predecessor'}, inplace=True)

        # remote duplicates. smallest vertex wins
        aggsOut = OrderedDict()
        aggsOut['predecessor'] = 'min'
        aggsOut['distance'] = 'min'
        _a = one_hop.groupby(['vertex'], as_index=False).agg(aggsOut)

        answer = cudf.concat([answer, _a])

        if len(coo_data) == 0:
            done = True

        if not done and len(frontier) == 0:
            done = True

    # all done, return the answer
    return answer
