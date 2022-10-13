# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import cugraph
import numpy as np


# TRIM Process:
#   - removed single vertex componenets
#   - select vertex with highest out degree
#   - forwards BFS
#   - backward BFS
#   - compute intersection = components
#   - remove component
#   - repeat


def EXPERIMENTAL__strong_connected_component(source, destination):
    """
    Generate the strongly connected components
    using the FW-BW-TRIM approach, but skipping the trimming)

    Parameters
    ----------
    source : cudf.Series
        A cudf series that contains the source side of an edge list

    destination : cudf.Series
        A cudf series that contains the destination side of an edge list

    Returns
    -------
    cdf : cudf.DataFrame - a dataframe for components
        df['vertex']   - the vertex ID
        df['id']       - the component ID

    sdf : cudf.DataFrame - a dataframe with single vertex components
        df['vertex']   - the vertex ID

    count - int - the number of components found


    Examples
    --------
    >>> # M = read_mtx_file(graph_file)
    >>> # sources = cudf.Series(M.row)
    >>> # destinations = cudf.Series(M.col)
    >>> # components, single_components, count =
    >>> #   cugraph.strong_connected_component(source, destination)

    """
    # FIXME: Uncomment out the above example
    max_value = np.iinfo(np.int32).max  # NOQA

    # create the FW and BW graphs - this version dopes nopt modify the graphs
    G_fw = cugraph.Graph()
    G_bw = cugraph.Graph()

    G_fw.add_edge_list(source, destination)
    G_bw.add_edge_list(destination, source)

    # get a list of vertices and sort the list on out_degree
    d = G_fw.degrees()
    d = d.sort_values(by="out_degree", ascending=False)

    num_verts = len(d)

    # create space for the answers
    components = [None] * num_verts
    single_components = [None] * num_verts

    # Counts - aka array indexies
    count = 0
    single_count = 0

    # remove vertices that cannot be in a component
    bad = d.query("in_degree == 0 or out_degree == 0")

    if len(bad):
        bad = bad.drop(["in_degree", "out_degree"])

        single_components[single_count] = bad
        single_count = single_count + 1
        d = _filter_list(d, bad)

    # ----- Start processing -----
    while len(d) > 0:

        v = d["vertex"][0]

        # compute the forward BFS
        bfs_fw = cugraph.bfs(G_fw, v)
        bfs_fw = bfs_fw.query("distance != @max_value")

        # Now backwards
        bfs_bw = cugraph.bfs(G_bw, v)
        bfs_bw = bfs_bw.query("distance != @max_value")

        # intersection
        common = bfs_fw.merge(bfs_bw, on="vertex", how="inner")

        if len(common) > 1:
            common["id"] = v
            components[count] = common
            d = _filter_list(d, common)
            count = count + 1

        else:
            # v is an isolated vertex
            vdf = cudf.DataFrame()
            vdf["vertex"] = v

            single_components[single_count] = vdf
            single_count = single_count + 1
            d = d.iloc[1:]

    # end of loop until vertex queue is empty

    comp = _compress_array(components, count)
    sing = _compress_array(single_components, single_count)

    return comp, sing, count


#  ---------


def _filter_list(vert_list, drop_list):
    t = cudf.DataFrame()
    t["vertex"] = drop_list["vertex"]
    t["d"] = 0

    df = vert_list.merge(t, on="vertex", how="left")

    df["d"] = df["d"].fillna(1)
    df = df.query("d == 1")
    df.drop("d", inplace=True)

    return df


def _compress_array(a, length):

    tmp = cudf.DataFrame()

    if length > 0:
        tmp_a = [None] * length

        for i in range(length):
            tmp_a[i] = a[i]

        tmp = cudf.concat(tmp_a)

    return tmp
