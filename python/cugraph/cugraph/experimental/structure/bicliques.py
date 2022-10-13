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

# Import needed libraries
import cudf
import numpy as np
from collections import OrderedDict


def EXPERIMENTAL__find_bicliques(
    df, k, offset=0, max_iter=-1, support=1.0, min_features=1, min_machines=10
):
    """
    Find the top k maximal bicliques

    Parameters
    ----------
    df :  cudf:DataFrame
        A dataframe containing the bipartite graph edge list
        Columns must be called 'src', 'dst', and 'flag'

    k  :  int
        The max number of bicliques to return
        -1 mean all

    offset : int

    Returns
    -------
    B : cudf.DataFrame
        A dataframe containing the list of machine and features.  This is not
        the full edge list to save space.  Since it is a biclique, it is ease
        to recreate the edges

        B['id']    - a cluster ID (this is a one up number - up to k)
        B['vert']  - the vertex ID
        B['type']  - 0 == machine, 1 == feature


    S : cudf.DataFrame
        A dataframe of statistics on the returned info.
        This dataframe is (relatively small) of size k.

        S['id']       - the cluster ID
        S['total']    - total vertex count
        S['machines'] - number of machine nodes
        S['features'] - number of feature vertices
        S['bad_ration'] - the ratio of bad machine / total machines
    """
    # must be factor of 10
    PART_SIZE = int(1000)

    x = [col for col in df.columns]
    if "src" not in x:
        raise NameError("src column not found")
    if "dst" not in x:
        raise NameError("dst column not found")
    if "flag" not in x:
        raise NameError("flag column not found")

    if support > 1.0 or support < 0.1:
        raise NameError("support must be between 0.1 and 1.0")

    # this removes a prep step that offset the values for CUDA process
    if offset > 0:
        df["dst"] = df["dst"] - offset

    # break the data into chunks to improve join/search performance
    src_by_dst, num_parts = _partition_data_by_feature(df, PART_SIZE)

    # Get a list of all the dst (features) sorted by degree
    f_list = _count_features(df, True)

    # create a dataframe for the answers
    bicliques = cudf.DataFrame()
    stats = cudf.DataFrame()

    # create a dataframe to help prevent duplication of work
    machine_old = cudf.DataFrame()

    # create a dataframe for stats
    stats = cudf.DataFrame()

    answer_id = 0
    iter_max = len(f_list)

    if max_iter != -1:
        iter_max = max_iter

    # Loop over all the features (dst) or until K is reached
    for i in range(iter_max):

        # pop the next feature to process
        feature = f_list["dst"][i]
        degree = f_list["count"][i]

        # compute the index to this item (which dataframe chunk is in)
        idx = int(feature / PART_SIZE)

        # get all machines that have this feature
        machines = get_src_from_dst(src_by_dst[idx], feature)

        # if this set of machines is the same as the last, skip this feature
        if not is_same_as_last(machine_old, machines):

            # now from those machines, hop out to the list of all the features
            feature_list = get_all_feature(src_by_dst, machines, num_parts)

            # summarize occurances
            ic = _count_features(feature_list, True)

            goal = int(degree * support)  # NOQA

            # only get dst nodes with the same degree
            c = ic.query("count >= @goal")

            # need more than X feature to make a biclique
            if len(c) > min_features:
                if len(machines) >= min_machines:
                    bicliques, stats = update_results(
                        machines, c, answer_id, bicliques, stats
                    )

                    answer_id = answer_id + 1

        # end - if same

        machine_old = machines

        if k > -1:
            if answer_id == k:
                break

    # end for loop

    # All done, reset data
    if offset > 0:
        df["dst"] = df["dst"] + offset

    return bicliques, stats


def _partition_data_by_feature(_df, PART_SIZE):

    # compute the number of sets
    m = int((_df["dst"].max() / PART_SIZE) + 1)

    _ui = [None] * (m + 1)

    # Partition the data into a number of smaller DataFrame
    s = 0
    e = s + PART_SIZE

    for i in range(m):
        _ui[i] = _df.query("dst >= @s and dst < @e")

        s = e
        e = e + PART_SIZE

    return _ui, m


def _count_features(_gdf, sort=True):

    aggs = OrderedDict()
    aggs["dst"] = "count"

    c = _gdf.groupby(["dst"], as_index=False).agg(aggs)

    c = c.rename(columns={"count_dst": "count"}, copy=False)

    if sort:
        c = c.sort_values(by="count", ascending=False)

    return c


# get all src vertices for a given dst
def get_src_from_dst(_gdf, id):

    _src_list = _gdf.query("dst == @id")

    _src_list.drop("dst", inplace=True)

    return _src_list


def is_same_as_last(_old, _new):
    status = False

    if len(_old) == len(_new):
        m = _old.merge(_new, on="src", how="left")

        if m["src"].null_count == 0:
            status = True

    return status


# get all the items used by the specified users
def get_all_feature(_gdf, src_list_df, N):

    c = [None] * N

    for i in range(N):
        c[i] = src_list_df.merge(_gdf[i], on="src", how="inner")

    return cudf.concat(c)


def update_results(m, f, key, b, s):
    """
    Input
    * m = machines
    * f = features
    * key = cluster ID
    * b = biclique answer
    * s = stats answer

    Returns
    -------
    B : cudf.DataFrame
        A dataframe containing the list of machine and features.  This is not
    the full edge list to save space. Since it is a biclique, it is ease
    to recreate the edges

        B['id']    - a cluster ID (this is a one up number - up to k)
        B['vert']  - the vertex ID
        B['type']  - 0 == machine, 1 == feature


    S : cudf.DataFrame
        A Pandas dataframe of statistics on the returned info.
        This dataframe is (relatively small) of size k.

        S['id']       - the cluster ID
        S['total']    - total vertex count
        S['machines'] - number of machine nodes
        S['features'] - number of feature vertices
        S['bad_ratio'] - the ratio of bad machine / total machines
    """
    B = cudf.DataFrame()
    S = cudf.DataFrame()

    m_df = cudf.DataFrame()
    m_df["vert"] = m["src"]
    m_df["id"] = int(key)
    m_df["type"] = int(0)

    f_df = cudf.DataFrame()
    f_df["vert"] = f["dst"].astype(np.int32)
    f_df["id"] = int(key)
    f_df["type"] = int(1)

    if len(b) == 0:
        B = cudf.concat([m_df, f_df])
    else:
        B = cudf.concat([b, m_df, f_df])

    # now update the stats
    num_m = len(m_df)
    num_f = len(f_df)
    total = num_m + num_f

    num_bad = len(m.query("flag == 1"))
    ratio = num_bad / total

    # now stats
    s_tmp = cudf.DataFrame()
    s_tmp["id"] = key
    s_tmp["total"] = total
    s_tmp["machines"] = num_m
    s_tmp["features"] = num_f
    s_tmp["bad_ratio"] = ratio

    if len(s) == 0:
        S = s_tmp
    else:
        S = cudf.concat([s, s_tmp])

    del m_df
    del f_df

    return B, S
