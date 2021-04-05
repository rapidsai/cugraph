# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import gc
import pytest
import cugraph
from cugraph.tests import utils
from cugraph.utilities import df_score_to_dictionary

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


print("Networkx version : {} ".format(nx.__version__))


def calc_nx_core_number(graph_file):
    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, source="0", target="1", create_using=nx.Graph()
    )
    nc = nx.core_number(Gnx)
    return nc


def calc_cg_core_number(graph_file):
    M = utils.read_csv_file(graph_file)
    G = cugraph.Graph()
    G.from_cudf_edgelist(M, source="0", destination="1")

    cn = cugraph.core_number(G)
    return cn


def calc_core_number(graph_file):
    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, source="0", target="1", create_using=nx.Graph()
    )
    nc = nx.core_number(Gnx)

    M = utils.read_csv_file(graph_file)
    G = cugraph.Graph()
    G.from_cudf_edgelist(M, source="0", destination="1")

    cn = cugraph.core_number(G)
    cn = cn.sort_values("vertex").reset_index(drop=True)

    pdf = [nc[k] for k in sorted(nc.keys())]
    cn["nx_core_number"] = pdf
    cn = cn.rename(columns={"core_number": "cu_core_number"}, copy=False)
    return cn


# FIXME: the default set of datasets includes an asymmetric directed graph
# (email-EU-core.csv), which currently causes an error with NetworkX:
# "networkx.exception.NetworkXError: Input graph has self loops which is not
#  permitted; Consider using G.remove_edges_from(nx.selfloop_edges(G))"
#
# https://github.com/rapidsai/cugraph/issues/1045
#
# @pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_core_number(graph_file):
    gc.collect()

    nx_num = calc_nx_core_number(graph_file)
    cg_num = calc_cg_core_number(graph_file)

    # convert cugraph dataframe to a dictionary
    cg_num_dic = df_score_to_dictionary(cg_num, k="core_number")

    assert cg_num_dic == nx_num


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_core_number_nx(graph_file):
    gc.collect()

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, source="0", target="1", create_using=nx.Graph()
    )
    nc = nx.core_number(Gnx)
    cc = cugraph.core_number(Gnx)

    assert nc == cc
