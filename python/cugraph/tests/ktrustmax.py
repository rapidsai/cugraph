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

import gc
from itertools import product

import pytest

import pandas as pd
import cugraph
import cudf
from cugraph.tests import utils
from cugraph.structure import symmetrize 

import rmm
from rmm import rmm_config

import networkx as nx


# DATASETS = ['../datasets/dolphins.csv',
#             '../datasets/netscience.csv']


def networkx_calc_x_truss_max(graph_file):
    NM = utils.read_csv_for_nx(graph_file)
    NM = NM.tocsr()

    k=3;
    Gnx = nx.Graph(NM)
    
    while(not nx.is_empty(Gnx)):
        Gnx = nx.k_truss(Gnx,k)
        k=k+1
    k=k-2;

    print ("NetworkX KMAX:")
    print(k)


    return k
    # nc = nx.core_number(Gnx)
    # pdf = pd.DataFrame(nc, index=[0]).T
    # cn['nx_core_number'] = pdf[0]
    # cn = cn.rename({'core_number': 'cu_core_number'})



def calc_k_truss_max(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    src, dst = cugraph.symmetrize(cu_M['0'], cu_M['1'])

    G = cugraph.Graph()
    # G.add_edge_list(src, dst)
    G.add_edge_list(cu_M['0'], cu_M['1'])

    # print(M['1']-1)

    # M['0']=M['0']-1
    # M['1']=M['1']-1

    k_max = cugraph.ktruss_max(G)


    print ("Python KMAX:")
    print(k_max)
    return k_max



rmm.finalize()
rmm_config.use_managed_memory = False
rmm_config.use_pool_allocator = False
rmm.initialize()

assert(rmm.is_initialized())


calc_k_truss_max("dolphins.csv")
networkx_calc_x_truss_max("dolphins.csv")
calc_k_truss_max("netscience.csv")
networkx_calc_x_truss_max("netscience.csv")



calc_k_truss_max("email-Enron.csv")
networkx_calc_x_truss_max("email-Enron.csv")

calc_k_truss_max("amazon0601.csv")
