# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
import pandas as pd
import networkx as nx
import numpy as np
import dask_cudf
import os
from cugraph.dask.common.mg_utils import (get_client)

#
# Datasets
#
DATASETS_UNDIRECTED = ['../datasets/karate.csv',  '../datasets/dolphins.csv']
DATASETS_UNRENUMBERED = ['../datasets/karate-disjoint.csv']

DATASETS = ['../datasets/karate-disjoint.csv',
            '../datasets/dolphins.csv',
            '../datasets/netscience.csv']
#            '../datasets/email-Eu-core.csv']

STRONGDATASETS = ['../datasets/dolphins.csv',
                  '../datasets/netscience.csv',
                  '../datasets/email-Eu-core.csv']

DATASETS_KTRUSS = [('../datasets/polbooks.csv',
                    '../datasets/ref/ktruss/polbooks.csv'),
                   ('../datasets/netscience.csv',
                    '../datasets/ref/ktruss/netscience.csv')]

DATASETS_SMALL = ['../datasets/karate.csv',
                  '../datasets/dolphins.csv',
                  '../datasets/polbooks.csv']


def read_csv_for_nx(csv_file, read_weights_in_sp=True):
    print('Reading ' + str(csv_file) + '...')
    if read_weights_in_sp is True:
        df = pd.read_csv(csv_file, delimiter=' ', header=None,
                         names=['0', '1', 'weight'],
                         dtype={'0': 'int32', '1': 'int32',
                                'weight': 'float32'})
    else:
        df = pd.read_csv(csv_file, delimiter=' ', header=None,
                         names=['0', '1', 'weight'],
                         dtype={'0': 'int32', '1': 'int32',
                                'weight': 'float64'})

    # nverts = 1 + max(df['0'].max(), df['1'].max())

    # return coo_matrix((df['2'], (df['0'], df['1'])), shape=(nverts, nverts))
    return df


def read_csv_file(csv_file, read_weights_in_sp=True):
    print('Reading ' + str(csv_file) + '...')
    if read_weights_in_sp is True:
        return cudf.read_csv(csv_file, delimiter=' ',
                             dtype=['int32', 'int32', 'float32'], header=None)
    else:
        return cudf.read_csv(csv_file, delimiter=' ',
                             dtype=['int32', 'int32', 'float64'], header=None)


def read_dask_cudf_csv_file(csv_file, read_weights_in_sp=True,
                            single_partition=True):
    print('Reading ' + str(csv_file) + '...')
    if read_weights_in_sp is True:
        if single_partition:
            chunksize = os.path.getsize(csv_file)
            return dask_cudf.read_csv(csv_file, chunksize=chunksize,
                                      delimiter=' ',
                                      names=['src', 'dst', 'weight'],
                                      dtype=['int32', 'int32', 'float32'],
                                      header=None)
        else:
            return dask_cudf.read_csv(csv_file, delimiter=' ',
                                      names=['src', 'dst', 'weight'],
                                      dtype=['int32', 'int32', 'float32'],
                                      header=None)
    else:
        if single_partition:
            chunksize = os.path.getsize(csv_file)
            return dask_cudf.read_csv(csv_file, chunksize=chunksize,
                                      delimiter=' ',
                                      names=['src', 'dst', 'weight'],
                                      dtype=['int32', 'int32', 'float32'],
                                      header=None)
        else:
            return dask_cudf.read_csv(csv_file, delimiter=' ',
                                      names=['src', 'dst', 'weight'],
                                      dtype=['int32', 'int32', 'float64'],
                                      header=None)


def generate_nx_graph_from_file(graph_file, directed=True):
    M = read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(M, create_using=(nx.DiGraph() if directed
                                                   else nx.Graph()),
                                  source='0', target='1')
    return Gnx


def generate_cugraph_graph_from_file(graph_file, directed=True):
    cu_M = read_csv_file(graph_file)
    G = cugraph.DiGraph() if directed else cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    return G


def generate_mg_batch_cugraph_graph_from_file(graph_file, directed=True):
    client = get_client()
    _ddf = read_dask_cudf_csv_file(graph_file)
    ddf = client.persist(_ddf)
    G = cugraph.DiGraph() if directed else cugraph.Graph()
    G.from_dask_cudf_edgelist(ddf)
    return G


def build_cu_and_nx_graphs(graph_file, directed=True):
    G = generate_cugraph_graph_from_file(graph_file, directed=directed)
    Gnx = generate_nx_graph_from_file(graph_file, directed=directed)
    return G, Gnx


def build_mg_batch_cu_and_nx_graphs(graph_file, directed=True):
    G = generate_mg_batch_cugraph_graph_from_file(graph_file,
                                                  directed=directed)
    Gnx = generate_nx_graph_from_file(graph_file, directed=directed)
    return G, Gnx


def random_edgelist(e=1024, ef=16,
                    dtypes={"src": np.int32, "dst": np.int32, "val": float},
                    drop_duplicates=True, seed=None):
    """ Create a random edge list

    Parameters
    ----------
    e : int
        Number of edges
    ef : int
        Edge factor (average number of edges per vertex)
    dtypes : dict
        Mapping of column names to types.
        Supported type is {"src": int, "dst": int, "val": float}
    drop_duplicates
        Drop duplicates
    seed : int (optional)
        Randomstate seed

    Examples
    --------
    >>> from cugraph.tests import utils
    >>> # genrates 20 df with 100M edges each and write to disk
    >>> for x in range(20):
    >>>    df = utils.random_edgelist(e=100000000, ef=64,
    >>>                               dtypes={'src':np.int32, 'dst':np.int32},
    >>>                               seed=x)
    >>>    df.to_csv('df'+str(x), header=False, index=False)
    >>>    #df.to_parquet('files_parquet/df'+str(x), index=False)
    """
    state = np.random.RandomState(seed)
    columns = dict((k, make[dt](e // ef, e, state))
                   for k, dt in dtypes.items())
    
    df = pd.DataFrame(columns)
    if drop_duplicates:
        df = df.drop_duplicates()
        print("Generated "+str(df.shape[0])+" edges")
    return cudf.from_pandas(df)


def make_int32(v, e, rstate):
    return rstate.randint(low=0, high=v, size=e, dtype=np.int32)

def make_int64(v, e, rstate):
    return rstate.randint(low=0, high=v, size=e, dtype=np.int64)

def make_float(v, e, rstate):
    return rstate.rand(e) * 2 - 1


make = {
    float: make_float,
    np.int32: make_int32,
    np.int64: make_int64
}
