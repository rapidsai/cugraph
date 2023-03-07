# Copyright (c) 2022, NVIDIA CORPORATION.
#
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

import pytest

from cugraph_service_server.testing import utils

graph_creation_extension1_file_contents = """
import cudf
from cugraph.experimental import PropertyGraph

def custom_graph_creation_function(server):
   edgelist = cudf.DataFrame(columns=['src', 'dst'],
                             data=[(0, 77), (1, 88), (2, 99)])
   pG = PropertyGraph()
   pG.add_edge_data(edgelist, vertex_col_names=('src', 'dst'))

   # smoke test the server object by accesing the "mg" attr
   server.is_multi_gpu

   return pG
"""

graph_creation_extension2_file_contents = """
import cudf
from cugraph.experimental import PropertyGraph

def __my_private_function():
   pass

def my_graph_creation_function(arg1:str, arg2:str, arg3:str, server):
   edgelist = cudf.DataFrame(columns=[arg1, arg2, arg3],
                             data=[(0, 1, 2), (88, 99, 77)])
   pG = PropertyGraph()
   pG.add_edge_data(edgelist, vertex_col_names=(arg1, arg2))
   return pG
"""

graph_creation_extension_long_running_file_contents = """
import time
import cudf
from cugraph.experimental import PropertyGraph

def long_running_graph_creation_function(server):
   time.sleep(10)
   pG = PropertyGraph()
   return pG
"""

graph_creation_extension_no_facade_arg_file_contents = """
import time
import cudf
from cugraph.experimental import PropertyGraph

def graph_creation_function(arg1, arg2):
   time.sleep(10)
   pG = PropertyGraph()
   return pG
"""

graph_creation_extension_bad_arg_order_file_contents = """
import time
import cudf
from cugraph.experimental import PropertyGraph

def graph_creation_function(server, arg1, arg2):
   pG = PropertyGraph()
   return pG
"""

graph_creation_extension_empty_graph_file_contents = """
import time
import cudf
from cugraph.experimental import PropertyGraph, MGPropertyGraph

def graph_creation_function(server):
   if server.is_multi_gpu:
      pG = MGPropertyGraph()
   else:
      pG = PropertyGraph()
   return pG
"""

graph_creation_extension_big_vertex_ids_file_contents = """
import cudf
import cupy
import dask_cudf
from cugraph.experimental import PropertyGraph, MGPropertyGraph

def graph_creation_function_vert_and_edge_data_big_vertex_ids(server):
   if server.is_multi_gpu:
      pG = MGPropertyGraph()
   else:
      pG = PropertyGraph()
   big_num = (2**32)+1
   df = cudf.DataFrame({"vert_id":cupy.arange(big_num, big_num+10,
                                              dtype="int64"),
                        "vert_prop":cupy.arange(big_num+100, big_num+110,
                                                dtype="int64")})
   if server.is_multi_gpu:
      df = dask_cudf.from_cudf(df, npartitions=2)
   pG.add_vertex_data(df, vertex_col_name="vert_id")

   df = cudf.DataFrame({"src":cupy.arange(big_num, big_num+10, dtype="int64"),
                        "dst":cupy.arange(big_num+1,big_num+11, dtype="int64"),
                        "edge_prop":cupy.arange(big_num+100, big_num+110,
                                                dtype="int64")})
   if server.is_multi_gpu:
      df = dask_cudf.from_cudf(df, npartitions=2)
   pG.add_edge_data(df, vertex_col_names=["src", "dst"])

   return pG
"""

graph_creation_extension_large_property_graph_file_contents = """
import cudf
import cupy
import dask_cudf
from cugraph.experimental import PropertyGraph, MGPropertyGraph

def graph_creation_extension_large_property_graph(server):
   if server.is_multi_gpu:
      pG = MGPropertyGraph()
   else:
      pG = PropertyGraph()

   num_verts = 10e6
   df = cudf.DataFrame({"vert_id":cupy.arange(num_verts, dtype="int32"),
                        "vert_prop":cupy.arange(num_verts, dtype="int32"),
                        })
   if server.is_multi_gpu:
      df = dask_cudf.from_cudf(df, npartitions=2)
   pG.add_vertex_data(df, vertex_col_name="vert_id")

   df = cudf.DataFrame({"src":cupy.arange(num_verts, dtype="int32"),
                        "dst":cupy.arange(1, num_verts+1, dtype="int32"),
                        "edge_prop":cupy.arange(num_verts, dtype="int32"),
                        })
   if server.is_multi_gpu:
      df = dask_cudf.from_cudf(df, npartitions=2)
   pG.add_edge_data(df, vertex_col_names=["src", "dst"])

   return pG
"""

extension1_file_contents = """
import cupy as cp


def my_nines_function(array1_size, array1_dtype, array2_size, array2_dtype):
    '''
    Returns 2 arrays of size and dtype specified containing only 9s
    '''
    array1 = cp.array([9] * array1_size, dtype=array1_dtype)
    array2 = cp.array([9] * array2_size, dtype=array2_dtype)
    return (array1, array2)
"""


extension_with_facade_file_contents = """
import cupy

def my_extension(arg1, arg2, server):

    # This extension assumes the server already has a single PG loaded via
    # calling graph_creation_extension1
    gid = server.get_graph_ids()[0]
    pG = server.get_graph(gid)

    edge_df = pG.get_edge_data()

    # Do an arbitrary operation on the PG based on the args, and return the
    # result as a cupy array.

    retval = cupy.array(edge_df[pG.edge_id_col_name] + arg1 + arg2)
    return retval
"""


extension_returns_none_file_contents = """

def my_extension():
    return None
"""


extension_adds_graph_file_contents = """
import cupy
import cudf
from cugraph.experimental import PropertyGraph

def my_extension(arg1, arg2, server):
    '''
    This extension creates a new graph, registers it with the server, and
    returns the new graph ID and some additional data.
    '''
    df = cudf.DataFrame({"src": [0, 1, 2],
                         "dst": [1, 2, 3],
                         "edge_prop": ["a", "b", "c"],
                         })
    pG = PropertyGraph()
    pG.add_edge_data(df, vertex_col_names=["src", "dst"])

    pG_gid = server.add_graph(pG)

    edge_df = pG.get_edge_data()
    values = cupy.array(edge_df[pG.edge_id_col_name] + arg1 + arg2)

    # UCX-Py transfers require cupy types, and cupy types are converted to host
    # for non-UCX-Py transfers.
    pG_gid = cupy.int8(pG_gid)

    return (pG_gid, values)
"""


###############################################################################
# module scope fixtures


@pytest.fixture(scope="module")
def server():
    """
    Start a cugraph_service server, stop it when done with the fixture.
    """
    from cugraph_service_client import CugraphServiceClient
    from cugraph_service_client.exceptions import CugraphServiceError

    host = "localhost"
    port = 9090
    client = CugraphServiceClient(host, port)

    try:
        client.uptime()
        print("FOUND RUNNING SERVER, ASSUMING IT SHOULD BE USED FOR TESTING!")
        yield

    except CugraphServiceError:
        # A server was not found, so start one for testing then stop it when
        # testing is done.
        server_process = utils.start_server_subprocess(host=host, port=port)

        # yield control to the tests, cleanup on return
        yield

        # tests are done, now stop the server
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


@pytest.fixture(scope="module")
def graph_creation_extension1():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension1_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def graph_creation_extension2():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension2_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def graph_creation_extension_long_running():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_long_running_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def graph_creation_extension_no_facade_arg():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_no_facade_arg_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def graph_creation_extension_bad_arg_order():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_bad_arg_order_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def graph_creation_extension_big_vertex_ids():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_big_vertex_ids_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def graph_creation_extension_empty_graph():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_empty_graph_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def graph_creation_extension_large_property_graph():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_large_property_graph_file_contents
    )

    yield tmp_extension_dir.name


# General (ie. not graph creation) extension


@pytest.fixture(scope="module")
def extension1():
    tmp_extension_dir = utils.create_tmp_extension_dir(extension1_file_contents)

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def extension_with_facade():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        extension_with_facade_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def extension_returns_none():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        extension_returns_none_file_contents
    )

    yield tmp_extension_dir.name


@pytest.fixture(scope="module")
def extension_adds_graph():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        extension_adds_graph_file_contents
    )

    yield tmp_extension_dir.name


###############################################################################
# function scope fixtures


@pytest.fixture(scope="function")
def client(server):
    """
    Creates a client instance to the running server, closes the client when the
    fixture is no longer used by tests.
    """
    from cugraph_service_client import CugraphServiceClient, defaults

    client = CugraphServiceClient(defaults.host, defaults.port)

    for gid in client.get_graph_ids():
        client.delete_graph(gid)

    # FIXME: should this fixture always unconditionally unload all extensions?
    # client.unload_graph_creation_extensions()

    # yield control to the tests, cleanup on return
    yield client

    client.close()
