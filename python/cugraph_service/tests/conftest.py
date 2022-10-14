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

from . import utils

graph_creation_extension1_file_contents = """
import cudf
from cugraph.experimental import PropertyGraph

def custom_graph_creation_function(server):
   edgelist = cudf.DataFrame(columns=['src', 'dst'],
                             data=[(0, 77), (1, 88), (2, 99)])
   pG = PropertyGraph()
   pG.add_edge_data(edgelist, vertex_col_names=('src', 'dst'))

   # smoke test the server object by accesing the "mg" attr
   server.is_mg

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
   if server.is_mg:
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
   if server.is_mg:
      pG = MGPropertyGraph()
   else:
      pG = PropertyGraph()
   big_num = (2**32)+1
   df = cudf.DataFrame({"vert_id":cupy.arange(big_num, big_num+10,
                                              dtype="int64"),
                        "vert_prop":cupy.arange(big_num+100, big_num+110,
                                                dtype="int64")})
   if server.is_mg:
      df = dask_cudf.from_cudf(df, npartitions=2)
   pG.add_vertex_data(df, vertex_col_name="vert_id")

   df = cudf.DataFrame({"src":cupy.arange(big_num, big_num+10, dtype="int64"),
                        "dst":cupy.arange(big_num+1,big_num+11, dtype="int64"),
                        "edge_prop":cupy.arange(big_num+100, big_num+110,
                                                dtype="int64")})
   if server.is_mg:
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
   if server.is_mg:
      pG = MGPropertyGraph()
   else:
      pG = PropertyGraph()

   num_verts = 10e6
   df = cudf.DataFrame({"vert_id":cupy.arange(num_verts, dtype="int32"),
                        "vert_prop":cupy.arange(num_verts, dtype="int32"),
                        })
   if server.is_mg:
      df = dask_cudf.from_cudf(df, npartitions=2)
   pG.add_vertex_data(df, vertex_col_name="vert_id")

   df = cudf.DataFrame({"src":cupy.arange(num_verts, dtype="int32"),
                        "dst":cupy.arange(1, num_verts+1, dtype="int32"),
                        "edge_prop":cupy.arange(num_verts, dtype="int32"),
                        })
   if server.is_mg:
      df = dask_cudf.from_cudf(df, npartitions=2)
   pG.add_edge_data(df, vertex_col_names=["src", "dst"])

   return pG
"""


###############################################################################
# module scope fixtures


@pytest.fixture(scope="module")
def graph_creation_extension1():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension1_file_contents
    )

    yield tmp_extension_dir


@pytest.fixture(scope="module")
def graph_creation_extension2():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension2_file_contents
    )

    yield tmp_extension_dir


@pytest.fixture(scope="module")
def graph_creation_extension_long_running():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_long_running_file_contents
    )

    yield tmp_extension_dir


@pytest.fixture(scope="module")
def graph_creation_extension_no_facade_arg():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_no_facade_arg_file_contents
    )

    yield tmp_extension_dir


@pytest.fixture(scope="module")
def graph_creation_extension_bad_arg_order():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_bad_arg_order_file_contents
    )

    yield tmp_extension_dir


@pytest.fixture(scope="module")
def graph_creation_extension_big_vertex_ids():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_big_vertex_ids_file_contents
    )

    yield tmp_extension_dir


@pytest.fixture(scope="module")
def graph_creation_extension_empty_graph():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_empty_graph_file_contents
    )

    yield tmp_extension_dir


@pytest.fixture(scope="module")
def graph_creation_extension_large_property_graph():
    tmp_extension_dir = utils.create_tmp_extension_dir(
        graph_creation_extension_large_property_graph_file_contents
    )

    yield tmp_extension_dir
