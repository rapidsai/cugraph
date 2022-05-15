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

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

graph_creation_extension_file_contents = """
import cudf
from cugraph.experimental import PropertyGraph

def __my_private_function():
   pass

def my_graph_creation_function(arg1, arg2):
   edgelist = cudf.DataFrame(columns=[arg1, arg2], data=[(0,1), (88,99)])
   pG = PropertyGraph()
   pG.add_edge_data(edgelist, vertex_col_names=(arg1, arg2))
   return pG
"""

###############################################################################
## fixtures

@pytest.fixture
def graph_creation_extension():

    with TemporaryDirectory() as tmp_extension_dir:
        # write graph creation extension .py file
        graph_creation_extension_file = open(
            Path(tmp_extension_dir)/"my_graph_creation_extension.py",
            "w")
        print(graph_creation_extension_file_contents,
              file=graph_creation_extension_file,
              flush=True)

        yield tmp_extension_dir


###############################################################################
## tests

def test_load_graph_creation_extensions(graph_creation_extension):
    """
    Ensures load_extensions reads the extensions and makes the new APIs they add
    available.
    """
    from gaas_server.gaas_handler import GaasHandler
    handler = GaasHandler()

    extension_dir = graph_creation_extension

    # DNE
    with pytest.raises(NotADirectoryError):
        handler.load_graph_creation_extensions("/path/that/does/not/exist")

    # Exists, but is a file
    with pytest.raises(NotADirectoryError):
        handler.load_graph_creation_extensions(__file__)

    # Load the extension and call the function defined in it
    handler.load_graph_creation_extensions(extension_dir)

    # Private function should not be callable
    with pytest.raises(AttributeError):
        handler.call_graph_creation_extension("__my_private_function")

    # Function which DNE in the extension
    with pytest.raises(AttributeError):
        handler.call_graph_creation_extension("bad_function_name")

    # Wrong number of args
    with pytest.raises(TypeError):
        handler.call_graph_creation_extension("my_graph_creation_function", "a")

    # This call should succeed and should result in a new PropertyGraph present
    # in the handler instance.
    new_graph_ID = handler.call_graph_creation_extension(
        "my_graph_creation_function", "a", "b")

    assert new_graph_ID in handler.get_graph_ids()

    # Inspect the PG and ensure it was created from my_graph_creation_function
    pG = handler._get_graph(new_graph_ID)
    edge_props = pG.edge_property_names
    assert ("a" in edge_props) and ("b" in edge_props)
