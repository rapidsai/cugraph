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

import pickle

import pytest


###############################################################################
## fixtures
# The fixtures used in these tests are defined in conftest.py


###############################################################################
## tests

def test_load_and_call_graph_creation_extension(graph_creation_extension2):
    """
    Ensures load_extensions reads the extensions and makes the new APIs they add
    available.
    """
    from gaas_server.gaas_handler import GaasHandler
    from gaas_client.exceptions import GaasError

    handler = GaasHandler()

    extension_dir = graph_creation_extension2

    # DNE
    with pytest.raises(GaasError):
        handler.load_graph_creation_extensions("/path/that/does/not/exist")

    # Exists, but is a file
    with pytest.raises(GaasError):
        handler.load_graph_creation_extensions(__file__)

    # Load the extension and call the function defined in it
    num_files_read = handler.load_graph_creation_extensions(extension_dir)
    assert num_files_read == 1

    # Private function should not be callable
    with pytest.raises(GaasError):
        handler.call_graph_creation_extension("__my_private_function",
                                              "()", "{}")

    # Function which DNE in the extension
    with pytest.raises(GaasError):
        handler.call_graph_creation_extension("bad_function_name",
                                              "()", "{}")

    # Wrong number of args
    with pytest.raises(GaasError):
        handler.call_graph_creation_extension("my_graph_creation_function",
                                              "('a',)", "{}")

    # This call should succeed and should result in a new PropertyGraph present
    # in the handler instance.
    new_graph_ID = handler.call_graph_creation_extension(
        "my_graph_creation_function", "('a', 'b', 'c')", "{}")

    assert new_graph_ID in handler.get_graph_ids()

    # Inspect the PG and ensure it was created from my_graph_creation_function
    pG = handler._get_graph(new_graph_ID)
    edge_props = pG.edge_property_names
    assert ("c" in edge_props)


def test_load_and_unload_graph_creation_extension(graph_creation_extension2):
    """
    Ensure extensions can be unloaded.
    """
    from gaas_server.gaas_handler import GaasHandler
    from gaas_client.exceptions import GaasError

    handler = GaasHandler()

    extension_dir = graph_creation_extension2

    # Load the extensions and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_ID = handler.call_graph_creation_extension(
        "my_graph_creation_function", "('a', 'b', 'c')", "{}")
    assert new_graph_ID in handler.get_graph_ids()

    # Unload then try to run the same call again, which should fail
    handler.unload_graph_creation_extensions()

    with pytest.raises(GaasError):
        handler.call_graph_creation_extension(
            "my_graph_creation_function", "('a', 'b', 'c')", "{}")


def test_load_and_unload_graph_creation_extension_no_args(
        graph_creation_extension1):
    """
    Test graph_creation_extension1 which contains an extension with no args.
    """
    from gaas_server.gaas_handler import GaasHandler
    handler = GaasHandler()

    extension_dir = graph_creation_extension1

    # Load the extensions and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_ID = handler.call_graph_creation_extension(
        "custom_graph_creation_function", "()", "{}")
    assert new_graph_ID in handler.get_graph_ids()


def test_load_and_unload_graph_creation_extension_no_facade_arg(
        graph_creation_extension_no_facade_arg):
    """
    Test an extension that has no facade arg.
    """
    from gaas_server.gaas_handler import GaasHandler
    handler = GaasHandler()

    extension_dir = graph_creation_extension_no_facade_arg

    # Load the extensions and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_ID = handler.call_graph_creation_extension(
        "graph_creation_function", "('a')", "{'arg2':33}")
    assert new_graph_ID in handler.get_graph_ids()


def test_load_and_unload_graph_creation_extension_bad_arg_order(
        graph_creation_extension_bad_arg_order):
    """
    Test an extension that has the facade arg in the wrong position.
    """
    from gaas_server.gaas_handler import GaasHandler
    from gaas_client.exceptions import GaasError

    handler = GaasHandler()

    extension_dir = graph_creation_extension_bad_arg_order

    # Load the extensions and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    with pytest.raises(GaasError):
        handler.call_graph_creation_extension(
            "graph_creation_function", "('a', 'b')", "{}")


def test_get_graph_data_large_vertex_ids(
        graph_creation_extension_big_vertex_ids):
    """
    Test that graphs with large vertex ID values (>int32) are handled.
    """
    from gaas_server.gaas_handler import GaasHandler

    handler = GaasHandler()

    extension_dir = graph_creation_extension_big_vertex_ids

    # Load the extension and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_id = handler.call_graph_creation_extension(
        "graph_creation_function_vert_and_edge_data_big_vertex_ids", "()", "{}")

    invalid_vert_id = 2
    vert_data = handler.get_graph_vertex_data(
    id_or_ids=invalid_vert_id,
        null_replacement_value=0,
        graph_id=new_graph_id,
        property_keys=None)

    assert len(pickle.loads(vert_data)) == 0

    large_vert_id = (2**32)+1
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=large_vert_id,
        null_replacement_value=0,
        graph_id=new_graph_id,
        property_keys=None)

    assert len(pickle.loads(vert_data)) == 1

    invalid_edge_id = (2**32)+1
    edge_data = handler.get_graph_edge_data(
    id_or_ids=invalid_edge_id,
        null_replacement_value=0,
        graph_id=new_graph_id,
        property_keys=None)

    assert len(pickle.loads(edge_data)) == 0

    small_edge_id = 2
    edge_data = handler.get_graph_edge_data(
        id_or_ids=small_edge_id,
        null_replacement_value=0,
        graph_id=new_graph_id,
        property_keys=None)

    assert len(pickle.loads(edge_data)) == 1

def test_get_graph_data_empty_graph(graph_creation_extension_empty_graph):
    """
    Tests that get_graph_*_data() handles empty graphs correctly.
    """
    from gaas_server.gaas_handler import GaasHandler

    handler = GaasHandler()

    extension_dir = graph_creation_extension_empty_graph

    # Load the extension and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_id = handler.call_graph_creation_extension(
        "graph_creation_function", "()", "{}")

    invalid_vert_id = 2
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=invalid_vert_id,
        null_replacement_value=0,
        graph_id=new_graph_id,
        property_keys=None)

    assert len(pickle.loads(vert_data)) == 0

    invalid_edge_id = 2
    edge_data = handler.get_graph_edge_data(
        id_or_ids=invalid_vert_id,
        null_replacement_value=0,
        graph_id=new_graph_id,
        property_keys=None)

    assert len(pickle.loads(edge_data)) == 0
