# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import sys
import pickle
from pathlib import Path

import pytest


# FIXME: Remove this once these pass in the CI environment.
pytest.skip(
    reason="FIXME: many of these tests fail in CI and are currently run "
    "manually only in dev environments.",
    allow_module_level=True,
)

###############################################################################
# fixtures
# The fixtures used in these tests are defined in conftest.py


###############################################################################
# tests


def test_load_and_call_graph_creation_extension(graph_creation_extension2):
    """
    Ensures load_extensions reads the extensions and makes the new APIs they
    add available.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler
    from cugraph_service_client.exceptions import CugraphServiceError

    handler = CugraphHandler()

    extension_dir = graph_creation_extension2

    # DNE
    with pytest.raises(CugraphServiceError):
        handler.load_graph_creation_extensions("/path/that/does/not/exist")

    # Exists, but is a file
    with pytest.raises(CugraphServiceError):
        handler.load_graph_creation_extensions(__file__)

    # Load the extension and call the function defined in it
    ext_mod_names = handler.load_graph_creation_extensions(extension_dir)
    assert len(ext_mod_names) == 1
    expected_mod_name = (Path(extension_dir) / "my_extension.py").as_posix()
    assert ext_mod_names[0] == expected_mod_name

    # Private function should not be callable
    with pytest.raises(CugraphServiceError):
        handler.call_graph_creation_extension("__my_private_function", "()", "{}")

    # Function which DNE in the extension
    with pytest.raises(CugraphServiceError):
        handler.call_graph_creation_extension("bad_function_name", "()", "{}")

    # Wrong number of args
    with pytest.raises(CugraphServiceError):
        handler.call_graph_creation_extension(
            "my_graph_creation_function", "('a',)", "{}"
        )

    # This call should succeed and should result in a new PropertyGraph present
    # in the handler instance.
    new_graph_ID = handler.call_graph_creation_extension(
        "my_graph_creation_function", "('a', 'b', 'c')", "{}"
    )

    assert new_graph_ID in handler.get_graph_ids()

    # Inspect the PG and ensure it was created from my_graph_creation_function
    pG = handler._get_graph(new_graph_ID)
    edge_props = pG.edge_property_names
    assert "c" in edge_props


def test_load_call_unload_extensions(graph_creation_extension2, extension1):
    """
    Ensure extensions can be loaded, run, and unloaded.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler
    from cugraph_service_client.exceptions import CugraphServiceError

    handler = CugraphHandler()

    graph_creation_extension_dir = graph_creation_extension2
    extension_dir = extension1

    # Loading
    gc_ext_mod_names = handler.load_graph_creation_extensions(
        graph_creation_extension_dir
    )
    ext_mod_names = handler.load_extensions(extension_dir)

    # Running
    new_graph_ID = handler.call_graph_creation_extension(
        "my_graph_creation_function", "('a', 'b', 'c')", "{}"
    )
    assert new_graph_ID in handler.get_graph_ids()

    results = handler.call_extension(
        "my_nines_function", "(33, 'int32', 21, 'float64')", "{}"
    )
    # results is a ValueWrapper object which Thrift will understand to be a
    # Value, which it can serialize. Check the ValueWrapper object here.
    assert len(results.list_value) == 2
    assert len(results.list_value[0].list_value) == 33
    assert len(results.list_value[1].list_value) == 21
    assert type(results.list_value[0].list_value[0].int32_value) is int
    assert type(results.list_value[1].list_value[0].double_value) is float
    assert results.list_value[0].list_value[0].int32_value == 9
    assert results.list_value[1].list_value[0].double_value == 9.0

    # Unloading
    with pytest.raises(CugraphServiceError):
        handler.unload_extension_module("invalid_module")

    for mod_name in gc_ext_mod_names:
        handler.unload_extension_module(mod_name)

    with pytest.raises(CugraphServiceError):
        handler.call_graph_creation_extension(
            "my_graph_creation_function", "('a', 'b', 'c')", "{}"
        )

    handler.call_extension("my_nines_function", "(33, 'int32', 21, 'float64')", "{}")

    for mod_name in ext_mod_names:
        handler.unload_extension_module(mod_name)

    with pytest.raises(CugraphServiceError):
        handler.call_extension(
            "my_nines_function", "(33, 'int32', 21, 'float64')", "{}"
        )


def test_extension_with_facade_graph_access(
    graph_creation_extension1, extension_with_facade
):
    """
    Creates a Graph then calls an extension that accesses the graph in order to
    return data.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()
    gc_extension_dir = graph_creation_extension1
    extension_dir = extension_with_facade

    # Load the extensions - use the graph creation extension to create a known PG
    # for use by the extension being tested.
    handler.load_graph_creation_extensions(gc_extension_dir)
    handler.load_extensions(extension_dir)

    new_graph_ID = handler.call_graph_creation_extension(
        "custom_graph_creation_function", "()", "{}"
    )
    assert new_graph_ID in handler.get_graph_ids()

    val1 = 33
    val2 = 22.1

    # Call the extension under test, it will access the PG loaded above to return
    # results. This extension just adds val1 + val2 to each edge ID.
    results = handler.call_extension("my_extension", f"({val1}, {val2})", "{}")

    # results is a ValueWrapper object which Thrift will understand to be a Value, which
    # it can serialize. Check the ValueWrapper object here, it should contain the 3 edge
    # IDs starting from 0 with the values added to each.
    assert len(results.list_value) == 3
    assert results.list_value[0].double_value == 0 + val1 + val2
    assert results.list_value[1].double_value == 1 + val1 + val2
    assert results.list_value[2].double_value == 2 + val1 + val2


def test_load_call_unload_testing_extensions():
    """ """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()
    num_loaded = handler.load_graph_creation_extensions(
        "cugraph_service_server.testing.benchmark_server_extension"
    )
    assert len(num_loaded) == 1

    gid1 = handler.call_graph_creation_extension(
        "create_graph_from_builtin_dataset", "('karate',)", "{}"
    )
    scale = 2
    edgefactor = 2
    gid2 = handler.call_graph_creation_extension(
        "create_graph_from_rmat_generator",
        "()",
        f"{{'scale': {scale}, 'num_edges': {(scale**2) * edgefactor}, "
        "'seed': 42, 'mg': False}",
    )
    assert gid1 != gid2

    graph_info1 = handler.get_graph_info(keys=[], graph_id=gid1)
    # since the handler returns a dictionary of objs used byt her serialization
    # code, convert each item to a native python type for easy checking.
    graph_info1 = {k: v.get_py_obj() for (k, v) in graph_info1.items()}
    assert graph_info1["num_vertices"] == 34
    assert graph_info1["num_edges"] == 78
    graph_info2 = handler.get_graph_info(keys=[], graph_id=gid2)
    graph_info2 = {k: v.get_py_obj() for (k, v) in graph_info2.items()}
    assert graph_info2["num_vertices"] <= 4
    assert graph_info2["num_edges"] <= 8


def test_load_call_unload_extensions_python_module_path(extension1):
    """
    Load, run, unload an extension that was loaded using a python module
    path (as would be used by an import statement) instead of a file path.
    """
    from cugraph_service_client.exceptions import CugraphServiceError
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()
    extension_dir = extension1
    extension_dir_path = Path(extension_dir).absolute()
    package_name = extension_dir_path.name  # last name in the path only

    # Create an __init__py file and add the dir to sys.path so it can be
    # imported as a package.
    with open(extension_dir_path / "__init__.py", "w") as f:
        f.write("")
    # FIXME: this should go into a fixture which can unmodify sys.path when done
    sys.path.append(extension_dir_path.parent.as_posix())

    # Create another .py file to test multiple module loading
    with open(extension_dir_path / "foo.py", "w") as f:
        f.write("def foo_func(): return 33")

    # Load everything in the package, ext_mod_names should be a list of python
    # files containing 3 files (2 modules + __init__.py file).
    # Assume the .py file in the generated extension dir is named
    # "my_extension.py"
    ext_mod_names1 = handler.load_extensions(package_name)
    assert len(ext_mod_names1) == 3
    assert str(extension_dir_path / "my_extension.py") in ext_mod_names1
    assert str(extension_dir_path / "foo.py") in ext_mod_names1
    assert str(extension_dir_path / "__init__.py") in ext_mod_names1

    results = handler.call_extension(
        "my_nines_function", "(33, 'int32', 21, 'float64')", "{}"
    )
    assert results.list_value[0].list_value[0].int32_value == 9
    assert results.list_value[1].list_value[0].double_value == 9.0

    result = handler.call_extension("foo_func", "()", "{}")
    assert result.int32_value == 33

    # unload
    for mod_name in ext_mod_names1:
        handler.unload_extension_module(mod_name)

    with pytest.raises(CugraphServiceError):
        handler.call_extension(
            "my_nines_function", "(33, 'int32', 21, 'float64')", "{}"
        )
    with pytest.raises(CugraphServiceError):
        handler.call_extension("foo_func", "()", "{}")

    # Load just an individual module in the package, ext_mod_names should only
    # contain 1 file.
    mod_name = f"{package_name}.my_extension"
    ext_mod_names2 = handler.load_extensions(mod_name)
    assert ext_mod_names2 == [str(extension_dir_path / "my_extension.py")]

    results = handler.call_extension(
        "my_nines_function", "(33, 'int32', 21, 'float64')", "{}"
    )
    assert results.list_value[0].list_value[0].int32_value == 9
    assert results.list_value[1].list_value[0].double_value == 9.0

    for mod_name in ext_mod_names2:
        handler.unload_extension_module(mod_name)

    with pytest.raises(CugraphServiceError):
        handler.call_extension(
            "my_nines_function", "(33, 'int32', 21, 'float64')", "{}"
        )


def test_load_call_unload_graph_creation_extension_no_args(graph_creation_extension1):

    """
    Test graph_creation_extension1 which contains an extension with no args.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()

    extension_dir = graph_creation_extension1

    # Load the extensions and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_ID = handler.call_graph_creation_extension(
        "custom_graph_creation_function", "()", "{}"
    )
    assert new_graph_ID in handler.get_graph_ids()


def test_load_call_unload_graph_creation_extension_no_facade_arg(
    graph_creation_extension_no_facade_arg,
):
    """
    Test an extension that has no facade arg.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()

    extension_dir = graph_creation_extension_no_facade_arg

    # Load the extensions and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_ID = handler.call_graph_creation_extension(
        "graph_creation_function", "('a')", "{'arg2':33}"
    )
    assert new_graph_ID in handler.get_graph_ids()


def test_load_call_unload_graph_creation_extension_bad_arg_order(
    graph_creation_extension_bad_arg_order,
):
    """
    Test an extension that has the facade arg in the wrong position.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler
    from cugraph_service_client.exceptions import CugraphServiceError

    handler = CugraphHandler()

    extension_dir = graph_creation_extension_bad_arg_order

    # Load the extensions and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    with pytest.raises(CugraphServiceError):
        handler.call_graph_creation_extension(
            "graph_creation_function", "('a', 'b')", "{}"
        )


def test_get_graph_data_large_vertex_ids(graph_creation_extension_big_vertex_ids):
    """
    Test that graphs with large vertex ID values (>int32) are handled.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()

    extension_dir = graph_creation_extension_big_vertex_ids

    # Load the extension and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_id = handler.call_graph_creation_extension(
        "graph_creation_function_vert_and_edge_data_big_vertex_ids", "()", "{}"
    )

    invalid_vert_id = 2
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=invalid_vert_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(vert_data)) == 0

    large_vert_id = (2**32) + 1
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=large_vert_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(vert_data)) == 1

    invalid_edge_id = (2**32) + 1
    edge_data = handler.get_graph_edge_data(
        id_or_ids=invalid_edge_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(edge_data)) == 0

    small_edge_id = 2
    edge_data = handler.get_graph_edge_data(
        id_or_ids=small_edge_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(edge_data)) == 1


def test_get_graph_data_empty_graph(graph_creation_extension_empty_graph):
    """
    Tests that get_graph_*_data() handles empty graphs correctly.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()

    extension_dir = graph_creation_extension_empty_graph

    # Load the extension and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_id = handler.call_graph_creation_extension(
        "graph_creation_function", "()", "{}"
    )

    invalid_vert_id = 2
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=invalid_vert_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(vert_data)) == 0

    invalid_edge_id = 2
    edge_data = handler.get_graph_edge_data(
        id_or_ids=invalid_edge_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(edge_data)) == 0


def test_get_server_info(graph_creation_extension1, extension1):
    """
    Ensures the server meta-data from get_server_info() is correct. This
    includes information about loaded extensions, so fixtures that provide
    extensions to be loaded are used.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()

    handler.load_graph_creation_extensions(graph_creation_extension1)
    handler.load_extensions(extension1)

    meta_data = handler.get_server_info()
    assert meta_data["num_gpus"].int32_value is not None
    assert (
        str(
            Path(
                meta_data["graph_creation_extensions"].list_value[0].get_py_obj()
            ).parent
        )
        == graph_creation_extension1
    )
    assert (
        str(Path(meta_data["extensions"].list_value[0].get_py_obj()).parent)
        == extension1
    )
