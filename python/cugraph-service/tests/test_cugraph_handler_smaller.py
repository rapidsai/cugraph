# Copyright (c) 2023, NVIDIA CORPORATION.
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
import importlib
import traceback
from inspect import signature

from cugraph_service_client.exceptions import CugraphServiceError

class ExtensionServerFacade:
    def __init__(self, cugraph_handler):
        self.__handler = cugraph_handler

    def get_graph_ids(self):
        return self.__handler.get_graph_ids()

    def get_graph(self, graph_id):
        return self.__handler._get_graph(graph_id)

    def add_graph(self, G):
        return self.__handler._add_graph(G)


class CugraphHandler:
    """
    Class which handles RPC requests for a cugraph_service server.
    """

    # The name of the param that should be set to a ExtensionServerFacade
    # instance for server extension functions.
    __server_facade_extension_param_name = "server"

    def __init__(self):
        self.__next_graph_id = 1
        self.__graph_objs = {}
        self.__graph_creation_extensions = {}

    ###########################################################################
    # Environment management
    def load_graph_creation_extensions(self, extension_dir_or_mod_path):
        """
        Loads ("imports") all modules matching the pattern *_extension.py in the
        directory specified by extension_dir_or_mod_path. extension_dir_or_mod_path
        can be either a path to a directory on disk, or a python import path to a
        package.

        The modules are searched and their functions are called (if a match is
        found) when call_graph_creation_extension() is called.

        The extensions loaded are to be used for graph creation, and the server assumes
        the return value of the extension functions is a Graph-like object which is
        registered and assigned a unique graph ID.
        """
        modules_loaded = []
        try:
            extension_files = self.__get_extension_files_from_path(
                extension_dir_or_mod_path
            )

            for ext_file in extension_files:
                module_file_path = ext_file.absolute().as_posix()
                spec = importlib.util.spec_from_file_location(
                    module_file_path, ext_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.__graph_creation_extensions[module_file_path] = module
                modules_loaded.append(module_file_path)

            return modules_loaded

        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    def call_graph_creation_extension(
        self, func_name, func_args_repr, func_kwargs_repr
    ):
        """
        Calls the graph creation extension function func_name and passes it the
        eval'd func_args_repr and func_kwargs_repr objects.  If successful, it
        associates the graph returned by the extension function with a new graph
        ID and returns it.

        func_name cannot be a private name (name starting with __).
        """
        print("in call_graph_creation_extension")
        graph_obj = self.__call_extension(
            self.__graph_creation_extensions,
            func_name,
            func_args_repr,
            func_kwargs_repr,
        )
        # FIXME: ensure graph_obj is a graph obj
        return self._add_graph(graph_obj)

    ###########################################################################
    # "Protected" interface - used for both implementation and test/debug. Will
    # not be exposed to a cugraph_service client, but will be used by extensions
    # via the ExtensionServerFacade.
    def _add_graph(self, G):
        """
        Create a new graph ID for G and add G to the internal mapping of
        graph ID:graph instance.
        """
        print("in _add_graph")
        gid = self.__next_graph_id
        self.__graph_objs[gid] = G
        self.__next_graph_id += 1
        return gid

    ###########################################################################
    # Private

    @staticmethod
    def __get_extension_files_from_path(extension_dir_or_mod_path):
        print("in __get_extension_files_from_path: ", extension_dir_or_mod_path)
        extension_path = Path(extension_dir_or_mod_path)
        # extension_dir_path is either a path on disk or an importable module path
        # (eg. import foo.bar.module)
        if (not extension_path.exists()) or (not extension_path.is_dir()):
            try:
                mod = importlib.import_module(str(extension_path))
            except ModuleNotFoundError:
                raise CugraphServiceError(f"bad path: {extension_dir_or_mod_path}")

            mod_file_path = Path(mod.__file__).absolute()

            # If mod is a package, find all the .py files in it
            if mod_file_path.name == "__init__.py":
                extension_files = mod_file_path.parent.glob("*.py")
            else:
                extension_files = [mod_file_path]
        else:
            extension_files = extension_path.glob("*_extension.py")

        return extension_files

    def __call_extension(
        self, extension_dict, func_name, func_args_repr, func_kwargs_repr
    ):
        """
        Calls the extension function func_name and passes it the eval'd
        func_args_repr and func_kwargs_repr objects. If successful, returns a
        Value object containing the results returned by the extension function.

        The arg/kwarg reprs are eval'd prior to calling in order to pass actual
        python objects to func_name (this is needed to allow arbitrary arg
        objects to be serialized as part of the RPC call from the
        client).

        func_name cannot be a private name (name starting with __).

        All loaded extension modules are checked when searching for func_name,
        and the first extension module that contains it will have its function
        called.
        """
        print("in __call_extension")
        if func_name.startswith("__"):
            raise CugraphServiceError(f"Cannot call private function {func_name}")

        for module in extension_dict.values():
            func = getattr(module, func_name, None)
            if func is not None:
                # FIXME: look for a way to do this without using eval()
                func_args = eval(func_args_repr)
                func_kwargs = eval(func_kwargs_repr)
                func_sig = signature(func)
                func_params = list(func_sig.parameters.keys())
                facade_param = self.__server_facade_extension_param_name

                # Graph creation extensions that have the last arg named
                # self.__server_facade_extension_param_name are passed a
                # ExtensionServerFacade instance to allow them to query the
                # "server" in a safe way, if needed.
                if facade_param in func_params:
                    if func_params[-1] == facade_param:
                        func_kwargs[facade_param] = ExtensionServerFacade(self)
                    else:
                        raise CugraphServiceError(
                            f"{facade_param}, if specified, must be the last param."
                        )
                try:
                    return func(*func_args, **func_kwargs)
                except Exception:
                    # FIXME: raise a more detailed error
                    raise CugraphServiceError(
                        f"error running {func_name} : {traceback.format_exc()}"
                    )

        raise CugraphServiceError(f"extension {func_name} was not found")

print("start test")

handler = CugraphHandler()
num_loaded = handler.load_graph_creation_extensions(
    "tests/extensions"
)
assert len(num_loaded) == 1

gid1 = handler.call_graph_creation_extension(
    "create_graph_from_builtin_dataset", "('karate',)", "{}"
)

print("DONE")
