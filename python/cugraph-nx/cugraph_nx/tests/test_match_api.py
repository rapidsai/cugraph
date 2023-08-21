# Copyright (c) 2023, NVIDIA CORPORATION.
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
import importlib
import inspect

import networkx as nx

import cugraph_nx as cnx
from cugraph_nx.utils import networkx_algorithm


def test_match_signature_and_names():
    """Simple test to ensure our signatures and basic module layout match networkx."""
    for name, func in vars(cnx.interface.BackendInterface).items():
        if not isinstance(func, networkx_algorithm):
            continue
        dispatchable_func = nx.utils.backends._registered_algorithms[name]
        orig_func = dispatchable_func.orig_func
        # Matching signatures?
        orig_sig = inspect.signature(orig_func)
        func_sig = inspect.signature(func)
        if not func.extra_params:
            assert orig_sig == func_sig
        else:
            # Ignore extra parameters added to cugraph-nx algorithm
            assert orig_sig == func_sig.replace(
                parameters=[
                    p
                    for name, p in func_sig.parameters.items()
                    if name not in func.extra_params
                ]
            )
        if func.can_run is not cnx.utils.decorators._default_can_run:
            assert func_sig == inspect.signature(func.can_run)
        # Matching function names?
        assert func.__name__ == dispatchable_func.__name__ == orig_func.__name__
        # Matching dispatch names?
        assert func.name == dispatchable_func.name
        # Matching modules (i.e., where function defined)?
        assert (
            "networkx." + func.__module__.split(".", 1)[1]
            == dispatchable_func.__module__
            == orig_func.__module__
        )
        # Matching package layout (i.e., which modules have the function)?
        cnx_path = func.__module__
        name = func.__name__
        while "." in cnx_path:
            # This only walks up the module tree and does not check sibling modules
            cnx_path, mod_name = cnx_path.rsplit(".", 1)
            nx_path = cnx_path.replace("cugraph_nx", "networkx")
            cnx_mod = importlib.import_module(cnx_path)
            nx_mod = importlib.import_module(nx_path)
            # Is the function present in the current module?
            present_in_cnx = hasattr(cnx_mod, name)
            present_in_nx = hasattr(nx_mod, name)
            if present_in_cnx is not present_in_nx:  # pragma: no cover (debug)
                if present_in_cnx:
                    raise AssertionError(
                        f"{name} exists in {cnx_path}, but not in {nx_path}"
                    )
                raise AssertionError(
                    f"{name} exists in {nx_path}, but not in {cnx_path}"
                )
            # Is the nested module present in the current module?
            present_in_cnx = hasattr(cnx_mod, mod_name)
            present_in_nx = hasattr(nx_mod, mod_name)
            if present_in_cnx is not present_in_nx:  # pragma: no cover (debug)
                if present_in_cnx:
                    raise AssertionError(
                        f"{mod_name} exists in {cnx_path}, but not in {nx_path}"
                    )
                raise AssertionError(
                    f"{mod_name} exists in {nx_path}, but not in {cnx_path}"
                )
