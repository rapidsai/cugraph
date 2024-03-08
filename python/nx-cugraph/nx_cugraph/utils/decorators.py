# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
from __future__ import annotations

from functools import partial, update_wrapper
from textwrap import dedent

import networkx as nx
from networkx.utils.decorators import nodes_or_number, not_implemented_for

from nx_cugraph.interface import BackendInterface

try:
    from networkx.utils.backends import _registered_algorithms
except ModuleNotFoundError:
    from networkx.classes.backends import _registered_algorithms


__all__ = ["not_implemented_for", "nodes_or_number", "networkx_algorithm"]


def networkx_class(api):
    def inner(func):
        func.__doc__ = getattr(api, func.__name__).__doc__
        return func

    return inner


class networkx_algorithm:
    name: str
    extra_doc: str | None
    extra_params: dict[str, str] | None
    version_added: str
    is_incomplete: bool
    is_different: bool
    _plc_names: set[str] | None

    def __new__(
        cls,
        func=None,
        *,
        name: str | None = None,
        # Extra parameter info that is added to NetworkX docstring
        extra_params: dict[str, str] | str | None = None,
        # Applies `nodes_or_number` decorator compatibly across versions (3.3 changed)
        nodes_or_number: list[int] | int | None = None,
        # Metadata (for introspection only)
        version_added: str,  # Required
        is_incomplete: bool = False,  # See self.extra_doc for details if True
        is_different: bool = False,  # See self.extra_doc for details if True
        _plc: str | set[str] | None = None,  # Hidden from user, may be removed someday
    ):
        if func is None:
            return partial(
                networkx_algorithm,
                name=name,
                extra_params=extra_params,
                nodes_or_number=nodes_or_number,
                version_added=version_added,
                is_incomplete=is_incomplete,
                is_different=is_different,
                _plc=_plc,
            )
        instance = object.__new__(cls)
        if nodes_or_number is not None and nx.__version__[:3] > "3.2":
            func = nx.utils.decorators.nodes_or_number(nodes_or_number)(func)
        # update_wrapper sets __wrapped__, which will be used for the signature
        update_wrapper(instance, func)
        instance.__defaults__ = func.__defaults__
        instance.__kwdefaults__ = func.__kwdefaults__
        instance.name = func.__name__ if name is None else name
        if extra_params is None:
            pass
        elif isinstance(extra_params, str):
            extra_params = {extra_params: ""}
        elif not isinstance(extra_params, dict):
            raise TypeError(
                f"extra_params must be dict, str, or None; got {type(extra_params)}"
            )
        instance.extra_params = extra_params
        if _plc is None or isinstance(_plc, set):
            instance._plc_names = _plc
        elif isinstance(_plc, str):
            instance._plc_names = {_plc}
        else:
            raise TypeError(
                f"_plc argument must be str, set, or None; got {type(_plc)}"
            )
        instance.version_added = version_added
        instance.is_incomplete = is_incomplete
        instance.is_different = is_different
        # The docstring on our function is added to the NetworkX docstring.
        instance.extra_doc = (
            dedent(func.__doc__.lstrip("\n").rstrip()) if func.__doc__ else None
        )
        # Copy __doc__ from NetworkX
        if instance.name in _registered_algorithms:
            instance.__doc__ = _registered_algorithms[instance.name].__doc__
        instance.can_run = _default_can_run
        setattr(BackendInterface, instance.name, instance)
        # Set methods so they are in __dict__
        instance._can_run = instance._can_run
        if nodes_or_number is not None and nx.__version__[:3] <= "3.2":
            instance = nx.utils.decorators.nodes_or_number(nodes_or_number)(instance)
        return instance

    def _can_run(self, func):
        """Set the `can_run` attribute to the decorated function."""
        if not func.__name__.startswith("_"):
            raise ValueError(
                "The name of the function used by `_can_run` must begin with '_'; "
                f"got: {func.__name__!r}"
            )
        self.can_run = func

    def __call__(self, /, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    def __reduce__(self):
        return _restore_networkx_dispatched, (self.name,)


def _default_can_run(*args, **kwargs):
    return True


def _restore_networkx_dispatched(name):
    return getattr(BackendInterface, name)
