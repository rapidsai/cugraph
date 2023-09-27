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
from __future__ import annotations

from functools import partial, update_wrapper

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

    def __new__(
        cls,
        func=None,
        *,
        name: str | None = None,
        extra_params: dict[str, str] | str | None = None,
    ):
        if func is None:
            return partial(networkx_algorithm, name=name, extra_params=extra_params)
        instance = object.__new__(cls)
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
        # The docstring on our function is added to the NetworkX docstring.
        instance.extra_doc = func.__doc__
        # Copy __doc__ from NetworkX
        if instance.name in _registered_algorithms:
            instance.__doc__ = _registered_algorithms[instance.name].__doc__
        instance.can_run = _default_can_run
        setattr(BackendInterface, instance.name, instance)
        # Set methods so they are in __dict__
        instance._can_run = instance._can_run
        return instance

    def _can_run(self, func):
        """Set the `can_run` attribute to the decorated function."""
        self.can_run = func

    def __call__(self, /, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    def __reduce__(self):
        return _restore_networkx_dispatched, (self.name,)


def _default_can_run(*args, **kwargs):
    return True


def _restore_networkx_dispatched(name):
    return getattr(BackendInterface, name)
