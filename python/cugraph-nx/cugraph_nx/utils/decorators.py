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
import inspect
from functools import partial, update_wrapper

from networkx.utils.decorators import not_implemented_for

from cugraph_nx.interface import BackendInterface

__all__ = ["not_implemented_for", "networkx_algorithm"]


def networkx_class(api):
    def inner(func):
        func.__doc__ = getattr(api, func.__name__).__doc__
        return func

    return inner


class networkx_algorithm:
    def __new__(cls, func=None, *, name=None):
        if func is None:
            return partial(networkx_algorithm, name=name)
        instance = object.__new__(cls)
        update_wrapper(instance, func)
        instance.__defaults__ = func.__defaults__
        instance.__kwdefaults__ = func.__kwdefaults__
        instance.name = func.__name__ if name is None else name
        instance.can_run = _default_can_run
        setattr(BackendInterface, instance.name, instance)
        return instance

    @property
    def __signature__(self):
        return inspect.signature(self.__wrapped__)

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
