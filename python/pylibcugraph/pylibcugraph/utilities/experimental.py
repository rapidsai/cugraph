# Copyright (c) 2022, NVIDIA CORPORATION.
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

import functools
import types
import warnings
import sys

import pylibcugraph

def add_obj_to_pylibcugraph_subnamespace(obj, sub_ns_name,
                                         sub_sub_ns_name=None,
                                         new_obj_name=None):
    """
    Adds an obj to the pylibcugraph.<sub_ns_name>.<sub_sub_ns_name> namespace,
    using the objects current namespace names under pylibcugraph as the default
    sub_sub_ns_name, creating the sub-namespace and sub-sub-namespaces if
    necessary.

    Example:
        for object foo in pylibcugraph.structure
            add_obj_to_pylibcugraph_subnamespace(foo, "experimental")
        results in:
            pylibcugraph.experimental.structure.foo

    All namespaces - current and new - must be under "pylibcugraph".

    If sub_sub_ns_name is provided, it will be used to override the obj's
    current namespace under pylibcugraph.
    Example:
        for object foo in pylibcugraph.structure
            add_obj_to_pylibcugraph_subnamespace(foo, "experimental",
                                                 sub_sub_ns_name="bar.baz")
        results in:
            pylibcugraph.experimental.bar.baz.foo

        for object foo in pylibcugraph.structure
            add_obj_to_pylibcugraph_subnamespace(foo, "experimental",
                                                 sub_sub_ns_name="")
        results in:
            pylibcugraph.experimental.foo

    Example:
        for object foo in pylibcugraph.structure:
            add_obj_to_pylibcugraph_subnamespace(foo, "experimental",
                                                 new_obj_name="new_pagerank")
        results in:
            pylibcugraph.experimental.structure.new_pagerank

    Returns a tuple of:
    (new namespace name, new obj name, new namespace module obj)
    """
    # Create a list of names representing the ns to create
    current_mod_name_parts = obj.__module__.split(".")
    # All namespaces - current and new - must be under "pylibcugraph"
    if current_mod_name_parts[0] != "pylibcugraph":
        raise ValueError(f"{obj.__name__} is not under the pylibcugraph "
                         "package")
    new_mod_name_parts = [sub_ns_name]
    if sub_sub_ns_name is None:
        new_mod_name_parts += current_mod_name_parts[1:]
    else:
        if sub_sub_ns_name != "":
            new_mod_name_parts += sub_sub_ns_name.split(".")

    # Create the new namespace
    mod_to_update = pylibcugraph
    mod_name_parts = ["pylibcugraph"]

    for ns in new_mod_name_parts:
        mod_name_parts.append(ns)
        if not(hasattr(mod_to_update, ns)):
            mod_to_update_name = ".".join(mod_name_parts)
            new_mod = types.ModuleType(mod_to_update_name)
            setattr(mod_to_update, ns, new_mod)
            sys.modules[mod_to_update_name] = new_mod
        mod_to_update = getattr(mod_to_update, ns)

    # Add obj to the new namespace
    new_obj_name = obj.__name__ if new_obj_name is None else new_obj_name
    setattr(mod_to_update, new_obj_name, obj)

    return (".".join(["pylibcugraph"] + new_mod_name_parts),
            new_obj_name,
            mod_to_update)


def get_callable_for_experimental(sub_namespace_name=None):
    """
    Returns a callable which can be used as the return value for the
    "experimental" decorator function, or as something which can be called
    directly.  Calling the returned callable with an object as the arg results
    in the object being added to the "experimental" namespace as described in
    the docstring for the experimental decorator function.

    If sub_namespace_name is provided, the returned callable will add the object
    to the sub namespace under experimental as described in the docstring for
    the experimental decorator function.
    """
    def experimental_ns_updater(obj):
        (new_ns_name, new_obj_name, new_ns) = \
            add_obj_to_pylibcugraph_subnamespace(
                obj,
                sub_ns_name="experimental",
                sub_sub_ns_name=sub_namespace_name,
                new_obj_name=obj.__name__.lstrip("__"))
        # Wrap the function in a function that prints a warning before
        # calling the obj. This is done after adding obj to the
        # experimental namespace so the warning message will have the
        # properly-generated experimental names.
        warning_msg = (f"{new_ns_name}.{new_obj_name} is experimental and "
                       "will change in a future release.")
        # built-in/extension types cannot have these attrs set
        try:
            obj.__module__ = new_ns_name
            obj.__qualname__ = new_obj_name
        except TypeError:
            pass
        @functools.wraps(obj)
        def call_with_warning(*args, **kwargs):
            warnings.warn(warning_msg, PendingDeprecationWarning)
            return obj(*args, **kwargs)

        wrapped_obj = call_with_warning
        # Replace obj in the experimental ns with the wrapped obj
        setattr(new_ns, new_obj_name, wrapped_obj)
        return obj
    return experimental_ns_updater


def experimental(*args, **kwargs):
    """
    Decorator function to add an obj to the pylibcugraph.experimental
    namespace.

    If no args are given, obj is copied to
    pylibcugraph.experimental.<current pylibcugraph subnamespace>.obj.

    Example:
        for the foo function in pylibcugraph.structure:
            @experimental
            def foo(...)
        results in:
            a foo() function in the
            pylibcugraph.experimental.structure namespace.

    If the sub_ns_name kwarg is given, it is used to replace the default
    subnamespace under pylibcugraph that the obj currently resides in.
    Example:
        for the foo function in pylibcugraph.structure:
            @experimental(sub_ns_name="bar")
            def foo(...)
        results in:
            a foo() function in the
            pylibcugraph.experimental.bar namespace.

        for the foo function in pylibcugraph.structure:
            @experimental(sub_ns_name="")
            def foo(...)
        results in:
            a foo() function added directly to
            pylibcugraph.experimental

    If the current obj is private by naming it with a leading __, the leading
    __ is removed from the obj in the new namespace.  This allows an
    experimental class/function to be private (hidden) in a non-experimental
    namespace but public in experimental.
    Example:
        for the __foo function in pylibcugraph.structure:
            @experimental(sub_ns_name="bar")
            def __pagerank(...)
        results in:
            a foo() function in the
            pylibcugraph.experimental.bar namespace.
    """
    kwa = list(kwargs.keys())
    if kwa and kwa != ["sub_ns_name"]:
        raise TypeError("Only the 'sub_ns_name' kwarg is allowed for "
                        "experimental()")
    sub_ns_name = kwargs.get("sub_ns_name")

    # python expects decorators to return function wrappers one of two ways: if
    # args specified, the decorator must return a callable that accepts an obj
    # to wrap. If no args, then the decorator returns the wrapped obj directly.

    if sub_ns_name is not None:  # called as @experimental(sub_ns_name="...")
        # Python will call the callable being returned here, which will then
        # setup the new ns, add the obj to it, and return the original obj
        return get_callable_for_experimental(sub_ns_name)

    else:  # called as @experimental
        if len(args) > 1:
            raise TypeError("Too many positional args to experimental()")
        obj = args[0]
        # Get a callable to update the experimental namespace and call it
        # directly here.
        update_experimental = get_callable_for_experimental()
        update_experimental(obj)
        # Return the obj as-is, no need to return anything wrapped since
        # update_experimental() did all the work.
        return obj
