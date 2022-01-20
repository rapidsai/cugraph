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
import warnings
import inspect

experimental_prefix = "EXPERIMENTAL"


def experimental_warning_wrapper(obj, make_public_name=True):
    """
    Return a callable obj wrapped in a callable the prints a warning about it
    being "experimental" (an object that is in the public API but subject to
    change or removal) prior to calling it and returning its value.

    If make_public_name is False, the object's name used in the warning message
    is left unmodified. If True (default), any leading __ and/or EXPERIMENTAL
    string are removed from the name used in warning messages. This allows an
    object to be named with a "private" name in the public API so it can remain
    hidden while it is still experimental, but have a public name within the
    experimental namespace so it can be easily discovered and used.
    """
    obj_name = obj.__qualname__
    if make_public_name:
        obj_name = obj_name.lstrip(experimental_prefix)
        obj_name = obj_name.lstrip("__")

    # Assume the caller of this function is the module containing the
    # experimental obj and try to get its namespace name. Default to no
    # namespace name if it could not be found.
    call_stack = inspect.stack()
    calling_frame = call_stack[1].frame
    ns_name = calling_frame.f_locals.get("__name__")
    if ns_name is not None:
        ns_name += "."
    else:
        ns_name = ""

    warning_msg = (f"{ns_name}{obj_name} is experimental and will change "
                   "or be removed in a future release.")

    @functools.wraps(obj)
    def callable_warning_wrapper(*args, **kwargs):
        warnings.warn(warning_msg, PendingDeprecationWarning)
        return obj(*args, **kwargs)

    return callable_warning_wrapper
