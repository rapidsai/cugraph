# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

# NOTE:
# This contains code with copyright by the scikit-learn project, subject to the
# license in /thirdparty/LICENSES/LICENSE.scikit_learn

import inspect
import re
import subprocess
from functools import partial
from operator import attrgetter

orig = inspect.isfunction


# See https://opendreamkit.org/2017/06/09/CythonSphinx/
def isfunction(obj):

    orig_val = orig(obj)

    new_val = hasattr(type(obj), "__code__")

    if (orig_val != new_val):
        return new_val

    return orig_val


inspect.isfunction = isfunction

REVISION_CMD = 'git rev-parse --short HEAD'

source_regex = re.compile(r"^File: (.*?) \(starting at line ([0-9]*?)\)$",
                          re.MULTILINE)


def _get_git_revision():
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()
    except (subprocess.CalledProcessError, OSError):
        print('Failed to execute git to get revision')
        return None
    return revision.decode('utf-8')


def _linkcode_resolve(domain, info, url_fmt, revision):
    """Determine a link to online source for a class/method/function

    This is called by sphinx.ext.linkcode

    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='http://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'http://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """

    if revision is None:
        return
    if domain != 'py':
        return
    if not info.get('module') or not info.get('fullname'):
        return

    class_name = info['fullname'].split('.')[0]
    module = __import__(info['module'], fromlist=[class_name])
    obj = attrgetter(info['fullname'])(module)

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    fn: str = None
    lineno: str = None

    obj_module = inspect.getmodule(obj)
    if not obj_module:
        print(f"could not infer source code link for: {info}")
        return
    module_name = obj_module.__name__.split('.')[0]

    module_dir_dict = {
        "cugraph_dgl": "cugraph-dgl",
        "cugraph_pyg": "cugraph-pyg",
        "cugraph_service_client": "cugraph-service/client",
        "cugraph_service_server": "cugraph-service/server",
        "cugraph": "cugraph",
        "pylibcugraph": "pylibcugraph",
    }
    module_dir = module_dir_dict.get(module_name)
    if not module_dir:
        print(f"no source path directory set for {module_name}")
        return

    obj_path = "/".join(obj_module.__name__.split(".")[1:])
    obj_file_ext = obj_module.__file__.split('.')[-1]
    source_ext = "pyx" if obj_file_ext == "so" else "py"
    fn = f"{module_dir}/{module_name}/{obj_path}.{source_ext}"

    # Get the line number if we need it. (Can work without it)
    if (lineno is None):
        try:
            lineno = inspect.getsourcelines(obj)[1]
        except Exception:

            # Can happen if its a cyfunction. See if it has `__code__`
            if (hasattr(obj, "__code__")):
                lineno = obj.__code__.co_firstlineno
            else:
                lineno = ''
    return url_fmt.format(revision=revision,
                          path=fn,
                          lineno=lineno)


def make_linkcode_resolve(url_fmt):
    """Returns a linkcode_resolve function for the given URL format

    revision is a git commit reference (hash or name)

    url_fmt is along the lines of ('https://github.com/USER/PROJECT/'
                                   'blob/{revision}/{package}/'
                                   '{path}#L{lineno}')
    """
    revision = _get_git_revision()
    return partial(_linkcode_resolve,
                   revision=revision,
                   url_fmt=url_fmt)
