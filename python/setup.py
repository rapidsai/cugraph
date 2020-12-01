# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

import os
import sys
import shutil

from setuptools import setup, find_packages, Command
from setuptools.extension import Extension
from setuputils import use_raft_package, get_environment_option

try:
    from Cython.Distutils.build_ext import new_build_ext as build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext

import versioneer
from distutils.sysconfig import get_python_lib


INSTALL_REQUIRES = ['numba', 'cython']

conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'
conda_include_dir = os.path.normpath(sys.prefix) + '/include'

CYTHON_FILES = ['cugraph/**/*.pyx']

CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

if not os.path.isdir(CUDA_HOME):
    raise OSError(
        "Invalid CUDA_HOME: " "directory does not exist: {CUDA_HOME}"
    )

cuda_include_dir = os.path.join(CUDA_HOME, "include")

if (os.environ.get('CONDA_PREFIX', None)):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_include_dir = conda_prefix + '/include'
    conda_lib_dir = conda_prefix + '/lib'

# Optional location of C++ build folder that can be configured by the user
libcugraph_path = get_environment_option('CUGRAPH_BUILD_PATH')
# Optional location of RAFT that can be confugred by the user
raft_path = get_environment_option('RAFT_PATH')

# FIXME: This could clone RAFT, even if it's not needed (eg. running --clean).
raft_include_dir = use_raft_package(raft_path, libcugraph_path,
                                    git_info_file='../cpp/CMakeLists.txt')


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = [('all', None, None), ]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        setupFileDir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(setupFileDir)
        os.system('rm -rf build')
        os.system('rm -rf dist')
        os.system('rm -rf dask-worker-space')
        os.system('rm -f cugraph/raft')
        os.system('find . -name "__pycache__" -type d -exec rm -rf {} +')
        os.system('rm -rf *.egg-info')
        os.system('find . -name "*.cpp" -type f -delete')
        os.system('find . -name "*.cpython*.so" -type f -delete')


cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext
cmdclass["clean"] = CleanCommand

EXTENSIONS = [
    Extension("*",
              sources=CYTHON_FILES,
              include_dirs=[conda_include_dir,
                            '../cpp/include',
                            "../thirdparty/cub",
                            raft_include_dir,
                            os.path.join(
                                conda_include_dir, "libcudf", "libcudacxx"),
                            cuda_include_dir],
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[conda_lib_dir],
              libraries=['cugraph', 'cudf', 'nccl'],
              language='c++',
              extra_compile_args=['-std=c++14'])
]

for e in EXTENSIONS:
    e.cython_directives = dict(
        profile=False, language_level=3, embedsignature=True
    )

setup(name='cugraph',
      description="cuGraph - GPU Graph Analytics",
      version=versioneer.get_version(),
      classifiers=[
          # "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          # "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"
      ],
      # Include the separately-compiled shared library
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=EXTENSIONS,
      packages=find_packages(include=['cugraph', 'cugraph.*']),
      install_requires=INSTALL_REQUIRES,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False)
