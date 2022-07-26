# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
import shutil

from setuptools import find_packages, Command
from skbuild import setup

from setuputils import get_environment_option
# from setuputils import get_cuda_version_from_header
# FIXME: Not necessary
from skbuild.command.build_ext import build_ext

import versioneer


INSTALL_REQUIRES = ['numba', 'cython']
CYTHON_FILES = ['cugraph/**/*.pyx']

UCX_HOME = get_environment_option("UCX_HOME")
CUDA_HOME = get_environment_option('CUDA_HOME')
CONDA_PREFIX = get_environment_option('CONDA_PREFIX')

# FIXME: No need to include the path to the conda dir
"""
conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'
conda_include_dir = os.path.normpath(sys.prefix) + '/include'

if CONDA_PREFIX:
    conda_include_dir = CONDA_PREFIX + '/include'
    conda_lib_dir = CONDA_PREFIX + '/lib'
"""

if not UCX_HOME:
    UCX_HOME = CONDA_PREFIX if CONDA_PREFIX else os.sys.prefix

# FIXME: add ucx to INSTALL_REQUIRES
ucx_include_dir = os.path.join(UCX_HOME, "include")
ucx_lib_dir = os.path.join(UCX_HOME, "lib")

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
# FIXME: This is causing a second version of cupy to be installed cupy-cuda115
"""
INSTALL_REQUIRES.append(
    "cupy-cuda" + get_cuda_version_from_header(cuda_include_dir)
)
"""


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
        os.system('find . -name "__pycache__" -type d -exec rm -rf {} +')
        os.system('rm -rf *.egg-info')
        os.system('find . -name "*.cpp" -type f -delete')
        os.system('find . -name "*.cpython*.so" -type f -delete')


cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["clean"] = CleanCommand
cmdclass["build_ext"] = build_ext

# FIXME: remove setup_requires, the .toml file does it
setup(name='cugraph',
      description="cuGraph - RAPIDS GPU Graph Analytics",
      version=versioneer.get_version(),
      classifiers=[
          # "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          # "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9"
      ],
      # Include the separately-compiled shared library
      author="NVIDIA Corporation",
      setup_requires=['Cython>=0.29,<0.30'],
      packages=find_packages(include=['cugraph', 'cugraph.*']),
      include_package_data=True,
      package_data={
          '': ['python/cugraph/cugraph/experimental/datasets/metadata/*.yaml',
               'python/cugraph/cugraph/experimental/datasets/*.yaml'],
      },
      install_requires=INSTALL_REQUIRES,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False)
