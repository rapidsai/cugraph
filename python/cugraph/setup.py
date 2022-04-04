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
import sys
import sysconfig
import shutil

# Must import in this order:
#   setuptools -> Cython.Distutils.build_ext -> setuptools.command.build_ext
# Otherwise, setuptools.command.build_ext ends up inheriting from
# Cython.Distutils.old_build_ext which we do not want
import setuptools

try:
    from Cython.Distutils.build_ext import new_build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext

from distutils.sysconfig import get_python_lib

import setuptools.command.build_ext
from setuptools import find_packages, setup, Command
from setuptools.extension import Extension

from setuputils import get_environment_option

import versioneer


INSTALL_REQUIRES = ['numba', 'cython']
CYTHON_FILES = ['cugraph/**/*.pyx']

UCX_HOME = get_environment_option("UCX_HOME")
CUDA_HOME = get_environment_option('CUDA_HOME')
CONDA_PREFIX = get_environment_option('CONDA_PREFIX')

conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'
conda_include_dir = os.path.normpath(sys.prefix) + '/include'

if CONDA_PREFIX:
    conda_include_dir = CONDA_PREFIX + '/include'
    conda_lib_dir = CONDA_PREFIX + '/lib'

if not UCX_HOME:
    UCX_HOME = CONDA_PREFIX if CONDA_PREFIX else os.sys.prefix

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
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")

# Optional location of C++ build folder that can be configured by the user
libcugraph_path = get_environment_option('CUGRAPH_BUILD_PATH')

if not libcugraph_path:
    libcugraph_path = conda_lib_dir

extensions = [
    Extension("*",
              sources=CYTHON_FILES,
              include_dirs=[
                  conda_include_dir,
                  ucx_include_dir,
                  '../cpp/include',
                  "../thirdparty/cub",
                  os.path.join(conda_include_dir, "libcudacxx"),
                  cuda_include_dir,
                  os.path.dirname(sysconfig.get_path("include"))
              ],
              library_dirs=[
                  get_python_lib(),
                  conda_lib_dir,
                  libcugraph_path,
                  ucx_lib_dir,
                  cuda_lib_dir,
                  os.path.join(os.sys.prefix, "lib")
              ],
              libraries=['cudart', 'cusparse', 'cusolver', 'cugraph', 'nccl'],
              language='c++',
              extra_compile_args=['-std=c++17'])
]


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


class build_ext_no_debug(_build_ext):

    def build_extensions(self):
        def remove_flags(compiler, *flags):
            for flag in flags:
                try:
                    compiler.compiler_so = list(
                        filter((flag).__ne__, compiler.compiler_so)
                    )
                except Exception:
                    pass
        # Full optimization
        self.compiler.compiler_so.append("-O3")
        # No debug symbols, full optimization, no '-Wstrict-prototypes' warning
        remove_flags(
            self.compiler, "-g", "-G", "-O1", "-O2", "-Wstrict-prototypes"
        )
        super().build_extensions()

    def finalize_options(self):
        if self.distribution.ext_modules:
            # Delay import this to allow for Cython-less installs
            from Cython.Build.Dependencies import cythonize

            nthreads = getattr(self, "parallel", None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                nthreads=nthreads,
                force=self.force,
                gdb_debug=False,
                compiler_directives=dict(
                    profile=False, language_level=3, embedsignature=True
                ),
            )
        # Skip calling super() and jump straight to setuptools
        setuptools.command.build_ext.build_ext.finalize_options(self)


cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext_no_debug
cmdclass["clean"] = CleanCommand

setup(name='cugraph',
      description="cuGraph - RAPIDS GPU Graph Analytics",
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
      setup_requires=['Cython>=0.29,<0.30'],
      ext_modules=extensions,
      packages=find_packages(include=['cugraph', 'cugraph.*']),
      install_requires=INSTALL_REQUIRES,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False)
