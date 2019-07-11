# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import versioneer
from distutils.sysconfig import get_python_lib


INSTALL_REQUIRES = ['numba', 'cython']

conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'
conda_include_dir = os.path.normpath(sys.prefix) + '/include'

CYTHON_FILES = ['cugraph/**/*.pyx']

if (os.environ.get('CONDA_PREFIX', None)):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_include_dir = conda_prefix + '/include'
    conda_lib_dir = conda_prefix + '/lib'

EXTENSIONS = [
    Extension("*",
              sources=CYTHON_FILES,
              include_dirs=[conda_include_dir,
                            '../cpp/include'],
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[conda_lib_dir],
              libraries=['cugraph', 'cudf'],
              language='c++',
              extra_compile_args=['-std=c++14'])
]

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
      ext_modules=cythonize(EXTENSIONS),
      packages=find_packages(include=['cugraph', 'cugraph.*']),
      install_requires=INSTALL_REQUIRES,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False)
