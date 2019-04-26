# Copyright (c) 2019, NVIDIA CORPORATION.
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
import subprocess
import shutil
import sys
import sysconfig
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
from cmake_setuptools import CMakeExtension, CMakeBuildExt, distutils_dir_name, \
    convert_to_manylinux, InstallHeaders

from distutils.sysconfig import get_python_lib

# setup does not clean up the build directory, so do it manually
shutil.rmtree('build', ignore_errors=True)

cuda_version = ''.join(os.environ.get('CUDA', 'unknown').split('.')[:2])

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# temporary fix. cugraph 0.5 will have a cugraph.get_include()
cudf_include = os.path.normpath(sys.prefix) + '/include'


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path
    for directory in path.split(os.pathsep):
        binpath = os.path.join(directory, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError(
                'The nvcc binary could not be located in your $PATH. '
                'Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64': os.path.join(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError(
                'The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


cuda_include = locate_cuda()['include']

cython_files = ['python/cugraph/*.pyx']

extensions = [
    CMakeExtension('cugraph', sourcedir='cpp'),
    Extension("cugraph",
              sources=cython_files,
              include_dirs=[numpy_include,
                            cudf_include,
                            cuda_include,
                            'cpp/src',
                            'cpp/include',
                            '../gunrock',
                            '../gunrock/externals/moderngpu/include',
                            '../gunrock/externals/cub'],
              library_dirs=[get_python_lib(), distutils_dir_name('lib')],
              libraries=['nvgraph'],
              language='c++',
              extra_compile_args=['-std=c++11'])
]

install_requires = [
    'numpy',
    'cython'
]

name = 'cugraph-cuda{}'.format(cuda_version)
version = os.environ.get('GIT_DESCRIBE_TAG', '0.0.0.dev0').lstrip('v')
setup(name=name,
      version=version,
      description='cuGraph - RAPIDS Graph Analytic Algorithms',
      long_description=open('README.md', encoding='UTF-8').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/rapidsai/cugraph',
      author='NVIDIA Corporation',
      classifiers=[
          "Intended Audience :: Developers",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"
      ],
      packages=find_packages(where='python'),
      package_dir={
          'cugraph': 'python/cugraph'
      },
      ext_modules=cythonize(extensions),
      install_requires=install_requires,
      license="Apache",
      cmdclass={
          'build_ext': CMakeBuildExt,
          'install_headers': InstallHeaders
      },
      headers=['cpp/include'],
      zip_safe=False)

convert_to_manylinux(name, version)
