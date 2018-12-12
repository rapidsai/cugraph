#
# Copyright (c) 2018, NVIDIA CORPORATION.
#
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
#


from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

from distutils.sysconfig import get_python_lib

install_requires = [
    'numpy',
    'cython'
]

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

cython_files = ['python/pagerank/pagerank_wrapper.pyx']

extensions = [
    Extension("cugraph",
              sources=cython_files,
              include_dirs=[numpy_include,
                            '/usr/local/include/gdf',
                            'src',
                            'include',
                            '../gunrock',
                            '../gunrock/externals/moderngpu/include',
                            '../gunrock/externals/cub'],
              library_dirs=[get_python_lib()],
              libraries=['cugraph'],
              language='c++',
              extra_compile_args=['-std=c++11'])
]

setup(name='cugraph',
      description='cuGraph - RAPIDS Graph Analytic Algorithms',
      author='NVIDIA Corporation',
      # todo: Add support for versioneer
      version='0.1',
      ext_modules=cythonize(extensions),
      install_requires=install_requires,
      license="Apache",
      zip_safe=False)
