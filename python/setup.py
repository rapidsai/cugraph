# Copyright (c) 2018, NVIDIA CORPORATION.

from distutils.sysconfig import get_python_lib
import os
from os.path import join as pjoin
import sys

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import versioneer


INSTALL_REQUIRES = ['numba', 'cython']


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path
    for directory in path.split(os.pathsep):
        binpath = pjoin(directory, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_nvgraph():
    if 'CONDA_PREFIX' in os.environ:
        nvgraph_found = find_in_path('lib/libnvgraph_st.so',
                                     os.environ['CONDA_PREFIX'])
    if nvgraph_found is None:
        nvgraph_found = find_in_path('libnvgraph_st.so',
                                     os.environ['LD_LIBRARY_PATH'])
        if nvgraph_found is None:
            raise EnvironmentError('The nvgraph library could not be located')
    nvgraph_config = {
        'include': pjoin(os.path.dirname(os.path.dirname(nvgraph_found)),
                         'include', 'nvgraph'),
        'lib': os.path.dirname(nvgraph_found)}

    return nvgraph_config


NVGRAPH = locate_nvgraph()

try:
    NUMPY_INCLUDE = numpy.get_include()
except AttributeError:
    NUMPY_INCLUDE = numpy.get_numpy_include()

CUDF_INCLUDE = os.path.normpath(sys.prefix) + '/include'
CYTHON_FILES = ['cugraph/*.pyx']

EXTENSIONS = [
    Extension("cugraph",
              sources=CYTHON_FILES,
              include_dirs=[NUMPY_INCLUDE,
                            CUDF_INCLUDE,
                            # NVGRAPH['include'],
                            '../cpp/src',
                            '../cpp/include',
                            '../cpp/build/gunrock',
                            '../cpp/build/gunrock/externals/moderngpu/include',
                            '../cpp/build/gunrock/externals/cub'],
              library_dirs=[get_python_lib(), NVGRAPH['lib']],
              libraries=['cugraph', 'cudf', 'nvgraph_st'],
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
      install_requires=INSTALL_REQUIRES,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False)
