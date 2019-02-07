# Copyright (c) 2018, NVIDIA CORPORATION.

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import versioneer
from distutils.sysconfig import get_python_lib
from os.path import join as pjoin
import os
import sys

install_requires = [
    'numba',
    'cython'
]

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
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
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

def locate_nvgraph():
    if 'CONDA_PREFIX' in os.environ:
        nvgraph_found = find_in_path('lib/libnvgraph_st.so', os.environ['CONDA_PREFIX'])
    if nvgraph_found is None:
        nvgraph_found = find_in_path('libnvgraph_st.so', os.environ['LD_LIBRARY_PATH'])
        if nvgraph_found is None:
            raise EnvironmentError('The nvgraph library could not be located') 
    nvgraph_config = {'include':pjoin(os.path.dirname(os.path.dirname(nvgraph_found)), 'include', 'nvgraph'),
    'lib':os.path.dirname(nvgraph_found)}
    return nvgraph_config
  
CUDA = locate_cuda()
NVGRAPH = locate_nvgraph()

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

cudf_include = os.path.normpath(sys.prefix) + '/include'
cython_files = ['cugraph/*.pyx']

extensions = [
    Extension("cugraph",
              sources=cython_files,
              include_dirs=[numpy_include,
                            cudf_include,
                            NVGRAPH['include'],
                            CUDA['include'],
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
      ext_modules=cythonize(extensions),
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False
      )
