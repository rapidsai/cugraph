from os.path import join as pjoin
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import sys

from distutils.sysconfig import get_python_lib

install_requires = [
    'numpy',
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
    print(nvgraph_config['include'], nvgraph_config['lib'])
    return nvgraph_config
  
CUDA = locate_cuda()
NVGRAPH = locate_nvgraph()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# temporary fix. cudf 0.5 will have a cudf.get_include()
cudf_include = os.path.normpath(sys.prefix) + '/include'
 
cython_files = ['python/cugraph.pyx']

extensions = [
    Extension("cugraph",
              sources=cython_files,
              include_dirs=[numpy_include,
                            cudf_include,
                            NVGRAPH['include'],
                            CUDA['include'],
                            'src',
                            'include',
                            '../gunrock',
                            '../gunrock/externals/moderngpu/include',
                            '../gunrock/externals/cub'],
              library_dirs=[get_python_lib(), NVGRAPH['lib']],
              libraries=['cugraph', 'cudf', 'nvgraph_st'],
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



