from setuptools import setup
import os 
os.system("git clone https://github.com/rapidsai/dask-cudf.git ../dask-cudf")
os.system("cd ../dask-cudf && pip install .")

install_requires = ['cugraph','dask_cudf']

setup(
   name='dask_cugraph',
   version='1.0',
   description='',
   author='NVIDIA Corporation',
   packages=['dask_cugraph'],  
   install_requires=install_requires
)
