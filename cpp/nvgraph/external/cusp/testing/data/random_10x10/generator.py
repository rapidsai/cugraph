from scipy.sparse import coo_matrix
from scipy.io import mmwrite
from numpy.random import permutation
M = N = 10

for nnz in [0, 1, 2, 5, 8, 10, 15, 20, 30, 50, 80, 100]:
    P = permutation(M * N)[:nnz]
    I = P / N
    J = P % N
    V = permutation(nnz) + 1

    A = coo_matrix( (V,(I,J)) , shape=(M,N))
    filename = '%03d_nonzeros.mtx' % (nnz,)
    mmwrite(filename, A)

