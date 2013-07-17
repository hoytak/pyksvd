from ksvd import KSVD
import numpy.random as rn
from numpy import array, zeros, dot

if __name__ == "__main__":

    factor = 2

    dict_size = 5
    target_sparsity = 3
    n_examples = 10
    dimension = 4

    rs = rn.RandomState(0)

    D = rs.normal(size = (dict_size, dimension) )

    M = zeros( (n_examples, dict_size + 1) )

    M[:, :target_sparsity] = rs.normal(size = (n_examples, target_sparsity) )
    
    M = M.ravel()[:n_examples*dict_size].reshape(n_examples, dict_size)

    X = dot(M, D)

    KSVD(X, dict_size, target_sparsity, 1000)

    

    
