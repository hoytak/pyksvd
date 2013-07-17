from ksvd import KSVD
import numpy.random as rn
from numpy import array, zeros, dot

if __name__ == "__main__":

    dict_size = 1024
    target_sparsity = 8
    n_examples = 153000
    dimension = 512

    rs = rn.RandomState(0)

    D = rs.normal(size = (dict_size, dimension) )

    M = zeros( (n_examples, dict_size + 1) )

    M[:, :target_sparsity] = rs.normal(size = (n_examples, target_sparsity) )
    
    M = M.ravel()[:n_examples*dict_size].reshape(n_examples, dict_size)

    X = dot(M, D)

    del M
    del D

    print "X generated."

    KSVD(X, dict_size, target_sparsity, 50,
         print_interval = 25,
         enable_printing = True, enable_threading = True)

    

    
