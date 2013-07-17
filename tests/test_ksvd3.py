from ksvd import KSVD
import numpy.random as rn
from numpy import array, zeros, dot, load

if __name__ == "__main__":

    dict_size = 250
    target_sparsity = 50
    n_examples = 500
    dimension = 96

    X = load('y1dump.npy').T

    KSVD(X, dict_size, target_sparsity, 500,
         print_interval = 25,
         D_init = "svd")


    

    

    

    
