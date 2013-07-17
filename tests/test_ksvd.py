from ksvd import KSVD
import numpy.random as rn
from numpy import array, zeros, dot

if __name__ == "__main__":

    D = array([ [1,0,1,0],
                [-1,0,0, 1],
                [1,1,1, 2] ])

    rs = rn.RandomState(0)

    p = D.shape[1]
    d = D.shape[0]
    n = 10

    M = zeros( (n, D.shape[0] + 1) )

    M[:, :2] = rs.normal(size = (n, 2) )
    
    M = M.ravel()[:n*D.shape[0]].reshape(n, D.shape[0])

    X = dot(M, D)

    KSVD(X, 3, 2, 500,
         print_interval = 1)
    

    
