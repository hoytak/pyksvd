from numpy cimport ndarray as ar
from numpy import empty, ascontiguousarray, dot
from scipy.linalg import eigh
import numpy.random as rn
from libcpp.map cimport map
from libcpp.string cimport string

cdef extern from "ksvd.hpp":

    void _KSVDNumpyWrapper(double *Dptr, double *Gammaptr, 
                           double *Xptr, size_t n, size_t d, size_t p,
                           size_t target_sparsity, size_t max_iterations,
                           map[string, double] params) except +

    void _KSVDEncodeNumpyWrapper(double *Gammaptr, double *Dptr,  
                             double *Xptr, size_t n, size_t d, size_t p,
                             size_t target_sparsity,
                             map[string, double] params) except +

    
def KSVD(Xo,
         size_t dict_size,
         size_t target_sparsity,
         size_t max_iterations,
         D_init = "svd",
         bint enable_printing = True,
         size_t print_interval = 25,
         size_t grad_descent_iterations = 2,
         size_t convergence_check_interval = 50,
         double convergence_threshhold = 0,
         bint enable_eigen_threading = False,
         bint enable_threading = True,
         bint enable_32bit_initialization = True,
         size_t max_initial_32bit_iterations = 0
         ):
    """
    KSVD(X,dict_size, target_sparsity,max_iterations,D_init = 'random', enable_printing = False, print_interval = 25, grad_descent_iterations = 2, convergence_check_interval = 50, convergence_threshhold = 0, enable_eigen_threading = False, enable_threading = True, enable_32bit_initialization = True, max_initial_32bit_iterations = 0)
    
    Runs an approximate KSVD algorithm using batch orthogonal matching pursuit.

    :`X`:
      n by p matrix of signals, where `n` is the number of signals and
      p is the dimension of each signal.

    :`dict_size`:
      The size of the target dictionary.

    :`target_sparsity`:
      The target sparsity of the signal.

    :`max_iterations`:
      The maximum number of iterations to perform.  Generally takes 500-5000.
    
    :`D_init`:
      Initialization mode of the dictionary.  If a `dict_size` by `p`
      array is given, then this is used to initialize the dictionary.
      Otherwise, if ``D_init == 'random'``, It is initialized
      randomly.  If ``D_init == 'svd'`` (default), part of the
      dictionary is initialized using a singular value decomposition
      of the signal.  This can give faster convergence in some cases,
      but hits a bad local minimum in others.

    :`enable_printing`:
      If True, prints periodic status messages about the
      convergence. (default = False).

    :`print_interval`:
      How often to print convergence information.

    :`grad_descent_iterations`:
      The number of gradient descent steps used to approximate the
      primary svd vectors at each iteration.  Default = 2.

    :`convergence_check_interval`:
      How often to check convergence.  This step can be expensive (default = 50).

    :`convergence_threshhold`:
      When the approximation accuracy falls below this, the algorithm
      terminates.  Approximation accuracy is measured by
      ``||X - D * Gamma||_2 / n``, where ``n`` is the number of signals.
      If 0 (default), the algorithm runs for ``max_iterations`` or to
      machine epsilon, whichever comes first.

    :`enable_eigen_threading`:
      Whether to enable threading in linear algebra operations in the
      Eigen libraries.  This is recommended only for very large
      problems. (default = False).
    
    :`enable_threading`:
      Whether to enable threading in calculating the sparse
      projections in the Batch OMP step.  Generally, this can give
      substantial speedup. (default = True).
    
    :`enable_32bit_initialization`:
      Whether to process as much as possible using the faster 32bit
      mode.  This is generally recommended, as the accuracy is
      typically good enough for the start of most problems.  Once the
      accuracy falls below what 32bit floats can accurately determine,
      the computation switches to 64bit.
    
    :`max_initial_32bit_iterations`:
      The maximum number of 32 bit iterations to do before switching
      to 64bit mode.  If 0 (default), no limit.

    Returns tuple (D, Gamma), where X \simeq Gamma * D.
    """

    cdef ar[double, ndim=2, mode='c'] X = ascontiguousarray(Xo, dtype='d')

    cdef size_t n = X.shape[0]
    cdef size_t p = X.shape[1]

    cdef ar[double, ndim=2, mode='c'] D = empty( (dict_size, p) )
    cdef ar[double, ndim=2, mode='c'] Gamma = empty( (n, dict_size) )

    if target_sparsity >= dict_size:
        raise ValueError("Target sparsity (%d) >= Dictionary size (%d)"
                         % (target_sparsity, dict_size))

    cdef map[string, double] params

    params["print_interval"] = print_interval
    params["convergence_threshhold"] = convergence_threshhold
    params["convergence_check_interval"] = convergence_check_interval
    params["grad_descent_iterations"] = grad_descent_iterations
    params["enable_eigen_threading"] = enable_eigen_threading
    params["enable_threading"] = enable_threading
    params["enable_printing"] = enable_printing
    params["max_initial_32bit_iterations"] = max_initial_32bit_iterations
    params["enable_32bit_initialization"] = enable_32bit_initialization

    if type(D_init) is str:
        if D_init.lower() == "svd":
            params["initialize_from_svd"] = True
            params["initialize_random"] = True
        elif D_init.lower() == "random":
            params["initialize_from_svd"] = False
            params["initialize_random"] = True
        else:
            raise ValueError("D_Init mode not recognized: '%s' not in ['svd', 'random']."
                             % (D_init.lower()))
    else:
        D[:,:] = D_init
        params["initialize_from_svd"] = False
        params["initialize_random"] = False

    _KSVDNumpyWrapper(&D[0,0], &Gamma[0,0], &X[0,0], n, dict_size, p,
                      target_sparsity, max_iterations,
                      params)
    
    return D, Gamma
    
def KSVD_Encode(Xo, Do, size_t sparsity):
    """
    Encode a signal X of size n by p using a precomputed dictionary D
    of size d by p. and sparsity `sparsity`.  Returns Gamma, the
    encoded basis.
    """

    cdef ar[double, ndim=2, mode='c'] X = ascontiguousarray(Xo, dtype='d')
    cdef ar[double, ndim=2, mode='c'] D = ascontiguousarray(Do, dtype='d')
    
    cdef size_t n = X.shape[0]
    cdef size_t p = X.shape[1]
    cdef size_t dict_size = D.shape[0]

    cdef ar[double, ndim=2, mode='c'] Gamma = empty( (n, dict_size) )
    
    if sparsity >= dict_size:
        raise ValueError("Target sparsity (%d) >= Dictionary size (%d)"
                         % (sparsity, dict_size))

    cdef map[string, double] params

    _KSVDEncodeNumpyWrapper(&Gamma[0,0], &D[0,0], &X[0,0], n, dict_size, p,
                            sparsity,
                            params)
    
    return Gamma
    

