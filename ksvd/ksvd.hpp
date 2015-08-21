#ifndef _KSVD_H_
#define _KSVD_H_

// Implements the K-SVD algorithm with batch omp.  Reentrant.

#include <exception>
#include <vector>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <map>
#include <set>
#include <omp.h>

#include "debug.hpp"
#include "../third-party/eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;

typedef map<string, double> param_t;

static const param_t default_ksvd_parameters = {
  {"print_interval", 25},
  {"convergence_check_interval", 50},
  {"convergence_threshhold", 0},
  {"grad_descent_iterations", 2},
  {"enable_threading", true},
  {"enable_printing", true},
  {"print_only_convergence_value", false},
  {"enable_32bit_initialization", true},
  {"max_initial_32bit_iterations", 0},
  {"order_shuffle_interval", 0},
  {"parallel_descent_num_signal_threshold", 10000},
  {"Initialize_from_svd", true},
  {"Initialize_random", true},
  {"random_seed", 0},
  {"first_scaleup_iterations", 0.25},
  {"second_scaleup_iterations", 0.25},
  {"first_scaleup_percent", 0.05},
  {"second_scaleup_percent", 0.2},
  {"internal_Xmap_switch_interval", 0.5}
};

static const param_t default_encoding_parameters = {
  {"print_interval", 1000},
  {"enable_threading", true},
  {"enable_printing", false},
  {"signal_index_offset", 0},
  {"total_signals", 0}
};

template <typename real_t>
struct BatchOMPData {

  typedef Matrix<real_t, Dynamic, 1> Vector_t;
  typedef Matrix<real_t, Dynamic, Dynamic, RowMajor> Matrix_t;

  BatchOMPData(size_t dict_size, size_t target_sparsity, size_t dimension)
    : Lfull(target_sparsity + 1, target_sparsity + 1)
    , L(target_sparsity + 1, target_sparsity + 1)
    , G_reorder(dict_size, dict_size)
    , alpha(dict_size)
    , w(target_sparsity)
    , alpha0_I(target_sparsity)
  {
#ifndef NO_UNROLL_BATCHOMP_SOLVER
    L1(0,0) = 1;
    L2.setZero();
    L3.setZero();
    L4.setZero();
    // L5.setZero();
    // L6.setZero();
    // L7.setZero();
    // L8.setZero();
#endif
  }

  Matrix_t Lfull;
  Matrix_t L;
  Matrix_t G_reorder;

  Vector_t alpha;
  Vector_t w;
  Vector_t alpha0_I;

  // Specific Ls
#ifndef NO_UNROLL_BATCHOMP_SOLVER
  Matrix<real_t, 0, 0, RowMajor> L0;
  Matrix<real_t, 1, 1, RowMajor> L1;
  Matrix<real_t, 2, 2, RowMajor> L2;
  Matrix<real_t, 3, 3, RowMajor> L3;
  Matrix<real_t, 4, 4, RowMajor> L4;
  // Matrix<real_t, 5, 5, RowMajor> L5;
  // Matrix<real_t, 6, 6, RowMajor> L6;
  // Matrix<real_t, 7, 7, RowMajor> L7;
  // Matrix<real_t, 8, 8, RowMajor> L8;
#endif
};

template <typename real_t> struct EpsChooser {};
template <> struct EpsChooser<double> {
  static constexpr double eps = 1e-12;
};

template <> struct EpsChooser<float> {
  static constexpr float eps = 1e-6;
};

////////////////////////////////////////////////////////////////////////////////
// Batch OMP functions

template < typename real_t,
	   typename GammaVector, 
	   typename AlphaVect> 
struct _BatchOMPIteration {
  real_t& global_max;
  vector<size_t>& indices;
  GammaVector& gamma;
  Matrix<real_t, Dynamic, 1>& alpha;
  const AlphaVect& alpha0;
  Matrix<real_t, Dynamic, 1>& alpha0_I;
  const Matrix<real_t, Dynamic, Dynamic, RowMajor>& G;
  Matrix<real_t, Dynamic, Dynamic, RowMajor>& G_reorder;
  const size_t& target_sparsity;

  template <int iteration>
  inline bool run(Matrix<real_t, iteration, iteration, RowMajor>& L_last,
		  Matrix<real_t, iteration + 1, iteration + 1, RowMajor>& L_this)
  {
    const real_t eps = EpsChooser<real_t>::eps;
    
    // Get the max coefficient.
    size_t k = 0;
    real_t max_value = 0;

    for(size_t i = 0; i < size_t(alpha.size()); ++i) {
      real_t v = abs(alpha[i]);
      if(v > max_value) { max_value = v; k = i; }
    }

    if(iteration == 0)
      global_max = max_value;

    if(max_value <= 1e-6 * global_max)
      return true;

    if(iteration >= 1) {
      
      Matrix<real_t, iteration, 1> w;
  
      for(size_t i = 0; i < iteration; ++i)
	w[i] = G(indices[i], k);

      L_last.template triangularView<Lower>().solveInPlace(w);
	
      L_this.row(iteration).template head<iteration>() = w;
      
      double w_norm = w.squaredNorm();

      assert_leq(w_norm, 1.0 + eps);
      
      while(w_norm >= 1.0 - eps) {
	w *= (1.0 - eps);
	w_norm *= (1.0 - eps)*(1.0 - eps);
      }
      
      L_this(iteration, iteration) = sqrt(1 - w_norm);
      L_this.template topLeftCorner<iteration, iteration>() = L_last;
    }
    
    indices.push_back(k);

    G_reorder.row(iteration) = G.row(k);
    alpha0_I[iteration] = alpha0[k];
				      
    gamma.template head<iteration + 1>() = alpha0_I.template head<iteration + 1>();

    // Solve L L' g = alpha0_I
    if(iteration != 0) { // L = [1] if iteration = 0
      L_this.template triangularView<Lower>()
	.solveInPlace(gamma.template head<iteration + 1>());

      L_this.transpose().template triangularView<Upper>()
	.solveInPlace(gamma.template head<iteration + 1>());
    }
  
    if(iteration == target_sparsity - 1)
      return true;
    
    alpha = alpha0 - (G_reorder.template topRows<iteration + 1>().transpose() 
		      * gamma.template head<iteration + 1>()); 

    return false;
  }

};

template <typename real_t, typename DestVector, typename AlphaVect, typename MatrixType>
static void _BatchOMP(DestVector& gamma, 
	       vector<size_t>& indices,
	       const AlphaVect& alpha0, 
	       const MatrixType& G, 
	       size_t target_sparsity,
	       BatchOMPData<real_t>& data)
{
  static_assert(is_same<decltype(G.sum()), real_t>::value, "Type Mismatch in G.");
  static_assert(is_same<decltype(gamma.sum()), real_t>::value, "Type Mismatch in gamma.");
  static_assert(is_same<decltype(alpha0.sum()), real_t>::value, "Type Mismatch in alpha0.");

  const real_t eps = EpsChooser<real_t>::eps;
  assert((1.0 - eps) != 1.0);

  typedef Matrix<real_t, Dynamic, 1> Vector_t;
  typedef Matrix<real_t, Dynamic, Dynamic, RowMajor> Matrix_t;

  indices.clear();

  real_t global_max = 0;

  Matrix_t& G_reorder = data.G_reorder;

  Vector_t& alpha = data.alpha;
  Vector_t& alpha0_I = data.alpha0_I;

  alpha = alpha0;

  // Now, do what we can with fixed sized matrices.  makes some things
  // go much faster.

  Vector_t& w = data.w;
  Matrix_t& Lfull = data.Lfull;
  Matrix_t& L = data.L;

#ifndef NO_UNROLL_BATCHOMP_SOLVER
  _BatchOMPIteration<real_t, DestVector, AlphaVect>
    _brun = {global_max, indices, gamma, alpha, alpha0, alpha0_I, G, G_reorder, target_sparsity};

  const size_t start_iter = 4;

  if(_brun.run(data.L0, data.L1)) return;
  if(_brun.run(data.L1, data.L2)) return;
  if(_brun.run(data.L2, data.L3)) return;
  if(_brun.run(data.L3, data.L4)) return;
  // if(_brun.run(data.L4, data.L5)) return;
  // if(_brun.run(data.L5, data.L6)) return;
  // if(_brun.run(data.L6, data.L7)) return;
  // if(_brun.run(data.L7, data.L8)) return;

  // Continue where we left off
  Lfull.setZero();
  Lfull.template topLeftCorner<4, 4>() = data.L4; 
  L = data.L4;

#else
  const size_t start_iter = 0;

  Lfull.setZero();
  Lfull(0,0) = 1;
  L.resize(1,1);
  L(0,0) = 1;

#endif

  for(size_t iteration = start_iter; iteration < target_sparsity; ++iteration) {
    
    // Get the max coefficient.
    size_t k = 0;
    real_t max_value = 0;

    for(size_t i = 0; i < size_t(alpha.size()); ++i) {
      real_t v = abs(alpha[i]);
      if(v > max_value) { max_value = v; k = i; }
    }

    if(iteration == 0)
      global_max = max_value;

    if(max_value <= 1e-6 * global_max)
      break;

    if(iteration >= 1) {
  
      for(size_t i = 0; i < indices.size(); ++i) 
	w[i] = G(indices[i], k);

      L.template triangularView<Lower>().solveInPlace(w.head(iteration));
	
      Lfull.row(iteration).head(iteration) = w.head(iteration);
      
      double w_norm = w.head(iteration).squaredNorm();

      assert_leq(w_norm, 1.0 + eps);

      while(w_norm >= 1.0 - eps) {
	w.head(iteration) *= (1.0 - eps);
	w_norm = w.head(iteration).squaredNorm();
      }

      Lfull(iteration, iteration) = sqrt(1 - w_norm);
      
      L = Lfull.topLeftCorner(iteration + 1, iteration + 1);
    }

    indices.push_back(k);

    G_reorder.row(iteration) = G.row(k);
    alpha0_I[iteration] = alpha0[k];
    
    gamma.head(iteration + 1) = alpha0_I.head(iteration + 1);

    // Solve L L' g = alpha0_I
    L.template triangularView<Lower>().solveInPlace(gamma.head(iteration + 1));
    L.transpose().template triangularView<Upper>().solveInPlace(gamma.head(iteration + 1));
    
    if(iteration == target_sparsity - 1)
      break;

    alpha = alpha0 - G_reorder.topRows(iteration + 1).transpose() * gamma.head(iteration + 1); 
  }
}


////////////////////////////////////////////////////////////////////////////////
// Interface to the batch omp stuff

template <typename MatrixType1, typename IntMatrixType, typename MatrixType2, typename MatrixType3> 
void OMPEncodeSignal(MatrixType1& Gamma,
                     IntMatrixType& Indices,
                     const MatrixType2& D,
                     const MatrixType3& X,
                     size_t sparsity,
                     const param_t& _options) {

  Eigen::setNbThreads(1);

  param_t options(default_encoding_parameters);

  for(auto p : _options) 
    options[p.first] = p.second;
  
  bool enable_threading = (options["enable_threading"] != 0);
  size_t print_interval = size_t(options["print_interval"]);
  size_t enable_printing = (options["enable_printing"] != 0);
  size_t signal_index_offset = size_t(options["signal_index_offset"]);
  size_t total_signals = max(size_t(X.rows()), size_t(options["total_signals"]));

  typedef decltype(D.sum()) real_t;
  typedef Matrix<real_t, Dynamic, 1> Vector_t;
  typedef Matrix<real_t, Dynamic, Dynamic, RowMajor> Matrix_t;

  static_assert(is_same<real_t, decltype(X.sum())>::value,
		"Matrices X and D must have the same type.");
  static_assert(is_same<real_t, decltype(Gamma.sum())>::value,
		"Matrices Gamma and D must have the same type.");

  static_assert(is_convertible<size_t, decltype(Indices.sum())>::value,
		"Indices matrix must be able to hold values of type size_t.");

  size_t dict_size = D.rows();
  size_t dimension = D.cols();
  size_t n = X.rows();

  bool sparse_mode;

  if(size_t(Gamma.rows()) != n)
    throw invalid_argument("Number of rows in Gamma much match number of rows in X.");

  if(Indices.size() == 0) {
    sparse_mode = false;

    if(size_t(Gamma.cols()) != dict_size)
      throw invalid_argument("Number of columns in Gamma much match number of columns in D.");

  } else {
    
    sparse_mode = true;

    if(size_t(Gamma.cols()) != sparsity)
      throw invalid_argument("Number of columns in Gamma must match the target sparsity.");

    if(size_t(Indices.rows()) != n)
      throw invalid_argument("Number of columns in Indices must match the number of rows in X.");

    if(size_t(Indices.cols()) != sparsity)
      throw invalid_argument("Number of columns in Indices must match the target sparsity.");
  }

  if(size_t(X.cols()) != dimension)
    throw invalid_argument("Number of columns in X much match number of columns in D.");

  const int n_threads = enable_threading ? omp_get_max_threads() : 1;

  vector<BatchOMPData<real_t> > bompdata(n_threads, {dict_size, sparsity, dimension});
  
  vector<vector<size_t> > indices(n_threads); 

  vector<Vector_t> gamma_v(n_threads);
  for(Vector_t& v : gamma_v) v.resize(sparsity);

  vector<Vector_t> alpha_v(n_threads);
  for(Vector_t& v : alpha_v) v.resize(dict_size);

  // X is n by p.  D is d by p  
  Matrix_t G = D * D.transpose();

  if(!sparse_mode)
    Gamma.setZero();

  size_t n_finished = 0;

#pragma omp parallel for if(enable_threading)
  for(size_t i = 0; i < n; ++i) {
    const int thread_num = enable_threading ? omp_get_thread_num() : 0;

    Vector_t& gamma = gamma_v[thread_num];
    vector<size_t>& idx_v = indices[thread_num];
    Vector_t& alpha = alpha_v[thread_num];
    
    alpha = X.row(i) * D.transpose();

    _BatchOMP(gamma, idx_v, alpha, G, sparsity, bompdata[thread_num]);
    
    if(sparse_mode) {
      size_t j = 0;
      for(;j < idx_v.size(); ++j) {
	Indices(i,j) = idx_v[j];
	Gamma(i,j) = gamma[j];
      }

      for(;j < sparsity; ++j) {
	Indices(i,j) = dict_size;
	Gamma(i,j) = 0;
      }	
    } else {
      for(size_t j = 0;j < idx_v.size(); ++j) {
	Gamma(i,idx_v[j]) = gamma[j];
      }
    }

#pragma omp atomic
    ++n_finished;

    if(enable_printing && (n_finished + 1 + signal_index_offset) % print_interval == 0) {
#pragma omp critical 
      {
	cout << "Encoded " 
	     << (n_finished + 1 + signal_index_offset) << "/" << total_signals 
	     << " signals." << endl;
      }
    }
  }
}

template <typename MatrixType1, typename MatrixType2> 
void OMPEncodeSignal(MatrixType1& Gamma,
                     const MatrixType1& D,
                     const MatrixType2& X,
                     size_t sparsity,
                     const param_t& options) {

  Matrix<size_t, Dynamic, Dynamic> int_dummy(0,0);
  OMPEncodeSignal(Gamma, int_dummy, D, X, sparsity, options);
}

////////////////////////////////////////////////////////////////////////////////
// KSVD stuff

template <typename real_t, typename MatrixType> 
static size_t _KSVD_internal(MatrixType& D, 
		      MatrixType& Gamma, 
		      const MatrixType& X, 
		      size_t target_sparsity, 
		      size_t max_iterations, 
		      const param_t& _options, 
		      bool disable_convergence_message = false,
		      size_t iteration_start_number = 0)
{
  Eigen::setNbThreads(1);

  param_t options(default_ksvd_parameters);

  for(auto p : _options) 
    options[p.first] = p.second;

  if(DEBUG_MODE) {
    cout << "options" << endl;
    for(auto p : options) 
      cout << p.first << ": " << p.second << endl;
  }

  size_t print_interval          = size_t(options["print_interval"]);
  size_t grad_descent_iterations = size_t(options["grad_descent_iterations"]);
  bool enable_threading          = (options["enable_threading"] != 0);
  bool enable_printing           = (options["enable_printing"] != 0);
  size_t convergence_check_interval = size_t(options["convergence_check_interval"]);
  real_t convergence_threshhold  = options["convergence_threshhold"];
  size_t order_shuffle_interval = size_t(options["order_shuffle_interval"]);
  bool print_only_convergence_value = (options["print_only_convergence_value"] != 0);

  size_t parallel_descent_num_signal_threshold = 
    size_t(options["parallel_descent_num_signal_threshold"]);

  bool initialize_from_svd = (options["initialize_from_svd"] != 0);
  bool initialize_random = (options["initialize_random"] != 0);
  size_t random_seed = size_t(options["random_seed"]);

  bool parallel_optimize_descent_code = false;
    // (parallel_descent_num_signal_threshold != 0
    //  && (size_t(X.rows()) >= parallel_descent_num_signal_threshold)
    //  && enable_threading);

  double first_scaleup_iterations = options["first_scaleup_iterations"];
  double second_scaleup_iterations = options["second_scaleup_iterations"];
  double first_scaleup_percent = options["first_scaleup_percent"];
  double second_scaleup_percent = options["second_scaleup_percent"];

  double internal_Xmap_switch_interval = options["internal_Xmap_switch_interval"];

  const real_t eps = EpsChooser<real_t>::eps;

  if(DEBUG_MODE)
    enable_threading = false;

  // p is the dimensionality of the signals in X
  // n is the number of observations, so X is n by p.
  // d is the dictionary size (L in the paper), so D is d by p
  // Thus Gamma, the reconstructed signal, must be of dimension n by d

  // Returns D and sparse Gamma such that X simeq Gamma D.

  typedef Matrix<real_t, Dynamic, 1> Vector_t;
  typedef Matrix<real_t, Dynamic, Dynamic, RowMajor> Matrix_t;

  const size_t dict_size = D.rows();
  const size_t dimension = D.cols();
  const size_t n = X.rows();

  const size_t n_xi = size_t(n * max(0.0, min(1.0, internal_Xmap_switch_interval)));

  // const size_t n_first_scaleup = min(n, max(size_t(n * first_scaleup_percent), 2*dimension));
  // const size_t n_second_scaleup = min(n, max(size_t(n * second_scaleup_percent), 2*dimension));

  // size_t iter_first_scaleup = size_t(max_iterations * first_scaleup_iterations);
  // size_t iter_second_scaleup = iter_first_scaleup + size_t(max_iterations * second_scaleup_iterations);

  cout << "grad_descent_iterations = " << grad_descent_iterations << endl;

  if(size_t(Gamma.rows()) != n)
    throw invalid_argument("Number of rows in Gamma much match number of rows in X.");

  if(size_t(Gamma.cols()) != dict_size)
    throw invalid_argument("Number of columns in Gamma much match number of columns in D.");

  if(size_t(X.cols()) != dimension)
    throw invalid_argument("Number of columns in X much match number of columns in D.");
    
  ranlux48 rng;
  rng.seed(random_seed);

  vector<size_t> idx_order(dict_size);
  for(size_t i = 0; i < dict_size; ++i) idx_order[i] = i;
  
  const int n_threads = enable_threading ? omp_get_max_threads() : 1;
  const int n_chunks = enable_threading ? 8*n_threads : 1;
  const size_t X_block_msize = (n / n_chunks) + ((n % n_chunks == 0) ? 0 : 1);

  vector<vector<size_t> > X_indices(n_threads);
  vector<vector<size_t> > indices_by_dict(dict_size);

  Matrix_t Gamma_I(n, dict_size);
  Matrix_t G(dict_size, dict_size);
  Matrix_t XI(n, dimension);

  Vector_t g(n);
  Vector_t gfull(n);
  Vector_t dv(dimension);

  Vector_t g_t1_full(n);
  Vector_t g_t2(n);

  vector<BatchOMPData<real_t> > bompdata(n_threads, {dict_size, target_sparsity, dimension});

  vector<Vector_t> gamma_v(n_threads);
  for(Vector_t& v : gamma_v) v.resize(target_sparsity);

  vector<Vector_t> alpha_v(n_threads);
  for(Vector_t& v : alpha_v) v.resize(dict_size);

  Matrix_t DV1(parallel_optimize_descent_code ? n_chunks : 0, dimension);
  Matrix_t DV2(parallel_optimize_descent_code ? n_chunks : 0, dict_size);
  Vector_t Ddv(parallel_optimize_descent_code ? dimension : 0);

  ////////////////////////////////////////////////////////////////////////////////
  // Initialize the svd stuff

  if(initialize_from_svd) {
    D.setZero();
    
    if(enable_threading)
      Eigen::setNbThreads(n_threads);

    Matrix_t XtX = X.transpose() * X;

    SelfAdjointEigenSolver<Matrix_t> eigh(XtX);

    D.topRows(min(dimension, dict_size)) = 
      eigh.eigenvectors().transpose().topRows(min(dimension, dict_size));

    if(enable_threading)
      Eigen::setNbThreads(1);
  }

  if(initialize_random) {
    if(!initialize_from_svd)
      D.setZero();

    std::normal_distribution<> ndist(0,1);

    for(size_t i = 0; i < dict_size; ++i)
      for(size_t j = 0; j < dimension; ++j)
	D(i,j) += 0.01 * ndist(rng);
  }

  // Normalize the incoming D matrix
  for(size_t i = 0; i < dict_size; ++i) {
    real_t norm = D.row(i).norm();

    if(norm < 1e-8) {
      std::normal_distribution<> ndist(0,1);
      for(size_t j = 0; j < dimension; ++j)
	D(i,j) = ndist(rng);
      
      D.row(i) /= D.row(i).norm();
    } else {
      D.row(i) /= norm;
    }
  }

  if(enable_printing) {
    cout << "Initialized. " << endl; 
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Begin the iterations

  size_t iteration = iteration_start_number;
  for(; iteration < max_iterations; ++iteration) {
    
    // Compute the minimum projection of this using the batch
    // orthogonal matching pursuit.
      
    // X is n by p.  D is d by p  
    G = D * D.transpose();
    Gamma.setZero();

    // cout << "D = \n" << D << endl;
    // // cout << "G = \n" << G << endl;
    // cout << "Gamma = \n" << Gamma << endl;
    // cout << "Alpha = \n" << Alpha << endl;

    // Now look at how well we are doing.
    if(enable_printing && DEBUG_MODE && print_interval != 0 && ((iteration + 1) % print_interval) == 0) {
      if(enable_threading)
	Eigen::setNbThreads(n_threads);

      cout << '\n' << (iteration + 1) << ".A\t Approximation Error (average L2-norm of error over signals) = " 
	   << (X - Gamma * D).norm() / (X.rows())
	   << flush;

      if(enable_threading)
	Eigen::setNbThreads(1);
    }

    for(auto& v : indices_by_dict) v.clear();

    size_t n_enabled = n;

    // if(iteration < iter_first_scaleup) {
    //   n_enabled = n_first_scaleup;
    // } else if(iteration < iter_second_scaleup) {
    //   n_enabled = n_second_scaleup;
    // }

    assert_leq(n_enabled, n);

#pragma omp parallel for if(enable_threading) shared(X_indices, X, D, \
						     gamma_v, alpha_v, Gamma, bompdata)
    for(size_t i = 0; i < n_enabled; ++i) {
      const int thread_num = enable_threading ? omp_get_thread_num() : 0;

      vector<size_t>& indices = X_indices[thread_num];

      Vector_t& gamma = gamma_v[thread_num];

      Vector_t& alpha = alpha_v[thread_num];
      alpha = X.row(i) * D.transpose();

      _BatchOMP(gamma, indices, alpha, G, target_sparsity, bompdata[thread_num]);

      // Gamma.row(i).setZero();
      for(size_t j = 0; j < indices.size(); ++j)
	Gamma(i, indices[j]) = gamma[j];

#pragma omp critical
      {
	for(size_t idx : indices)
	  indices_by_dict[idx].push_back(i);
      }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Test the convergence if appropriate

    real_t approx_error = 1e10;
    bool approx_error_calculated = false;

    auto calc_approx_error = [&,X,Gamma,D,n_enabled, enable_threading](){
      if(enable_threading) Eigen::setNbThreads(n_threads);
      real_t approx_error = (X.topRows(n_enabled) - 
			     Gamma.topRows(n_enabled) * D).norm() / (n_enabled);
      if(enable_threading) Eigen::setNbThreads(1);
      return approx_error;
    };

    // Now look at how well we are doing.
    if(print_interval != 0 && ((iteration + 1) % print_interval) == 0) {
      approx_error = calc_approx_error();
      approx_error_calculated = true;

      if(enable_printing) {
#pragma omp critical 
	{
          if(print_only_convergence_value)
            cout << (iteration + 1) << "\t" << approx_error << endl;
	  else 
            cout << '\n' << (iteration + 1) << ".\t Approximation Error (average L2-norm of error over signals) = "
                 << approx_error << ". "
                 << flush;
	}
      }

      if(iteration >= 25 && approx_error < eps) {
#pragma omp critical 
	{
	  if(enable_printing && !disable_convergence_message && !print_only_convergence_value) {
	    cout << "\nBreaking iteration; accuracy below machine epsilon." 
		 << endl;
	  }
	}
	break;
      }
    } else {
#pragma omp critical 
      {
	if(enable_printing && !print_only_convergence_value)
	  cout << '.' << flush;
      }
    }

    if (convergence_threshhold > 0 
	&& (approx_error_calculated 
	    || ((iteration + 1) % convergence_check_interval == 0))) {
      
      if(!approx_error_calculated) {
	approx_error = calc_approx_error();
	approx_error_calculated = true;
      }
      
      if(approx_error < max(convergence_threshhold, eps)) {
	// if(iteration < iter_second_scaleup) {
	//   iter_second_scaleup = 0;
	//   iter_first_scaleup = 0;
	//   continue;
	// }
	break;
      }
    }
    
    // Now look at how well we are doing.
    if(enable_printing && DEBUG_MODE && print_interval != 0 && ((iteration + 1) % print_interval) == 0) {
      cout << '\n' << (iteration + 1) << ".B\t Approximation Error (average L2-norm of error over signals) = " 
	   << (X - Gamma * D).norm() / (X.rows())
	   << flush;
    }
    
#ifndef NDEBUG
    for(auto& v : indices_by_dict) {
      assert_leq(v.size(), n);
      for(auto& vi : v)
	assert_leq(vi, n);
      assert_equal(set<size_t>(v.begin(), v.end()).size(), v.size());
    }
#endif

    // if(order_shuffle_interval != 0 
    //    && iteration % order_shuffle_interval == 0) {
    //   // Do it in a random order every few times
    //   shuffle(idx_order.begin(), idx_order.end(), rng);
    // }

    for(size_t jj = 0; jj < dict_size; ++jj) {

      size_t j = idx_order[jj];

      // Zero out this particular dictionary element
      const vector<size_t>& d_indices = indices_by_dict[j];

      if(d_indices.empty())
	continue;

      const size_t dsize = d_indices.size();
      const bool xi_mode = (dsize < n_xi);
      
      // cerr  << ". start: D.row(" << j << ") = " << D.row(j) << endl;

      if(!xi_mode)
	gfull.setZero();

      D.row(j).setZero();

      // cerr << "X: (" << X.rows() << "," << X.cols() << ")" << endl;
      // cerr << "gfull: (" << gfull.rows() << "," << gfull.cols() << ")" << endl;
      // cerr << "D: (" << D.rows() << "," << D.cols() << ")" << endl;
      // cerr << "Gamma_I: (" << Gamma_I.rows() << "," << Gamma_I.cols() << ")" << endl;
      // cerr << "g: (" << g.rows() << "," << g.cols() << ")" << endl;
      
      // Get the coefficients of Gamma using this dictionary element
#pragma omp parallel for if(enable_threading && dsize > 5000)
      for(size_t k = 0; k < dsize; ++k) {
	size_t d_idx = d_indices[k];
	
	// cerr << k << ".\t" << d_idx << ", Gamma.row(d_idx) = " << Gamma.row(d_idx) << endl;

	Gamma_I.row(k) = Gamma.row(d_idx);
	assert_notequal(Gamma(d_idx, j), 0);
	real_t v = g[k] = Gamma(d_idx, j);

	assert_equal(size_t(gfull.size()), n);
	assert_lt(d_idx, n);
	assert_equal(size_t(g.size()), n);
	assert_lt(k, n);

	if(!xi_mode)  
	  gfull[d_idx] = v;
	else
	  XI.row(k) = X.row(d_indices[k]);
      }

      // cerr << "HERE4 - OMP = " << omp_get_thread_num() << endl;

      switch(2*((parallel_optimize_descent_code 
		 && n_enabled > parallel_descent_num_signal_threshold) ? 1 : 0) 
	     + (xi_mode ? 1 : 0)) {

      case 2*0 + 0:

	for(size_t grad_desc = 0; grad_desc < max(size_t(1), grad_descent_iterations); ++grad_desc) {

	  dv = X.transpose() * gfull 
	    - D.transpose() * (Gamma_I.topRows(dsize).transpose() 
			       * g.head(dsize));

	  dv /= (1e-32 + dv.norm());

	  g_t1_full = X * dv;
	  g_t2.head(dsize) = Gamma_I.topRows(dsize) * (D * dv);

	  for(size_t k = 0; k < dsize; ++k)
	    gfull[d_indices[k]] = (g[k] = (g_t1_full(d_indices[k]) - g_t2(k)));
	}
	break;

      case 2*1 + 1: 
	// Eigen::setNbThreads(n_threads);
      case 2*0 + 1:

	for(size_t grad_desc = 0; grad_desc < max(size_t(1), grad_descent_iterations); ++grad_desc) {
	  dv = XI.topRows(dsize).transpose() * g.head(dsize)
	    - D.transpose() * (Gamma_I.topRows(dsize).transpose()
			       * g.head(dsize));

	  dv /= (1e-32 + dv.norm());
	  g.head(dsize) = XI.topRows(dsize) * dv - Gamma_I.topRows(dsize) * (D * dv);
	}
	Eigen::setNbThreads(1);
	break;

      case (2*1 + 0): 

	for(size_t grad_desc = 0; grad_desc < max(size_t(1), grad_descent_iterations); ++grad_desc) {
	  // cerr << "HERE5 - OMP = " << omp_get_thread_num() << endl;

	  const size_t d_block_msize = (dsize / n_chunks) + ((dsize % n_chunks == 0) ? 0 : 1);

#pragma omp parallel for shared(X, DV1, DV2, gfull, g, Gamma_I)
	  for(int i = 0; i < n_chunks; ++i) {
	    size_t X_block_start = i*X_block_msize;
	    size_t X_block_size = min((i+1)*X_block_msize, n) - X_block_start;

	    DV1.row(i) = (X.middleRows(X_block_start, X_block_size).transpose() 
			  * gfull.segment(X_block_start, X_block_size));

	    size_t d_block_start = i*d_block_msize;
	    size_t d_block_size = min((i+1)*d_block_msize, dsize) - d_block_start;

	    DV2.row(i) = (Gamma_I.middleRows(d_block_start, d_block_size).transpose() 
			  * g.segment(d_block_start, d_block_size));
	  }

	  dv = DV1.colwise().sum().transpose() - D.transpose() * DV2.colwise().sum().transpose();
	  dv /= (1e-32 + dv.norm());

	  Ddv = D * dv;
	  g_t2.head(dsize) = Gamma_I.topRows(dsize) * Ddv;
	  
#pragma omp parallel for shared(g_t1_full, X, d_indices, Gamma_I, g_t2, Ddv)
	  for(int i = 0; i < n_chunks; ++i) {
	    size_t X_block_start = i*X_block_msize;
	    size_t X_block_size = min((i+1)*X_block_msize, n) - X_block_start;

	    g_t1_full.segment(X_block_start, X_block_size) = X.middleRows(X_block_start, X_block_size) * dv;
	    
	    // DUNNO why, but this causes some major crash in eigen
	    // size_t d_block_start = i*d_block_msize;
	    // size_t d_block_size = min((i+1)*d_block_msize, dsize) - d_block_start;
	    // g_t2.segment(d_block_start, d_block_size) = Gamma_I.middleRows(d_block_start, d_block_size) * Ddv;
	    // }
	  }

#pragma omp parallel for shared(gfull, g, g_t1_full, g_t2, d_indices)
	  for(int i = 0; i < n_chunks; ++i) {
	    size_t d_block_start = i*d_block_msize;
	    size_t d_block_end = min((i+1)*d_block_msize, dsize);
	    for(size_t k = d_block_start; k < d_block_end; ++k)
	      gfull[d_indices[k]] = (g[k] = (g_t1_full(d_indices[k]) - g_t2(k)));
	  }
	}
	break; 
      }

      D.row(j) = dv;

      for(size_t k = 0; k < dsize; ++k)
	Gamma(d_indices[k], j) = g(k);
    }
  }

  if(enable_printing && !disable_convergence_message) {
#pragma omp critical 
    {
      cout << "\n" << endl;
    }
  }

  if(enable_threading)
    Eigen::setNbThreads(0);

  return (iteration + 1);
}

template <typename MatrixType> 
inline void KSVD(MatrixType& D, 
		 MatrixType& Gamma, 
		 const MatrixType& X,
		 size_t target_sparsity, 
		 size_t max_iterations, 
		 const param_t& options = param_t(),
		 typename enable_if<is_same<float, decltype(D.sum())>::value>::type* = 0)
{
  _KSVD_internal<float>(D, Gamma, X, target_sparsity, 
			max_iterations, options);
}

template <typename MatrixType>
void KSVD(MatrixType& D, MatrixType& Gamma, 
	  const MatrixType& X,
	  size_t target_sparsity, 
	  size_t max_iterations, 
	  param_t options = param_t(),
	  typename enable_if<is_same<double, decltype(D.sum())>::value>::type* = 0)
{
  size_t iter_start = 0;

  auto param = [&, options](const string& n) {
    param_t::const_iterator it;
    
    if((it = options.find(n)) == options.end())
      it = default_ksvd_parameters.find(n);

    return it->second;
  };

  size_t max_initial_32bit_iterations =			
    size_t(param("max_initial_32bit_iterations"));
  
  bool enable_32bit_initialization =		
    (param("enable_32bit_initialization") != 0);

  if(enable_32bit_initialization > 0) { 
    Matrix<float, Dynamic, Dynamic, RowMajor> _D = 
      D.template cast<float>();
    Matrix<float, Dynamic, Dynamic, RowMajor> _Gamma = 
      Gamma.template cast<float>();
    Matrix<float, Dynamic, Dynamic, RowMajor> _X = 
      X.template cast<float>();
    
    size_t _max_iterations = ((max_initial_32bit_iterations == 0)
			      ? max_iterations
			      : min(max_iterations, max_initial_32bit_iterations));

    size_t n_iter = _KSVD_internal<float>(_D, _Gamma, _X, target_sparsity, 
					  _max_iterations, options, true, 0);

    D = _D.cast<double>();
    Gamma = _Gamma.cast<double>();
    iter_start = n_iter;
    
    options["initialize_from_svd"] = false;
    options["initialize_random"] = false;
  }

  if(iter_start < max_iterations) {
    _KSVD_internal<double>(D, Gamma, X, target_sparsity, 
			   max_iterations, options, false, iter_start);
  }
}

void _KSVDNumpyWrapper(double *Dptr, double *Gammaptr, 
		       double *Xptr, size_t n, size_t d, size_t p,
		       size_t target_sparsity, size_t max_iterations, 
		       param_t params)
{
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > D(Dptr, d, p);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > Gamma(Gammaptr, n, d);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > X(Xptr, n, p);

    KSVD(D, Gamma, X, target_sparsity, max_iterations, params);
}

void _KSVDEncodeNumpyWrapper(double *Gammaptr, double *Dptr,  
                             double *Xptr, size_t n, size_t d, size_t p,
                             size_t target_sparsity,
                             param_t params)
{
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > Gamma(Gammaptr, n, d);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > D(Dptr, d, p);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > X(Xptr, n, p);

    OMPEncodeSignal(Gamma, D, X, target_sparsity, params);
}

#endif /* _KSVD_H_ */

