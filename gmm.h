
#ifndef __GMM_H
#define __GMM_H

#include <vector>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <limits>
#include <omp.h>

#include "stat.h"

//#include "../ciiiutils/parameter_reader.h"
#include "simd_math.h"

struct gmm_builder_param 
{
  gmm_builder_param(): 
    max_iter(100), 
    min_count(10),
    llh_diff_thr(0.00001), 
    grow_factor(0.1),
    min_gamma(1e-12),
    variance_floor(1.0e-9), 
    variance_floor_factor(0.01) {}
  int max_iter;         // max. number of EM iterations
  int min_count;        // min. count of samples to update a Gaussian
  float llh_diff_thr;   // average Log-Likelihood difference threshold
  float grow_factor;    // growing factor for split
  float min_gamma;      // min. posterior prob. for a sample
  float variance_floor; // hard variance floor
  float variance_floor_factor; // factor for the adaptive flooring
  int read( const char *param_file, const char *category = "gmm_builder" );
  void print();
};

/// \class    gmm_builder gmm.h "gmm.h"
///  
/// \brief    Gaussian (diagonal covariance) Mixture Model using EM-algorithm
///
/// \author   Jorge Sanchez
/// \date     29/07/2009

template<class T>
class gmm_builder 
{
public:

  gmm_builder( const char* modelfile );

  gmm_builder( int n_gauss, int n_dim );

  gmm_builder( gmm_builder_param &p, int n_gauss, int n_dim );

  ~gmm_builder();

  void set_model( std::vector<T*> &mean, 
                  std::vector<T*> &var,
                  std::vector<T>  &coef );

  void random_init( std::vector<T*> &samples, int seed=-1 );

  void e_step( std::vector<T*> &samples );
  void m_step( std::vector<T*> &samples );
  void em( std::vector<T*> &samples );

  T log_likelihood( std::vector<T*> &samples );

  void posterior( T* sample, T *pst );

  inline T* mean( int k ){ return gmm_mean[k]; }
  inline T* variance( int k ){ return gmm_var[k]; }
  inline T  coef( int k ){ return gmm_pi[k]; }

  void print( bool pi=true, bool mean=false, bool var=false );

  int n_dim(){ return ndim; }
  int n_gauss(){ return ngauss; }

  int load_model( const char* filename );
  int save_model( const char* filename );

  T min_gamma(){ return (T)param.min_gamma; }

private:

  void set_mean( std::vector<T*> &mean );
  void set_variance( std::vector<T*> &var );
  void set_mixing_coefficients( std::vector< T > &coef ); 

protected:

  gmm_builder_param param;

  void init();
  void clean();

  void prepare();

  void compute_variance_threshold( std::vector<T*> &x );

  // mean, variance and mixing coefficients
  T **gmm_mean, **gmm_var, *gmm_pi;

  // Responsibilities (matrix of size (ngauss)x(nsamples) )
  T **gamma_t;

  T **i_var, *var_thr, *log_pi;

  double *log_var_sum; // accumulate as double

  int ngauss, ndim, nsamples;

  T em_min_pi;

  T ndim_log_2pi;

  T log_likelihood( T* x, T *log_pst=0 );

  T log_gauss( int k, T* x );

};


#endif
