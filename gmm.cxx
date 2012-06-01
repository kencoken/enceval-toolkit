
#include "gmm.h"

/// \bief constructor
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
gmm_builder<T>::gmm_builder( const char* modelfile )
  : gmm_mean(0), gmm_var(0), gmm_pi(0), gamma_t(0), i_var(0), var_thr(0), log_pi(0), log_var_sum(0), ngauss(0), ndim(0)
{  
  int err = this->load_model( modelfile );
  assert( err==0 );
}

template<class T>
gmm_builder<T>::gmm_builder( int n_gauss, int n_dim )
  : ngauss(n_gauss), ndim(n_dim)
{
  init();
}

template<class T>
gmm_builder<T>::gmm_builder( gmm_builder_param &p, int n_gauss, int n_dim )
  : param(p), ngauss(n_gauss), ndim(n_dim)
{
  init();
}

template<class T>
void 
gmm_builder<T>::init()
{
  if( (ngauss==0) || (ndim==0) )
  {
    gmm_mean=0;
    gmm_var=0;
    gmm_pi=0;

    gamma_t=0;
    log_pi=0;
    log_var_sum=0;
    i_var=0;

    var_thr = 0;
  }
  else
  {
    gmm_mean = new T*[ngauss];
    for( int k=ngauss; k--; )
      gmm_mean[k] = new T[ndim];

    gmm_var = new T*[ngauss];
    for( int k=ngauss; k--; )
      gmm_var[k] = new T[ndim];

    gmm_pi = new T[ngauss];

    gamma_t = new T*[ngauss];
    for( int k=ngauss; k--; )
      gamma_t[k] = 0;

    log_pi = new T[ngauss];

    log_var_sum = new double[ngauss];

    i_var = new T*[ngauss];
    for( int k=ngauss; k--; )
      i_var[k] = new T[ndim];

    // initial default values  
    for( int k=ngauss; k--; )
    {
      for( int i=ndim; i--; )
      {
        gmm_mean[k][i] = 0.0;
        gmm_var[k][i] = 1.0;
      }
      gmm_pi[k] = 1.0/T(ngauss);      
    }

    ndim_log_2pi = (T)(double(ndim)*log(2.0*M_PI));

    var_thr = 0;
  }

}

/// \bief destructor
/// 
/// \param none
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
gmm_builder<T>::~gmm_builder()
{
  clean();
}

template<class T>
void
gmm_builder<T>::clean()
{
  if( gmm_mean )
  {
    for( int k=ngauss; k--; )
      delete[] gmm_mean[k];
    delete[] gmm_mean;
    gmm_mean = 0;
  }

  if( gmm_var )
  {
    for( int k=ngauss; k--; )
      delete[] gmm_var[k];
    delete[] gmm_var;
    gmm_var = 0;
  }

  if( gmm_pi )
  {
    delete[] gmm_pi;
    gmm_pi = 0;
  }

  if( gamma_t )
  {
    for( int k=ngauss; k--; )
      if( gamma_t[k] ) delete[] gamma_t[k];
    delete[] gamma_t;
    gamma_t = 0;
  }

  if( log_pi )
  {
    delete[] log_pi;
    log_pi = 0;
  }

  if( log_var_sum )
  {
    delete[] log_var_sum;
    log_var_sum = 0;
  }

  if( i_var )
  {
    for( int k=ngauss; k--; )
      delete[] i_var[k];
    delete[] i_var;
    i_var = 0;
  }

  if( var_thr )
  {
    delete[] var_thr;
    var_thr = 0;
  }
  
}

/// \bief set mean for the GMM 
/// 
/// \param mean new means (of the same dim)
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::set_mean( std::vector<T*> &_mean )
{
  assert( (int)_mean.size()==ngauss );
  for( int k=0; k<ngauss; k++ )
  {
    for( int i=0; i<ndim; i++ )  
    {
      gmm_mean[k][i] = _mean[k][i];
    }
  }
}

/// \bief set variance for the GMM 
/// 
/// \param var new variances (of the same dim)
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::set_variance( std::vector<T*> &var )
{
  assert( (int)var.size()==ngauss );

  for( int k=ngauss; k--; )
  {
    for( int i=ndim; i--; )
    {
      gmm_var[k][i] = std::max( (T)param.variance_floor, (T)var[k][i] );

      // // if <floor, make it responsible for a large number of points
      // if( var[k][i]<param.variance_floor )
      //   gmm_var[k][i] = 1.0;
    }
  }

}

/// \bief set mixing coefficients for the GMM 
/// 
/// \param coef new mixing coefficients (of the same dim)
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::set_mixing_coefficients( std::vector<T> &_coef )
{
  assert( (int)_coef.size()==ngauss );
  T sum=0.0;
  for( int k=ngauss; k--; )
    sum += _coef[k];
  assert(sum>0.0);
  for( int k=ngauss; k--; )
    gmm_pi[k] = _coef[k]/sum;
}

/// \bief set parameters for the GMM 
/// 
/// \param mean new means
/// \param var new variances
/// \param coef new mixing coefficients
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::set_model( std::vector<T*> &_mean,
                           std::vector<T*> &_var,
                           std::vector<T>  &_coef )
{
  set_mean( _mean );
  set_variance( _var );
  set_mixing_coefficients( _coef );
  prepare();
}

/// \bief random initialization of mean vectors
/// 
/// \param samples samples list
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::random_init( std::vector<T*>& samples, int seed )
{
  int N=samples.size();
  assert( N>0 );

  T dmin[ndim], dmax[ndim];
  for( int i=0; i<ndim; ++i )
    dmin[i] = dmax[i] = samples[0][i];

  for( int n=1; n<N; ++n )
  {
    for( int i=0; i<ndim; ++i )
    {
      if( samples[n][i]<dmin[i] )
        dmin[i] = samples[n][i];
      else if( samples[n][i]>dmax[i] )
        dmax[i] = samples[n][i];
    }
  }
  T m[ndim], v[ndim];
  sample_mean( samples, m, ndim );
  sample_variance( samples, m, v, ndim );
  
  srand( seed );

  for( int k=ngauss; k--; )
  {
    for( int i=ndim; i--; )
    {
      T drange = dmax[i]-dmin[i];
      gmm_mean[k][i] = dmin[i]+drange*T(rand())/T(RAND_MAX);
      gmm_var[k][i] = std::max( (T)param.variance_floor, (T)0.1*drange*drange );
    }
    gmm_pi[k] = 1.0/T(ngauss);
  }

  prepare();

}

/// \bief EM-iteration
/// 
/// \param samples samples list
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::em( std::vector<T*> &samples )
{
  nsamples = samples.size();
  assert( nsamples>0 );

  em_min_pi = (T)param.min_count/(T)nsamples;

  compute_variance_threshold(samples);

  std::cout << " Number of Gaussians: " << ngauss << std::endl;
  std::cout << " Number of samples: " << nsamples << std::endl;
  std::cout << " Sample dimensions: " << ndim << std::endl;

  T llh_init=0, llh_prev=0, llh_curr=0, llh_diff=0;

  std::cout << " EM-iterations / average Log-Likelihood" << std::endl;  

  for( int iter=0; iter<param.max_iter; ++iter )
  {  

    prepare();

    e_step( samples );

    if( iter==0 )
    {
      llh_init = log_likelihood( samples );
      llh_curr = llh_init;
      std::cout << "   (0) Log-Likelihood = " << llh_init << std::endl;  
    }
    else
    {
      llh_prev = llh_curr;
      llh_curr = log_likelihood( samples );
      llh_diff = (llh_curr-llh_prev)/(llh_curr-llh_init);

      std::cout << "   (" << iter << ") Log-Likelihood = " << llh_curr << " (" << llh_diff << ")" << std::endl;

      if( llh_diff<param.llh_diff_thr )
        break;
    }

    m_step( samples );

  }

}

/// \bief Adaptive variance floornig
/// 
/// \param samples samples list
///
/// \return none
///
/// \author Jorge Sanchez
/// \date   October 2010

template<class T>
void
gmm_builder<T>::compute_variance_threshold( std::vector<T*> &x )
{
  if( var_thr )
    delete[] var_thr;
  var_thr = new T[ndim];

  // variance of the sample set
  sample_variance( x, var_thr, ndim );

  // simd::scale( ndim, var_thr, 1.0/(T)(ngauss*ngauss) );

  for( int i=ndim; i--;  ) 
  {
    // proportional to the sample variance
    var_thr[i] = param.variance_floor_factor * var_thr[i];

    // and floored
    var_thr[i] = std::max( (T)var_thr[i], (T)param.variance_floor );
  }
}

/// \bief E-step
/// 
/// \param samples samples list
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::e_step( std::vector<T*> &samples )
{

  for( int k=ngauss; k--; )
  {
    if( gamma_t[k] ) 
      delete[] gamma_t[k];
    gamma_t[k] = new T[nsamples];
  }

#pragma omp parallel for
  for( int n=0; n<nsamples; n++ )
  {
    T *gamma_t_n = new T[ngauss];
    posterior( samples[n], gamma_t_n );

    for( int k=ngauss; k--; ) // transpose
    { 
      gamma_t[k][n] = gamma_t_n[k];
    }

    delete[] gamma_t_n;
  }
}

/// \brief Posterior of a sample (responsibilities)
/// 
/// \param x sample
/// \param pst k-dimensional vector (output)
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2010

template<class T>
void
gmm_builder<T>::posterior( T *x, T *pst )
{

  log_likelihood( x, pst );

  T pst_sum=0.0;

#pragma omp parallel for reduction(+:pst_sum)
  for( int k=0; k<ngauss; ++k )
  {
    pst[k] = exp( pst[k] );
    pst_sum += pst[k];
  }

  if( pst_sum>0.0 )
  {
    simd::scale( ngauss, pst, (T)1.0/pst_sum );
  }

}

/// \brief log-likelihood (and log-posterior) of a sample
///        avoiding numerical underflow.
///
/// The likelihood of a sample x can be written as:
///
///    p(x) = sum_k ( pi_k * N_k(x) )
///         = sum_k exp( log(pi_k) + log(N_k(x)) )
///
/// and the log-likelihood:
///
///   llh = log[ sum_k( exp(y_k) ) ]
///
/// where y_k = log(pi_k) + log(N_k(x)). Let's write y_max = max_k(y_k).
/// Defining y'_k = y_k-y_max, one can write:
///
///   log[ sum_k( exp(y_k) ) = log[ sum_k( exp(y'_k+y_max) ) ]
///                          = y_max + log[ sum_k( exp(y'_k) ) ]
///
/// \param x sample
/// \param log_pst k-dimensional vector (output)
///
/// \return log-likelihood of the sample
///
/// \author Jorge Sanchez
/// \date    August 2010

template<class T>
T
gmm_builder<T>::log_likelihood( T *x, T *log_pst )
{

  T *lp=0;
  if( log_pst )
  {
    lp = log_pst;
  }
  else
  {
    lp = new T[ngauss];
  }

#pragma omp parallel for
  for( int k=0; k<ngauss; ++k )
  {
    lp[k] = log_gauss( k, x );
  }
  simd::add( ngauss, lp, log_pi );

  T log_p_max = lp[0];
  for( int k=1; k<ngauss; ++k )
  {
    if( lp[k] > log_p_max )
    {
      log_p_max = lp[k];
    }
  }

  T log_p_sum = 0.0;
#pragma omp parallel for reduction(+:log_p_sum)
  for( int k=0; k<ngauss; ++k )
  {
    log_p_sum += exp( lp[k]-log_p_max );
  }
  log_p_sum = log_p_max + log(log_p_sum);

  if( log_pst )  // Compute Log-Posteriors
  {
    simd::offset( ngauss, lp, -log_p_sum );
  }
  else // Just interested on the Log-likelihood of the sample
  {
    delete[] lp;
    lp = 0;
  }
  
  return (T)log_p_sum;
}

/// \bief Gaussian Log-probability
/// 
/// \param k Gaussian component
/// \param x sample
///
/// \return log(p_k)
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
T
gmm_builder<T>::log_gauss( int k, T* x )
{
  // T log_p = ndim_log_2pi + log_var_sum[k];
  // for( int i=ndim; i--; )
  // {
  //   T xc = x[i]-gmm_mean[k][i];
  //   log_p += xc*xc*i_var[k][i];
  // }
  // return -0.5*log_p;

  double log_p = (double) ndim_log_2pi + log_var_sum[k];
  log_p += (double)simd::weighted_l2_sq( ndim, x, gmm_mean[k], i_var[k] );
  return (T)(-0.5*log_p);

}


/// \bief M-step (batch version, see PRML for an incremental version)
/// 
/// \param samples samples list
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::m_step( std::vector<T*> &samples )
{

#pragma omp parallel for
  for( int k=0; k<ngauss; k++ )
  {

    // if less than certain threshold do not update this Gaussian
    if( gmm_pi[k]<=em_min_pi )
      continue;

    T *gamma_k = gamma_t[k];
    T *mean_k = gmm_mean[k];
    T *var_k = gmm_var[k];
    
    T i_Nk = 0.0;
    int nused = 0;
    for( int n=nsamples; n--; )
    {
      if( gamma_k[n]<param.min_gamma )
        continue;
      i_Nk += gamma_k[n];
      nused++;
    }

    assert( i_Nk>0.0 );
    i_Nk = 1.0/i_Nk;
      
    // update means
    memset( mean_k, 0, ndim*sizeof(T) );
    for( int n=nsamples; n--; )
    {
      if( gamma_k[n]<param.min_gamma )
        continue;
      simd::accumulate_stat( ndim, mean_k, samples[n], gamma_k[n]);
    }
    simd::scale( ndim, mean_k, i_Nk );

    // update variance    
    memset( var_k, 0, ndim*sizeof(T) );
    for( int n=nsamples; n--; )
    {
      if( gamma_k[n]<param.min_gamma )
        continue;     
      simd::accumulate_stat_centered( ndim, var_k, samples[n], mean_k, gamma_k[n]);
    }
    simd::scale( ndim, var_k, i_Nk );

    for( int i=ndim; i--; )
    {
      var_k[i] = std::max( var_k[i], var_thr[i] );
    }

    // update mixing coeficient
    gmm_pi[k] = std::max( (T)1.0/(i_Nk*(T)nused), (T)em_min_pi );
  }

  // enforce normalization
  T pi_sum=0.0;
  for( int k=ngauss; k--; )  
  {
    pi_sum += gmm_pi[k];
  }
  assert( pi_sum>0.0 );
  simd::scale( ngauss, gmm_pi, 1.0/pi_sum );
  
  for( int k=ngauss; k--; )  
  {
    assert(gmm_pi[k] > 0.0);
  }

}

/// \bief Average Log-Likelihood.
/// 
/// \param samples samples list
///
/// \return average Log-Likelihood
///
/// \author Jorge Sanchez
/// \date   August 2009

template<class T>
T
gmm_builder<T>::log_likelihood( std::vector<T*> &samples )
{
  nsamples = samples.size();
  assert( nsamples>0 );

  long double llh=0.0;
  for( int i=0; i<nsamples; i++ ) 
  {
    llh += (long double)log_likelihood( samples[i] );
  }
  return (T)(llh/(long double)nsamples);
}


/// \bief prepare auxiliary variables
/// 
/// \param none
///
/// \return none
///
/// \author Jorge Sanchez
/// \date   August 2009

template<class T>
void
gmm_builder<T>::prepare()
{

#pragma omp parallel for
  for( int k=0; k<ngauss; k++ ) 
  {
    log_pi[k] = log( gmm_pi[k] );

    log_var_sum[k] = 0.0;
    for( int i=ndim; i--; ) 
    {
      i_var[k][i] = 1.0/gmm_var[k][i];
      log_var_sum[k] += (double)log( gmm_var[k][i] );
    }
  }

}

/// \bief print model
/// 
/// \param none
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
void
gmm_builder<T>::print( bool print_pi, bool print_mean, bool print_var )
{
  assert( print_pi || print_mean || print_var );

  for( int k=0; k<ngauss; ++k )
  {    
    if( print_pi )
      std::cout << " pi[" << k << "] = " << gmm_pi[k] << std::endl;

    if( print_mean )
    {
      std::cout << " mean[" << k << "] = [";
      for( int i=0; i<ndim; ++i )
        std::cout << " " << gmm_mean[k][i];
      std::cout << " ]" << std::endl;
    }

    if( print_var )
    {
      std::cout << " variance[" << k << "] = [";
      for( int i=0; i<ndim; ++i )
        std::cout << " " << gmm_var[k][i];
      std::cout << " ]" << std::endl;
    }
  }
}

/// \bief load model from file
/// 
/// \param none
///
/// \return -1 if error
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
int 
gmm_builder<T>::load_model( const char* filename )
{
  clean();

  std::ifstream fin( filename, std::ios::in | std::ios::binary );
  if( fin.fail() )
    return -1;

  fin.read( (char*)&ngauss, sizeof(int) );
  fin.read( (char*)&ndim,   sizeof(int) );

  init();
  
  for( int k=0; k<ngauss; k++ )
  {
    fin.read( (char*)&gmm_pi[k], sizeof(T) );
    fin.read( (char*)&gmm_mean[k][0], ndim*sizeof(T) );
    fin.read( (char*)&gmm_var[k][0], ndim*sizeof(T) );
  }

  fin.close();
  if( fin.fail() )
    return -1;

  prepare();

  return 0;
}

/// \bief save model to file
/// 
/// \param none
///
/// \return -1 if error
///
/// \author Jorge Sanchez
/// \date    August 2009

template<class T>
int 
gmm_builder<T>::save_model( const char* filename )
{
  std::ofstream fout( filename, std::ios::out | std::ios::binary );
  if( fout.fail() )
    return -1;

  fout.write( (char*)&ngauss, sizeof(int) );
  fout.write( (char*)&ndim, sizeof(int) );
  
  for( int k=0; k<ngauss; k++ )
  {
    fout.write( (char*)&gmm_pi[k], sizeof(T) );
    fout.write( (char*)&gmm_mean[k][0], ndim*sizeof(T) );
    fout.write( (char*)&gmm_var[k][0], ndim*sizeof(T) );
  }

  fout.close();
  if( fout.fail() )
    return -1;

  return 0;
}


/// \bief print
/// 
/// \param none
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

void
gmm_builder_param::print()
{
  std::cout << "  max_iter = " << max_iter << std::endl;
  std::cout << "  min_count = " << min_count << std::endl;
  std::cout << "  llh_diff_thr = " << llh_diff_thr << std::endl;
  std::cout << "  grow_factor = " << grow_factor << std::endl;
  std::cout << "  min_gamma = " << min_gamma << std::endl;
  std::cout << "  variance_floor = " << variance_floor << std::endl;
  std::cout << "  variance_floor_factor = " << variance_floor_factor << std::endl;
}


// /// \brief    Read parameters
// ///
// /// \param    param_file parameters file
// /// \param    category [category]
// ///
// /// \return   -1 in case of error
// ///
// /// \author   Jorge A. Sanchez
// /// \date     23/07/2010

// int 
// gmm_builder_param::read( const char *param_file, const char *category )
// {
//   parameter_reader parm_file( param_file );
//   int e=0;

//   e += parm_file.get_value( max_iter, category, "max_iter" );
//   e += parm_file.get_value( min_count, category, "min_count" );
//   e += parm_file.get_value( llh_diff_thr, category, "llh_diff_thr" );
//   e += parm_file.get_value( grow_factor, category, "grow_factor" );
//   e += parm_file.get_value( min_gamma, category, "min_gamma" );
//   e += parm_file.get_value( variance_floor, category, "variance_floor" );  
//   e += parm_file.get_value( variance_floor_factor, category, "variance_floor_factor" );

//   if(e)
//   {
//     std::cout << "error in \"" << category << "\" parameters" << std::endl;
//     return -1;
//   }
//   return 0;
// }


template class gmm_builder<float>;
template class gmm_builder<double>;

