
#include "fisher.h"

template<class T>
fisher<T>::fisher( fisher_param &_param )
  : param(_param), gmm(0), iwgh(0), istd(0)
{
  ngrad = (int)param.grad_weights + (int)param.grad_means + (int)param.grad_variances;
  assert( (param.alpha>0.0) && (param.alpha<=1.0) ); 
}

template<class T>
fisher<T>::~fisher()
{
  gmm = 0;

  if(iwgh)
    delete[] iwgh;
  iwgh=0;

  if(istd)
    delete[] istd;
  istd=0;

}

template<class T>
void
fisher<T>::set_model( gmm_builder<T> &_gmm )
{
  gmm = &_gmm;
  gmmdim = gmm->n_dim();
  ngauss = gmm->n_gauss();

  fkdim = 0;
  if( param.grad_weights )
    fkdim += ngauss;
  if( param.grad_means )
    fkdim += ngauss*gmmdim;
  if( param.grad_variances )
    fkdim += ngauss*gmmdim;      

  // gmm->print(1,1,1);

  if( iwgh )
    delete[] iwgh;

#if WEIGHTS_NORM
  iwgh = 0;
#else
  // precompute inverse weights
  iwgh = new T[ngauss];
  for( int j=0; j<ngauss; ++j )
  {
    assert( gmm->coef(j)>0.0 );
    iwgh[j] = 1.0/gmm->coef(j);
  } 
#endif

  // precompute inverse standard deviations
  if( param.grad_means || param.grad_variances )
  {
    if( istd )
      delete[] istd;
    istd = new T[ngauss*gmmdim];

    for( int j=0; j<ngauss; ++j ) 
    {
      T *var_j = gmm->variance(j);
      T *istd_j = istd+j*gmmdim;
      for( int k=0; k<gmmdim; ++k ) 
      {
        assert( var_j[k]>0.0 );
        istd_j[k] = 1.0/sqrt( var_j[k] );
      }
    }    
  }
}


template<class T>
int
fisher<T>::compute( std::vector<T*> &x, T *fk )
{
  std::vector<T> wghx( x.size(), 1.0 );  
  return compute( x, wghx, fk );
}

/// \brief Computes the FV for a set of samples
///
/// \param x     vector of (pointers to) low level samples
/// \param wghx  corresponding weights
/// \param fk    allocated output array

template<class T>
int 
fisher<T>::compute( std::vector<T*> &x, std::vector<T> &wghx, T *fk )
{  

  assert(gmm);

  assert( x.size()==wghx.size() );

  T wghsum=0.0;
#pragma omp parallel for reduction(+:wghsum)
  for( unsigned i=0; i<wghx.size(); ++i ) 
  {
    wghsum += wghx[i];
  }
  if( wghsum==0.0 )
  {
    memset( fk, 0, fkdim*sizeof(T) );
    return 0;
  }

  T *sum_gamma=0; 
  sum_gamma = new T[ngauss];  
  memset( sum_gamma, 0, ngauss*sizeof(T) );
  
  T *sum_gamma_x=0;
  if( param.grad_means || param.grad_variances ) 
  {
    sum_gamma_x = new T[ngauss*gmmdim];
    memset( sum_gamma_x, 0, ngauss*gmmdim*sizeof(T) );
  }

  T *sum_gamma_x2=0;
  if( param.grad_variances )
  {
    sum_gamma_x2 = new T[ngauss*gmmdim];
    memset( sum_gamma_x2, 0, ngauss*gmmdim*sizeof(T) );
  }

  T *gamma = new T[ngauss];

  for( unsigned i=0; i<x.size(); ++i ) 
  {

    gmm->posterior( x[i], gamma );

    // for( int j=ngauss; j--; ) 
    // {
    //   gamma[j] *= wghx[i];
    //   sum_gamma[j] += gamma[j];
    // }
    simd::scale( ngauss, gamma, wghx[i] );
    simd::add( ngauss, sum_gamma, gamma );

    T *x_i = x[i];
    if( param.grad_variances )
    {
#pragma omp parallel for
      for( int j=0; j<ngauss; ++j ) 
      {
        if( gamma[j]<gmm->min_gamma() )
          continue;        
        T *sum_gamma_x_j  = sum_gamma_x  + j*gmmdim;
        T *sum_gamma_x2_j = sum_gamma_x2 + j*gmmdim;
        // for( int k=gmmdim; k--; )
        // {
        //   T aux = gamma[j]*x_i[k];
        //   sum_gamma_x_j[k] += aux;
        //   sum_gamma_x2_j[k] += aux*x_i[k];
        // }
        simd::accumulate_stat( gmmdim, sum_gamma_x_j, sum_gamma_x2_j, x_i, gamma[j] );
      }
    }
    else if( param.grad_means )
    {
#pragma omp parallel for
      for( int j=0; j<ngauss; ++j ) 
      {
        if( gamma[j]<gmm->min_gamma() )
          continue;
        T *sum_gamma_x_j = sum_gamma_x+j*gmmdim;
        // for( int k=gmmdim; k--; ) 
        // {
        //   sum_gamma_x_j[k] += gamma[j]*x_i[k];
        // }
        simd::accumulate_stat( gmmdim, sum_gamma_x_j, x_i, gamma[j] );
      }
    }

  } // for i  

  delete[] gamma;
  gamma=0;

  T *p=fk;

  // Gradient w.r.t. the mixing weights 
  // without the constraint \sum_i pi_i=1 => Soft-BoV
  if( param.grad_weights )
  {
    for( int j=ngauss; j--; ) 
    {        
#if WEIGHTS_NORM
      p[j] = sum_gamma[j] / wghsum;
#else
      p[j] = sum_gamma[j] / ( wghsum*sqrt(iwgh[j]) );
#endif
    } 
    p += ngauss;
  }

  // Gradient w.r.t. the means
  if( param.grad_means )
  {
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *sum_gamma_x_j = sum_gamma_x+j*gmmdim;
      T *mean_j = gmm->mean(j);
      T *istd_j = istd+j*gmmdim;
      T *p_j = p+j*gmmdim;
#if WEIGHTS_NORM
      T mc = 1.0/wghsum;
#else
      T mc = sqrt(iwgh[j])/wghsum;
#endif

      // for( int k=gmmdim; k--; ) 
      // {
      //   p_j[k] = mc * ( sum_gamma_x_j[k] - mean_j[k]*sum_gamma[j] ) * istd_j[k];
      // }

      int ndim=gmmdim;
#if defined(__GNUC__) && defined(__SSE2__)
      if( (sizeof(T)==4) && (ndim%4==0) )
      {
        typedef float v4sf __attribute__ ((vector_size(16)));
        float _mc[4]; 
        _mc[0] = _mc[1] = _mc[2] = _mc[3] = mc;
        v4sf mcv = *(v4sf*)_mc;
        float _sum_gamma_j[4]; 
        _sum_gamma_j[0] = _sum_gamma_j[1] = _sum_gamma_j[2] = _sum_gamma_j[3] = sum_gamma[j];
        v4sf sum_gamma_j_v = *(v4sf*)_sum_gamma_j;

        while( ndim >= 4 )
        {
          v4sf sum_gamma_x_j_v = *(v4sf*)sum_gamma_x_j;
          v4sf mean_j_v = *(v4sf*)mean_j;
          v4sf istd_j_v = *(v4sf*)istd_j;
          *(v4sf*)p_j = mcv * ( sum_gamma_x_j_v - mean_j_v * sum_gamma_j_v ) * istd_j_v;
          sum_gamma_x_j += 4;
          mean_j += 4;
          istd_j += 4;
          p_j += 4;
          ndim -= 4;
        }
      }
      else if( (sizeof(T)==8) && (ndim%2==0) )
      {
        typedef double v2df __attribute__ ((vector_size(16)));
        double _mc[2]; 
        _mc[0] = _mc[1] = mc;
        v2df mcv = *(v2df*)_mc;
        double _sum_gamma_j[2]; 
        _sum_gamma_j[0] = _sum_gamma_j[1] = sum_gamma[j];
        v2df sum_gamma_j_v = *(v2df*)_sum_gamma_j;
        while( ndim >= 2 )
        {
          v2df sum_gamma_x_j_v = *(v2df*)sum_gamma_x_j;
          v2df mean_j_v = *(v2df*)mean_j;
          v2df istd_j_v = *(v2df*)istd_j;
          *(v2df*)p_j = mcv * ( sum_gamma_x_j_v - mean_j_v * sum_gamma_j_v ) * istd_j_v;
          sum_gamma_x_j += 2;
          mean_j += 2;
          istd_j += 2;
          p_j += 2;
          ndim -= 2;
        }    
      }
#endif
      while( ndim-- ) 
      {
        *p_j = mc * ( *sum_gamma_x_j - *mean_j * sum_gamma[j] ) * *istd_j;
        p_j++;
        mean_j++;
        istd_j++;
        sum_gamma_x_j++;
      }
    }
    p += ngauss*gmmdim;
  }

  // Gradient w.r.t. the variances
  if( param.grad_variances )
  {

#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *sum_gamma_x_j = sum_gamma_x+j*gmmdim;
      T *sum_gamma_x2_j = sum_gamma_x2+j*gmmdim;
      T *mean_j = gmm->mean(j);
      T *var_j = gmm->variance(j);
      T *p_j = p+j*gmmdim;
#if WEIGHTS_NORM
      T vc = 1.0/wghsum;
#else
      T vc = sqrt(0.5*iwgh[j])/wghsum;
#endif

      // for( int k=gmmdim; k--; ) 
      // {
      //   p_j[k] = vc * ( ( sum_gamma_x2_j[k] + mean_j[k] * ( mean_j[k]*sum_gamma[j] - 2.0*sum_gamma_x_j[k] ) ) / var_j[k] - sum_gamma[j] );
      // }

      int ndim=gmmdim;
#if defined(__GNUC__) && defined(__SSE2__)
      if( (sizeof(T)==4) && (ndim%4==0) )
      {
        typedef float v4sf __attribute__ ((vector_size(16)));
        float _vc[4];
        _vc[0] = _vc[1] = _vc[2] = _vc[3] = vc;
        v4sf vcv = *(v4sf*)_vc;
        float _sum_gamma_j[4]; 
        _sum_gamma_j[0] = _sum_gamma_j[1] = _sum_gamma_j[2] = _sum_gamma_j[3] = sum_gamma[j];
        v4sf sum_gamma_j_v = *(v4sf*)_sum_gamma_j;
        while( ndim >= 4 )
        {
          v4sf sum_gamma_x_j_v = *(v4sf*)sum_gamma_x_j;
          v4sf sum_gamma_x2_j_v = *(v4sf*)sum_gamma_x2_j;
          v4sf mean_j_v = *(v4sf*)mean_j;
          v4sf var_j_v = *(v4sf*)var_j;
          *(v4sf*)p_j = vcv * ( ( sum_gamma_x2_j_v + mean_j_v * ( mean_j_v * sum_gamma_j_v - sum_gamma_x_j_v - sum_gamma_x_j_v ) ) / var_j_v - sum_gamma_j_v );
          sum_gamma_x2_j += 4;
          sum_gamma_x_j += 4;
          mean_j += 4;
          var_j += 4;
          p_j += 4;
          ndim -= 4;
        }    
      }
      else if( (sizeof(T)==8) && (ndim%2==0) )
      {
        typedef double v2df __attribute__ ((vector_size(16)));
        double _vc[2];
        _vc[0] = _vc[1] = vc;
        v2df vcv = *(v2df*)_vc;
        double _sum_gamma_j[2]; 
        _sum_gamma_j[0] = _sum_gamma_j[1] = sum_gamma[j];
        v2df sum_gamma_j_v = *(v2df*)_sum_gamma_j;
        while( ndim >= 2 )
        {
          v2df sum_gamma_x_j_v = *(v2df*)sum_gamma_x_j;
          v2df sum_gamma_x2_j_v = *(v2df*)sum_gamma_x2_j;
          v2df mean_j_v = *(v2df*)mean_j;
          v2df var_j_v = *(v2df*)var_j;
          *(v2df*)p_j = vcv * ( ( sum_gamma_x2_j_v + mean_j_v * ( mean_j_v * sum_gamma_j_v - sum_gamma_x_j_v - sum_gamma_x_j_v ) ) / var_j_v - sum_gamma_j_v );
          sum_gamma_x2_j += 2;
          sum_gamma_x_j += 2;
          mean_j += 2;
          var_j += 2;
          p_j += 2;
          ndim -= 2;
        }    
      }
#endif
      while( ndim-- ) 
      {
        *p_j = vc * ( ( *sum_gamma_x2_j + *mean_j * ( *mean_j * sum_gamma[j] - 2.0 * *sum_gamma_x_j ) ) / *var_j - sum_gamma[j] );
        sum_gamma_x2_j++;
        sum_gamma_x_j++;
        mean_j++;
        var_j++;
        p_j++;
      }

    }
  } 

  alpha_and_lp_normalization(fk);
  
  if(sum_gamma)
    delete[] sum_gamma;
  sum_gamma=0;

  if(sum_gamma_x)
    delete[] sum_gamma_x;
  sum_gamma_x=0;

  if(sum_gamma_x2)
    delete[] sum_gamma_x2;
  sum_gamma_x2=0;

  return 0;
}


template<class T>
void
fisher<T>::alpha_and_lp_normalization( T *fk )
{
  // alpha normalization
  if( !equal(param.alpha,(float)1.0) )
  {
    if( equal(param.alpha,(float)0.5) )
    {
#pragma omp parallel for
      for( int i=0; i<fkdim; i++ )
      {
        if( fk[i]<0.0 )
          fk[i] = -std::sqrt(-fk[i]);
        else
          fk[i] = std::sqrt(fk[i]);
      }
    }
    else
    {
#pragma omp parallel for
      for( int i=0; i<fkdim; i++ )
      {
        if( fk[i]<0.0 )
          fk[i] = -std::pow(-fk[i],(T)param.alpha);
        else
          fk[i] = std::pow(fk[i],(T)param.alpha);
      }
    }
  }

  // Lp normalization
  if( !equal(param.pnorm,(float)0.0) )
  {
    T pnorm=0;
    if( equal(param.pnorm,(float)1.0) )
    {
#pragma omp parallel for reduction(+:pnorm)
      for( int i=0; i<fkdim; ++i )
      {
        pnorm += std::fabs(fk[i]);
      }
    }
    else if( equal(param.pnorm,2.0) )
    {
// #pragma omp parallel for reduction(+:pnorm)
//       for( int i=0; i<fkdim; ++i )
//       {
//         pnorm += fk[i]*fk[i];
//       }
//       pnorm = std::sqrt(pnorm);
      pnorm = sqrt( simd::dot_product( fkdim, fk, fk ) );
    }
    else
    {
#pragma omp parallel for reduction(+:pnorm)
      for( int i=0; i<fkdim; ++i )
      {
        pnorm += std::pow( std::fabs(fk[i]), (T)param.pnorm );
      }
      pnorm = std::pow( static_cast<float>(pnorm), static_cast<float>(1.0/(T)param.pnorm) );
    }

    if( pnorm>0.0 )
    {
//       pnorm = 1.0/pnorm;
// #pragma omp parallel for
//       for( int i=0; i<fkdim; ++i )
//       {
//         fk[i] *= pnorm;
//       }
      simd::scale( fkdim, fk, 1.0/pnorm );
    }
  }
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
fisher_param::print()
{
  std::cout << "  grad_weights = " << grad_weights << std::endl;
  std::cout << "  grad_means = " << grad_means << std::endl;
  std::cout << "  grad_variances = " << grad_variances << std::endl;
  std::cout << "  alpha = " << alpha << std::endl;
  std::cout << "  pnorm = " << pnorm << std::endl;
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

// int fisher_param::read( const char *param_file, const char *category )
// {
//   parameter_reader parm_file( param_file );
//   int e=0;

//   e += parm_file.get_value( grad_weights, category, "grad_weights" );
//   e += parm_file.get_value( grad_means, category, "grad_means" );
//   e += parm_file.get_value( grad_variances, category, "grad_variances" );
//   e += parm_file.get_value( alpha, category, "alpha" );
//   e += parm_file.get_value( pnorm, category, "pnorm" );

//   if(e)
//   {
//     std::cout << "error in \"" << category << "\" parameters" << std::endl;
//     return -1;
//   }
//   return 0;
// }


template class fisher<float>;
template class fisher<double>;
