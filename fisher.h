
/// \Class    fisher fisher.h "fisher.h"
///  
/// \brief
///
/// \version  1.0
/// \author   Jorge A. Sanchez
/// \date     02/08/2010

#ifndef __FISHER_H
#define __FISHER_H

#include <limits>
#include <omp.h>

//#include "../ciiiutils/parameter_reader.h"
#include "simd_math.h"

#include "gmm.h"

// -------------------------------------------------------------------------
// Fisher Vector

struct fisher_param {
  fisher_param() :
    grad_weights(false), 
    grad_means(true), 
    grad_variances(true),
    alpha(0.5), 
    pnorm(2.0) { }
  bool grad_weights;
  bool grad_means;
  bool grad_variances;
  float alpha;
  float pnorm;
  float gamma_eps;
  //int read( const char *param_file, const char *category = "fisher" );
  void print();
};

// -------------------------------------------------------------------------

#define WEIGHTS_NORM 0

// -------------------------------------------------------------------------

// #include "ciiilibs.h"
// template<class T> class t_fisher;
// typedef t_fisher<ciii::real> fisher;

template<class T>
class fisher
{

public:
  
  fisher( fisher_param &_param );
  ~fisher( );
  
  void set_model( gmm_builder<T> &_gmm );

  // unweighted
  int compute( std::vector<T*> &x, T *fk );

  // weighted
  int compute( std::vector<T*> &x, std::vector<T> &wgh, T *fk);

  int dim(){ return fkdim; }

  float alpha(){ return param.alpha; }
  float pnorm(){ return param.pnorm; }
  void set_alpha( float a ){ param.alpha = a; }
  void set_pnorm( float p ){ param.pnorm = p; }

private:

  bool equal( T a, T b )
  { 
    if( fabs((T)a-(T)b)<std::numeric_limits<T>::epsilon() )
      return true;
    return false;
  }

  void alpha_and_lp_normalization( T *fk );

protected:

  fisher_param param;

  int gmmdim, ngauss, ngrad, fkdim;

  gmm_builder<T> *gmm;  

  T *iwgh, *istd;
};

#endif
