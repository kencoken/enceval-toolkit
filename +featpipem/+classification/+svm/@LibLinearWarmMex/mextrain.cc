#include <vector>
#include "mex.h"
#include "matrix.h"
#include "featpipec/src/classification/svm/liblinear_warm.h"
#include "../../../class_handle.h"

// f(obj, input, labels, init_w)
void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]) {
  // get properties from LibLinearWarmMex object
  double* c = mxGetPr(mxGetProperty(prhs[0],0,"c"));

  // get input parameters
  float* input = (float*)mxGetData(prhs[1]);
  mwSize feat_dim = mxGetM(prhs[1]);
  mwSize n = mxGetN(prhs[1]);

  const mxArray* mx_labels = prhs[2];
  mwSize n_classes = mxGetNumberOfElements(prhs[2]);
  
  float* init_w = 0;
  if (nrhs > 3) {
      init_w = (float*)mxGetData(prhs[3]);
      mwSize w_dim = mxGetM(prhs[3]);
      mwSize w_count = mxGetN(prhs[3]);
      if (w_dim != (feat_dim+1)) {
          mexErrMsgTxt("If specifying initial w, dimensionality must be feat_dim + 1");
      }
      if (w_count != n_classes) {
          mexErrMsgTxt("If specifying initial w, the number of w's specified must be equal to n_classes");
      }
  }

  // processing

  mexPrintf("\n----------------------------\n");
  mexPrintf("Feature Dimensionality: %d\n", feat_dim);
  mexPrintf("Feature Count: %d\n", n); //invert m&m Matlab->C
  mexPrintf("----------------------------\n\n");

  mexPrintf("Doing computation...\n");

  // copy cell array of matrices (MATLAB) to vector of vectors (C++)
  std::vector<std::vector<int> > labels(n_classes);
  for (int ci = 0; ci < n_classes; ++ci) {
    mxArray* mx_cls_labels = mxGetCell(mx_labels,ci);

    if (mxIsEmpty(mx_cls_labels)) {
      mexErrMsgTxt("Ground truth data not specified for some classes");
    }

    double* cls_labels = mxGetPr(mx_cls_labels);

    mwSize scn = mxGetN(mx_cls_labels);
    mwSize scm = mxGetM(mx_cls_labels);
    mwSize sample_count = (scn > scm) ? scn : scm;

    labels[ci].resize(sample_count);
    for (int ei = 0; ei < sample_count; ++ei) {
      labels[ci][ei] = cls_labels[ei]-1; // convert from 1-indexing to 0-indexing
    }
  }

  // get class handle
  class_handle<featpipe::LiblinearWarm>* solver = 
    convertMat2Ptr<featpipe::LiblinearWarm>(mxGetProperty(prhs[0],0,"c_handle"));

  // setup parameters
  solver->set_c(c[0]);

  // do training

  solver->train(input, feat_dim, n, labels, init_w);

  mexPrintf("Copying to MATLAB class property...\n");

  float* w = solver->get_w();

  mwSize w_dims[] = {feat_dim+1, n_classes};
  mxArray* mx_w = mxCreateNumericArray(2, w_dims, mxSINGLE_CLASS, mxREAL);
  float* w_out = (float*)mxGetData(mx_w);

  for (int i = 0; i < ((feat_dim+1)*n_classes); ++i) {
    w_out[i] = w[i];
  }

  plhs[0] = mxDuplicateArray(prhs[0]);
  mxSetProperty(plhs[0],0,"model",mx_w);
  
}
