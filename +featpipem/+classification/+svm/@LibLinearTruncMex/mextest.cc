#include "mex.h"
#include "matrix.h"
#include "featpipec/src/classification/svm/liblineartrunc.h"
#include "../../../class_handle.h"

// [est_label, scoremat] = f(obj, input)
void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]) {
  // get input parameters
  float* input = (float*)mxGetData(prhs[1]);
  mwSize feat_dim = mxGetM(prhs[1]);
  mwSize n = mxGetN(prhs[1]);

  // processing

  mexPrintf("\n----------------------------\n");
  mexPrintf("Feature Dimensionality: %d\n", feat_dim);
  mexPrintf("Feature Count: %d\n", n); //invert m&m Matlab->C
  mexPrintf("----------------------------\n\n");

  mexPrintf("Doing computation...\n");

  // get class handle
  class_handle<featpipe::LiblinearTrunc>* solver = 
    convertMat2Ptr<featpipe::LiblinearTrunc>(mxGetProperty(prhs[0],0,"c_handle"));

  // do testing

  if (solver->get_feat_dim() != feat_dim) {
    mexErrMsgTxt("Feature dimensionality of input is incorrect");
  }

  int* est_label = new int[n];
  int num_classes = solver->get_num_classes();
  float* scoremat = new float[num_classes*n];

  solver->test(input, n, &est_label, &scoremat);

  mexPrintf("Copying to MATLAB class property...\n");

  mwSize est_label_dims[] = {n, 1};
  plhs[0] = mxCreateNumericArray(2, est_label_dims, mxINT32_CLASS, mxREAL);
  int* est_label_out = (int*)mxGetData(plhs[0]);

  for (int i = 0; i < n; ++i) {
    est_label_out[i] = est_label[i];
  }

  mwSize scoremat_dims[] = {n, num_classes};
  plhs[1] = mxCreateNumericArray(2, scoremat_dims, mxSINGLE_CLASS, mxREAL);
  float* scoremat_out = (float*)mxGetData(plhs[1]);

  for (int i = 0; i < (n*num_classes); ++i) {
    scoremat_out[i] = scoremat[i];
  }

  delete[] est_label;
  delete[] scoremat;
  
}
