#include <string.h>
#include "mex.h"
#include "matrix.h"
#include "../../../../../featpipec/src/classification/svm/liblinear.h"
#include "../../../class_handle.h"

// f(obj, command)
void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]) {

  // get string name of command from first input parameter
  char *command = mxArrayToString(prhs[1]);

  // processing

  if (strcmp(command,"init") == 0) {
    class_handle<featpipe::Liblinear>* solver =
      new class_handle<featpipe::Liblinear>;

    plhs[0] = mxDuplicateArray(prhs[0]);

    mxSetProperty(plhs[0],0,"c_handle",convertPtr2Mat(solver));
  } else if (strcmp(command,"clear") == 0) {
    class_handle<featpipe::Liblinear>* solver = 
      convertMat2Ptr<featpipe::Liblinear>(mxGetProperty(prhs[0],0,"c_handle"));

    if (solver) {
      delete solver;
    }
  } else {
    mexErrMsgTxt("Command not recognized");
  }
  
}
