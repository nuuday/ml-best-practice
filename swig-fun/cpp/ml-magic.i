%module divergence
%{
    #define SWIG_FILE_WITH_INIT
/* Includes the header in the wrapper code */
#include "divergence.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* mean1, int mean1_n, int mean1_m)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* mean2, int mean2_n, int mean2_m)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* cov1, int cov1_n, int cov1_m)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* cov2, int cov2_n, int cov2_m)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* divergence, int n, int m)}; 



/* Parse the header file to generate wrappers */
%include "divergence.h"
