%module ml_magic
%{
    #define SWIG_FILE_WITH_INIT
/* Includes the header in the wrapper code */
#include "ml-magic.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* arg1, int arg1_n, int arg1_m)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* arg2, int arg2_n, int arg2_m)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* res, int n, int m)}; 

/* Parse the header file to generate wrappers */
%include "ml-magic.h"
