#include <iostream>
#include <Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixType;
typedef Map<MatrixType> MapType;

void inplace_element_mult1(double* arg1, int arg1_n, int arg1_m,
                           double* arg2, int arg2_n, int arg2_m
                           double* res, int n, int m
              ) {


  
  #pragma omp parallel for 
  for(int r=0; r<n; r++) {
    double* arg1_r = arg1 + m*r;
    double* arg2_r = arg2 + m*r;
    double* res_r = res + m*r
    for(int c=0; c<m; c++) {
      *res_r++ =  arg1_r++ * arg2_r++;
    }
  }
  
}

void inplace_element_mult2(double* arg1, int arg1_n, int arg1_m,
			   double* arg2, int arg2_n, int arg2_m
			   double* res, int n, int m
	      ) {

  
  MapType m1 = Map<MatrixType >(arg1, arg1_n, arg1_m);
  MapType m2 = Map<MatrixType >(arg2, arg2_n, arg2_m);

  MatrixType m3 = m1.array()*m2.array();

  double m3_data = m3.data()
  for(int r<n*m; r++){
    *res++ = m3_data++;
  }  
}

