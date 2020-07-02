#include <iostream>
#include <Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixType;
typedef Map<MatrixType> MapType;


MatrixType* lower_to_full(double* clow, int n, int m) {


  MatrixType* full = new MatrixType[m];

  for(int i=0; i<m; i++) {
    full[i].resize(n,n);
    for(int r=0; r<n; r++) {
      for(int c=0; c<=r; c++) {
	full[i](r, c) = full[i](c, r) = *clow++;
      }
    }
  }

  return full;
}


MatrixType* inverse(MatrixType* c, int m) {
  MatrixType* inv = new MatrixType[m];

  #pragma omp parallel for 
  for(int i=0; i<m; i++) {
    inv[i] = c[i].inverse();
  }

  return inv;
}

double* log_determinant(MatrixType* c, int m) {
  double* logd = new double[m];

  #pragma omp parallel for 
  for(int i=0; i<m; i++) {
    logd[i] = log(c[i].determinant());
  }

  return logd;
}

inline double identity(double x) {
  return x;
}

template <double _f (double) = identity> 
void base_skl(double* mean1, int mean1_n, int mean1_m,
	      double* mean2, int mean2_n, int mean2_m,
	      double* cov1, int cov1_n, int cov1_m,
	      double* cov2, int cov2_n, int cov2_m,
	      double* divergence, int n, int m
	      ) {

  
  MapType m1 = Map<MatrixType >(mean1, mean1_n, mean1_m);
  MapType m2 = Map<MatrixType >(mean2, mean2_n, mean2_m);

  MatrixType * c1 = lower_to_full(cov1, mean1_m, mean1_n);
  MatrixType * c2 = lower_to_full(cov2, mean2_m, mean2_n);
  
  MatrixType * ic1 = inverse(c1, mean1_n);
  MatrixType * ic2 = inverse(c2, mean2_n);

  #pragma omp parallel for 
  for(int r=0; r<n; r++) {
    double* dv = divergence + m*r;
    for(int c=0; c<m; c++) {
      VectorXd dmu = m1.row(r) - m2.row(c);
      auto icov = ic1[r] +  ic2[c];
      *dv++ = _f((ic1[r].array() * c2[c].array()).sum() + (c1[r].array()* ic2[c].array()).sum() + dmu.dot(icov * dmu) - 2.0*mean1_m);   
      }
  }
  
  delete[](c1);
  delete[](c2);
  delete[](ic1);
  delete[](ic2);
  
}


template <double _f (double) = identity> 
void base_js(double* mean1, int mean1_n, int mean1_m,
	      double* mean2, int mean2_n, int mean2_m,
	      double* cov1, int cov1_n, int cov1_m,
	      double* cov2, int cov2_n, int cov2_m,
	      double* divergence, int n, int m
	      ) {

  
  MapType m1 = Map<MatrixType >(mean1, mean1_n, mean1_m);
  MapType m2 = Map<MatrixType >(mean2, mean2_n, mean2_m);

  MatrixType * c1 = lower_to_full(cov1, mean1_m, mean1_n);
  MatrixType * c2 = lower_to_full(cov2, mean2_m, mean2_n);

  double * logd1 = log_determinant(c1, mean1_n);
  double * logd2 = log_determinant(c2, mean2_n);

  #pragma omp parallel for 
  for(int r=0; r<n; r++) {
    double* dv = divergence + m*r;
    for(int c=0; c<m; c++) {
      MatrixType sigma_m =  0.5*(c1[r] + c2[c] + m1.row(r).transpose()*m1.row(r) + m2.row(c).transpose()* m2.row(c))
	- 0.25*(m1.row(r) + m2.row(c)).transpose() * (m1.row(r) + m2.row(c));
      *dv++ = 2.0*log(sigma_m.determinant()) - logd1[r] - logd2[c];
      }
  }
  
  delete[](c1);
  delete[](c2);
  delete[](logd1);
  delete[](logd2);
  
}

void skl(double* mean1, int mean1_n, int mean1_m,
	 double* mean2, int mean2_n, int mean2_m,
	 double* cov1, int cov1_n, int cov1_m,
	 double* cov2, int cov2_n, int cov2_m,
	 double* divergence, int n, int m
	 ) {
  
  return base_skl(mean1, mean1_n, mean1_m,
		  mean2, mean2_n, mean2_m,
		  cov1, cov1_n, cov1_m,
		  cov2, cov2_n, cov2_m,
		  divergence, n, m);
		  
}

void sqrt_skl(double* mean1, int mean1_n, int mean1_m,
	      double* mean2, int mean2_n, int mean2_m,
	      double* cov1, int cov1_n, int cov1_m,
	      double* cov2, int cov2_n, int cov2_m,
	      double* divergence, int n, int m
	      ) {
  
  return base_skl<sqrt>(mean1, mean1_n, mean1_m,
			mean2, mean2_n, mean2_m,
			cov1, cov1_n, cov1_m,
			cov2, cov2_n, cov2_m,
			divergence, n, m);
		  
}


void log_skl(double* mean1, int mean1_n, int mean1_m,
	     double* mean2, int mean2_n, int mean2_m,
	     double* cov1, int cov1_n, int cov1_m,
	     double* cov2, int cov2_n, int cov2_m,
	     double* divergence, int n, int m
	     ) {
  
  return base_skl<log>(mean1, mean1_n, mean1_m,
		       mean2, mean2_n, mean2_m,
		       cov1, cov1_n, cov1_m,
		       cov2, cov2_n, cov2_m,
		       divergence, n, m);
		  
}


void js(double* mean1, int mean1_n, int mean1_m,
	double* mean2, int mean2_n, int mean2_m,
	double* cov1, int cov1_n, int cov1_m,
	double* cov2, int cov2_n, int cov2_m,
	double* divergence, int n, int m
	) {
  
  return base_js(mean1, mean1_n, mean1_m,
		 mean2, mean2_n, mean2_m,
		 cov1, cov1_n, cov1_m,
		 cov2, cov2_n, cov2_m,
		 divergence, n, m);
  
}

void sqrt_js(double* mean1, int mean1_n, int mean1_m,
	     double* mean2, int mean2_n, int mean2_m,
	     double* cov1, int cov1_n, int cov1_m,
	     double* cov2, int cov2_n, int cov2_m,
	     double* divergence, int n, int m
	     ) {
  
  return base_js<sqrt>(mean1, mean1_n, mean1_m,
			mean2, mean2_n, mean2_m,
			cov1, cov1_n, cov1_m,
			cov2, cov2_n, cov2_m,
			divergence, n, m);
		  
}
