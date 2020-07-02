
void skl(double* mean1, int mean1_n, int mean1_m,
	 double* mean2, int mean2_n, int mean2_m,
	 double* cov1, int cov1_n, int cov1_m,
	 double* cov2, int cov2_n, int cov2_m,
	 double* divergence, int n, int m
	 );

void sqrt_skl(double* mean1, int mean1_n, int mean1_m,
	      double* mean2, int mean2_n, int mean2_m,
	      double* cov1, int cov1_n, int cov1_m,
	      double* cov2, int cov2_n, int cov2_m,
	      double* divergence, int n, int m
	      );


void log_skl(double* mean1, int mean1_n, int mean1_m,
	     double* mean2, int mean2_n, int mean2_m,
	     double* cov1, int cov1_n, int cov1_m,
	     double* cov2, int cov2_n, int cov2_m,
	     double* divergence, int n, int m
	     );


void js(double* mean1, int mean1_n, int mean1_m,
	 double* mean2, int mean2_n, int mean2_m,
	 double* cov1, int cov1_n, int cov1_m,
	 double* cov2, int cov2_n, int cov2_m,
	 double* divergence, int n, int m
	 );

void sqrt_js(double* mean1, int mean1_n, int mean1_m,
	     double* mean2, int mean2_n, int mean2_m,
	     double* cov1, int cov1_n, int cov1_m,
	     double* cov2, int cov2_n, int cov2_m,
	     double* divergence, int n, int m
	     );
