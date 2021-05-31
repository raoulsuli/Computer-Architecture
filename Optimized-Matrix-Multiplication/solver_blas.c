/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"
#include <string.h>
#include "cblas.h"

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
	double *AB = calloc(N * N, sizeof(double));
	double *C = calloc(N * N, sizeof(double));

	memcpy(AB, B, N * N * sizeof(double));
	memcpy(C, A, N * N * sizeof(double));
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
		CblasNonUnit, N, N, 1.0, A, N, AB, N);
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1.0, A, N, C, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);
	
	free(AB);
	return C;
}
