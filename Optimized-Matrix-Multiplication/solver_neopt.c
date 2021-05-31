/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
	double *AB = calloc(N * N, sizeof(double));
	double *ABB_ = calloc(N * N, sizeof(double));
	double *A_A = calloc(N * N, sizeof(double));
	double *C = calloc(N * N, sizeof(double));
	int i, j, k;

	for (i = 0; i < N; i++) { // A x B
        for (j = 0; j < N; j++) {
            for (k = i; k < N; k++) {
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

	for (i = 0; i < N; i++) { // A x B x B_transpus
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABB_[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	for (i = 0; i < N; i++) { // A_transpus * A
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				A_A[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) { // A x B x  B_transpus + A_transpus * A
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_[i * N + j] + A_A[i * N + j]; 
		}
	}

	free(AB);
	free(ABB_);
	free(A_A);
	return C;
}
