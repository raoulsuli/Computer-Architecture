/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

/*
 * Add your optimized implementation here
 */
 
double* my_solver(int N, double *A, double* B) {
	register double *AB = calloc(N * N, sizeof(double));
	register double *ABB_ = calloc(N * N, sizeof(double));
	register double *A_A = calloc(N * N, sizeof(double));
	register double *A_tr = calloc(N * N, sizeof(double));
	register double *B_tr = calloc(N * N, sizeof(double));
	register double *C = calloc(N * N, sizeof(double));
	register int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A_tr[i * N + j] = A[j * N + i];
			B_tr[i * N + j] = B[j * N + i];
		}
	}

	for (i = 0; i < N; i++) { // A x B
        for (j = 0; j < N; j++) {
			register double sum = 0.0;
            for (k = i; k < N; k++) {
                sum += A[i * N + k] * B_tr[j * N + k];
            }
			AB[i * N + j] = sum;
        }
    }

	for (i = 0; i < N; i++) { // A x B x B_transpus
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			for (k = 0; k < N; k++) {
				sum += AB[i * N + k] * B[j * N + k];
			}
			ABB_[i * N + j] = sum;
		}
	}

	for (i = 0; i < N; i++) { // A_transpus * A
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			for (k = 0; k <= i; k++) {
				sum += A_tr[i * N + k] * A_tr[j * N + k];
			}
			A_A[i * N + j] = sum;
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
	free(A_tr);
	free(B_tr);
	return C;
}
