#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SIZE 2000  // Maximum matrix size

// Function to initialize matrix with random values
void initialize_matrix(double A[MAX_SIZE][MAX_SIZE], int N) {
    srand(42);  // Fixed seed for consistency
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (rand() % 100) + 1;  // Values between 1 and 100
}

// **Parallel LU Decomposition (Column-Major)**
void LU_Decomposition_Auto_Column(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N, int max_threads) {
    omp_set_num_threads(max_threads);  
    double start = omp_get_wtime();  // Start time

    for (int i = 0; i < N; i++) {
        // Compute Lower Triangular Matrix (L) first
        #pragma omp parallel for schedule(dynamic)
        for (int k = i; k < N; k++) {
            if (i != k) {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }

        // Compute Upper Triangular Matrix (U)
        #pragma omp parallel for schedule(dynamic)
        for (int k = i; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }
    }

    double end = omp_get_wtime();  // End time
    printf("Column-Major LU Execution Time (%d threads): %.6f sec\n", max_threads, end - start);
}

// **Parallel LU Decomposition (Row-Major)**
void LU_Decomposition_Auto_Row(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N, int max_threads) {
    omp_set_num_threads(max_threads);  
    double start = omp_get_wtime();  // Start time

    for (int i = 0; i < N; i++) {
        // Compute Upper Triangular Matrix (U) first
        #pragma omp parallel for schedule(dynamic)
        for (int k = i; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }

        // Compute Lower Triangular Matrix (L)
        #pragma omp parallel for schedule(dynamic)
        for (int k = i; k < N; k++) {
            if (i != k) {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }

    double end = omp_get_wtime();  // End time
    printf("Row-Major LU Execution Time (%d threads): %.6f sec\n", max_threads, end - start);
}

int main() {
    int sizes[] = {512, 1024, 2000};  // Different matrix sizes to test
    int max_threads = omp_get_max_threads();  // Detect total available CPU cores

    printf("\n--- Running with Auto-Detected %d Threads ---\n", max_threads);

    for (int s = 0; s < 3; s++) {
        int N = sizes[s];
        printf("\nMatrix Size: %d x %d\n", N, N);

        static double A[MAX_SIZE][MAX_SIZE], L[MAX_SIZE][MAX_SIZE], U[MAX_SIZE][MAX_SIZE];
        initialize_matrix(A, N);

        LU_Decomposition_Auto_Column(A, L, U, N, max_threads);
        LU_Decomposition_Auto_Row(A, L, U, N, max_threads);
    }

    return 0;
}
