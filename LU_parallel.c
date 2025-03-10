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
void LU_Decomposition_Parallel_Column(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N, int num_threads) {
    omp_set_num_threads(num_threads);  
    double start = omp_get_wtime();  // Start time

    int k, j;  // Declare k and j before using them in parallel loops

    for (int i = 0; i < N; i++) {
        // Compute Lower Triangular Matrix (L) first (Dynamic Scheduling)
        #pragma omp parallel for private(k, j) shared(A, L, U, N) schedule(dynamic)
        for (k = i; k < N; k++) {
            if (i != k) {
                double sum = 0;
                for (j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];

                #pragma omp critical  // Ensuring only one thread modifies L[k][i] at a time
                {
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }

        // Compute Upper Triangular Matrix (U) (Static Scheduling)
        #pragma omp parallel for private(k, j) shared(A, L, U, N) schedule(static)
        for (k = i; k < N; k++) {
            double sum = 0;
            for (j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];

            #pragma omp critical  // Ensuring safe write to U[i][k]
            {
                U[i][k] = A[i][k] - sum;
            }
        }
    }

    double end = omp_get_wtime();  // End time
    printf("Column-Major LU Execution Time (%d threads): %.6f sec\n", num_threads, end - start);
}

// **Parallel LU Decomposition (Row-Major)**
void LU_Decomposition_Parallel_Row(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N, int num_threads) {
    omp_set_num_threads(num_threads);  
    double start = omp_get_wtime();  // Start time

    int k, j;  // Declare k and j before using them in parallel loops

    for (int i = 0; i < N; i++) {
        // Compute Upper Triangular Matrix (U) (Dynamic Scheduling)
        #pragma omp parallel for private(k, j) shared(A, L, U, N) schedule(dynamic)
        for (k = i; k < N; k++) {
            double sum = 0;
            for (j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];

            #pragma omp critical  // Ensuring safe write to U[i][k]
            {
                U[i][k] = A[i][k] - sum;
            }
        }

        // Compute Lower Triangular Matrix (L) (Static Scheduling)
        #pragma omp parallel for private(k, j) shared(A, L, U, N) schedule(static)
        for (k = i; k < N; k++) {
            if (i != k) {
                double sum = 0;
                for (j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];

                #pragma omp critical  // Ensuring only one thread modifies L[k][i] at a time
                {
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }
    }

    double end = omp_get_wtime();  // End time
    printf("Row-Major LU Execution Time (%d threads): %.6f sec\n", num_threads, end - start);
}

int main() {
    int sizes[] = {512, 1024, 2000};  // Different matrix sizes to test
    int num_threads[] = {1, 4, 8};    // Fixed thread counts

    for (int t = 0; t < 3; t++) {  // Loop over different thread counts
        omp_set_num_threads(num_threads[t]);  
        printf("\n--- Testing with %d Threads ---\n", num_threads[t]);

        for (int s = 0; s < 3; s++) {
            int N = sizes[s];
            printf("\nMatrix Size: %d x %d\n", N, N);

            static double A[MAX_SIZE][MAX_SIZE], L[MAX_SIZE][MAX_SIZE], U[MAX_SIZE][MAX_SIZE];
            initialize_matrix(A, N);

            LU_Decomposition_Parallel_Column(A, L, U, N, num_threads[t]);
            LU_Decomposition_Parallel_Row(A, L, U, N, num_threads[t]);
        }
    }

    return 0;
}
