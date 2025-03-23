#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SIZE 512  // Maximum matrix size
#define CHUNK_SIZE 500 // Processing chunk size
#define NUM_THREADS 8  // Set to 4 threads

// Function to initialize matrix with random values
void initialize_matrix(double A[MAX_SIZE][MAX_SIZE], int N) {
    srand(42);  // Fixed seed for consistency
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (rand() % 100) + 1;  // Values between 1 and 100
}

// **Parallel LU Decomposition (Column-Major) with Static Scheduling and Chunks**
void LU_Decomposition_Static_Column(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N) {
    omp_set_num_threads(NUM_THREADS);
    double start = omp_get_wtime();  // Start time

    for (int i = 0; i < N; i += CHUNK_SIZE) {
        int end_chunk = (i + CHUNK_SIZE > N) ? N : (i + CHUNK_SIZE);

        // Compute Lower Triangular Matrix (L) first
        #pragma omp parallel for schedule(static)
        for (int k = i; k < end_chunk; k++) {
            if (i != k) {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }

        // Compute Upper Triangular Matrix (U)
        #pragma omp parallel for schedule(static)
        for (int k = i; k < end_chunk; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }
    }

    double end = omp_get_wtime();  // End time
    printf("Column-Major LU Execution Time (Static, %d threads, Chunk %d): %.6f sec\n", NUM_THREADS, CHUNK_SIZE, end - start);
}

// **Parallel LU Decomposition (Row-Major) with Static Scheduling and Chunks**
void LU_Decomposition_Static_Row(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N) {
    omp_set_num_threads(NUM_THREADS);
    double start = omp_get_wtime();  // Start time

    for (int i = 0; i < N; i += CHUNK_SIZE) {
        int end_chunk = (i + CHUNK_SIZE > N) ? N : (i + CHUNK_SIZE);

        // Compute Upper Triangular Matrix (U) first
        #pragma omp parallel for schedule(static)
        for (int k = i; k < end_chunk; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }

        // Compute Lower Triangular Matrix (L)
        #pragma omp parallel for schedule(static)
        for (int k = i; k < end_chunk; k++) {
            if (i != k) {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }

    double end = omp_get_wtime();  // End time
    printf("Row-Major LU Execution Time (Static, %d threads, Chunk %d): %.6f sec\n", NUM_THREADS, CHUNK_SIZE, end - start);
}

int main() {
    int N = 512;  
    printf("\n--- Running with %d Threads, Matrix Size: %d x %d, Chunk Size: %d ---\n", NUM_THREADS, N, N, CHUNK_SIZE);
    
    static double A[MAX_SIZE][MAX_SIZE], L[MAX_SIZE][MAX_SIZE], U[MAX_SIZE][MAX_SIZE];
    initialize_matrix(A, N);
    
    LU_Decomposition_Static_Column(A, L, U, N);
    LU_Decomposition_Static_Row(A, L, U, N);

    return 0;
}
