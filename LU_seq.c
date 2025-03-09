#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SIZE 2000  // Maximum matrix size

// Function to initialize matrix with random values
void initialize_matrix(double A[MAX_SIZE][MAX_SIZE], int N) {
    srand(42);  // Fixed seed for consistency
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (rand() % 100) + 1;  // Values between 1 and 100
}

// **Sequential LU Decomposition (Column-Major)**
void LU_Decomposition_Sequential_Column(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N) {
    clock_t start = clock();  // Start time

    for (int i = 0; i < N; i++) {
        // Compute Lower Triangular Matrix (L) first
        for (int k = i; k < N; k++) {
            if (i != k) {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }

        // Compute Upper Triangular Matrix (U)
        for (int k = i; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }
    }

    clock_t end = clock();  // End time
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Column-Major LU Execution Time: %.6f sec\n", time_taken);
}

// **Sequential LU Decomposition (Row-Major)**
void LU_Decomposition_Sequential_Row(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], double U[MAX_SIZE][MAX_SIZE], int N) {
    clock_t start = clock();  // Start time

    for (int i = 0; i < N; i++) {
        // Compute Upper Triangular Matrix (U) first
        for (int k = i; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }

        // Compute Lower Triangular Matrix (L)
        for (int k = i; k < N; k++) {
            if (i != k) {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }

    clock_t end = clock();  // End time
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Row-Major LU Execution Time: %.6f sec\n", time_taken);
}

int main() {
    int sizes[] = {512, 1024, 2000};  // Different matrix sizes to test

    for (int s = 0; s < 3; s++) {
        int N = sizes[s];
        printf("\nMatrix Size: %d x %d\n", N, N);

        static double A[MAX_SIZE][MAX_SIZE], L[MAX_SIZE][MAX_SIZE], U[MAX_SIZE][MAX_SIZE];
        initialize_matrix(A, N);

        LU_Decomposition_Sequential_Column(A, L, U, N);
        LU_Decomposition_Sequential_Row(A, L, U, N);
    }

    return 0;
}
