#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>
#include <string.h>

typedef struct {
  int num_rows;
  int num_cols;
  int nnz;
  int *rows;
  int *cols;
  double *values;
} coo_matrix;

coo_matrix coo_from_dense(double *mat, int m, int n) {
  int nnz = 0;
  for (int i = 0; i < m * n; i++) {
    if (mat[i] != 0) nnz++;
  }

  coo_matrix M;
  M.num_rows = m;
  M.num_cols = n;
  M.nnz = nnz;
  M.rows = (int*) malloc(nnz * sizeof(int));
  M.cols = (int*) malloc(nnz * sizeof(int));
  M.values = (double*) malloc(nnz * sizeof(double));

  int ctr = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (mat[i * n + j] != 0) {
        M.rows[ctr] = i;
        M.cols[ctr] = j;
        M.values[ctr] = mat[i * n + j];
        ctr++;
      }
    }
  }

  return M;
}

void coo_matrix_mult(coo_matrix A, coo_matrix B, coo_matrix *C) {
  int m = A.num_rows;
  int n = B.num_cols;
  int k = A.num_cols;

  C->num_rows = m;
  C->num_cols = n;
  int c_size = m * n;
  double *c_dense = (double*) calloc(c_size, sizeof(double));

  clock_t start = clock();

  #pragma omp parallel for
  for (int i = 0; i < A.nnz; i++) {
    int row = A.rows[i];
    int col = A.cols[i];
    double val = A.values[i];
    for (int j = 0; j < B.nnz; j++) {
      if (B.rows[j] == col) {
        int b_col = B.cols[j];
        double b_val = B.values[j];
        #pragma omp atomic
        c_dense[row * n + b_col] += val * b_val;
      }
    }
  }

  clock_t end = clock();
  double mult_time = ((double) (end - start)) / CLOCKS_PER_SEC;

  int c_nnz = 0;
  for (int i = 0; i < c_size; i++) {
    if (c_dense[i] != 0) c_nnz++;
  }

  C->nnz = c_nnz;
  C->rows = (int*) malloc(c_nnz * sizeof(int));
  C->cols = (int*) malloc(c_nnz * sizeof(int));
  C->values = (double*) malloc(c_nnz * sizeof(double));

  int ctr = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (c_dense[i * n + j] != 0) {
        C->rows[ctr] = i;
        C->cols[ctr] = j;
        C->values[ctr] = c_dense[i * n + j];
        ctr++;
      }
    }
  }

  free(c_dense);

  int a_nonzeros = A.nnz;
  int b_nonzeros = B.nnz;
  int c_nonzeros = C->nnz;

  double a_density = 100.0 * a_nonzeros / (A.num_rows * A.num_cols);
  double b_density = 100.0 * b_nonzeros / (B.num_rows * B.num_cols);
  double c_density = 100.0 * c_nonzeros / (C->num_rows * C->num_cols);

  int flops = 2 * c_nonzeros;
  double gflops = flops / mult_time / 1e9;
  double total_mem = (a_nonzeros + b_nonzeros + c_nonzeros) * (sizeof(double) + 2 * sizeof(int));
  double peak_mem = (a_nonzeros + b_nonzeros + c_size) * sizeof(double) + (a_nonzeros + b_nonzeros) * 2 * sizeof(int);

  int num_threads = omp_get_max_threads();

  printf("Running with %d OpenMP threads\n", num_threads);
  printf("\n==================\n");
  printf("Profiling Results:\n");
  printf("==================\n");
  printf("Multiplication Time: %.4f seconds\n", mult_time);
  printf("Peak Memory Usage: %.2f MB\n", peak_mem / 1024 / 1024);
  printf("Total FLOPS: %.2e\n", (double)flops);
  printf("Performance: %.2f GFLOPS\n", gflops);
  printf("\n==================\n");
  printf("Matrix Statistics:\n");
  printf("==================\n");
  printf("Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)\n", A.num_rows, A.num_cols, a_nonzeros, a_density);
  printf("Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)\n", B.num_rows, B.num_cols, b_nonzeros, b_density);
  printf("Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)\n", C->num_rows, C->num_cols, c_nonzeros, c_density);
}

int main() {
  clock_t init_start = clock();

  int m = 1000;
  int n = 1000;
  double sparsity = 0.01;

  double *mat_a = (double*) malloc(m * n * sizeof(double));
  double *mat_b = (double*) malloc(m * n * sizeof(double));

  for (int i = 0; i < m * n; i++) {
    if ((double) rand() / RAND_MAX < sparsity) {
      mat_a[i] = (double) rand() / RAND_MAX;
      mat_b[i] = (double) rand() / RAND_MAX;
    } else {
      mat_a[i] = 0;
      mat_b[i] = 0;
    }
  }

  clock_t init_end = clock();
  double init_time = ((double) (init_end - init_start)) / CLOCKS_PER_SEC;

  clock_t start = clock();

  coo_matrix A = coo_from_dense(mat_a, m, n);
  coo_matrix B = coo_from_dense(mat_b, m, n);
  coo_matrix C;

  coo_matrix_mult(A, B, &C);

  clock_t end = clock();
  double total_time = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Initialization Time: %.4f seconds\n", init_time);
  printf("\nTotal Time: %.4f seconds\n", total_time);

  free(A.rows); free(A.cols); free(A.values);
  free(B.rows); free(B.cols); free(B.values);
  free(C.rows); free(C.cols); free(C.values);

  return 0;
}
