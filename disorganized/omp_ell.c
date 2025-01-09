#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

typedef struct {
  int num_rows;
  int num_cols;
  int max_nonzeros_per_row;
  int *cols;
  double *values;
} ell_matrix;

ell_matrix ell_from_dense(double *mat, int m, int n) {
  int max_nnz_per_row = 0;
  for (int i = 0; i < m; i++) {
    int nnz_in_row = 0;
    for (int j = 0; j < n; j++) {
      if (mat[i * n + j] != 0) {
        nnz_in_row++;
      }
    }
    if (nnz_in_row > max_nnz_per_row) {
      max_nnz_per_row = nnz_in_row;
    }
  }

  ell_matrix M;
  M.num_rows = m;
  M.num_cols = n;
  M.max_nonzeros_per_row = max_nnz_per_row;
  M.cols = (int*) malloc(m * max_nnz_per_row * sizeof(int));
  M.values = (double*) malloc(m * max_nnz_per_row * sizeof(double));

  memset(M.cols, 0, m * max_nnz_per_row * sizeof(int));
  memset(M.values, 0, m * max_nnz_per_row * sizeof(double));

  for (int i = 0; i < m; i++) {
    int ctr = 0;
    for (int j = 0; j < n; j++) {
      if (mat[i * n + j] != 0) {
        M.cols[i * max_nnz_per_row + ctr] = j;
        M.values[i * max_nnz_per_row + ctr] = mat[i * n + j];
        ctr++;
      }
    }
  }

  return M;
}

void ell_matrix_mult(ell_matrix A, ell_matrix B, ell_matrix *C) {
  int m = A.num_rows;
  int n = B.num_cols;
  int k = A.num_cols;

  C->num_rows = m;
  C->num_cols = n;
  C->max_nonzeros_per_row = A.max_nonzeros_per_row * B.max_nonzeros_per_row;
  C->cols = (int*) malloc(m * C->max_nonzeros_per_row * sizeof(int));
  C->values = (double*) calloc(m * C->max_nonzeros_per_row, sizeof(double));

  clock_t start = clock();

  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < A.max_nonzeros_per_row; j++) {
      int a_col = A.cols[i * A.max_nonzeros_per_row + j];
      double a_val = A.values[i * A.max_nonzeros_per_row + j];
      if (a_val != 0) {
        for (int k = 0; k < B.max_nonzeros_per_row; k++) {
          int b_col = B.cols[a_col * B.max_nonzeros_per_row + k];
          double b_val = B.values[a_col * B.max_nonzeros_per_row + k];
          if (b_val != 0) {
            #pragma omp atomic
            C->values[i * C->max_nonzeros_per_row + b_col] += a_val * b_val;
            C->cols[i * C->max_nonzeros_per_row + b_col] = b_col;
          }
        }
      }
    }
  }

  clock_t end = clock();
  double mult_time = ((double) (end - start)) / CLOCKS_PER_SEC;

  // profile and print results
  int a_nonzeros = 0, b_nonzeros = 0, c_nonzeros = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < A.max_nonzeros_per_row; j++) {
      if (A.values[i * A.max_nonzeros_per_row + j] != 0) {
        a_nonzeros++;
      }
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < B.max_nonzeros_per_row; j++) {
      if (B.values[i * B.max_nonzeros_per_row + j] != 0) {
        b_nonzeros++;
      }
    }
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < C->max_nonzeros_per_row; j++) {
      if (C->values[i * C->max_nonzeros_per_row + j] != 0) {
        c_nonzeros++;
      }
    }
  }

  double a_density = 100.0 * a_nonzeros / (A.num_rows * A.num_cols);
  double b_density = 100.0 * b_nonzeros / (B.num_rows * B.num_cols);
  double c_density = 100.0 * c_nonzeros / (C->num_rows * C->num_cols);

  int flops = 2 * c_nonzeros;
  double gflops = flops / mult_time / 1e9;
  double total_mem = (A.num_rows * A.max_nonzeros_per_row + B.num_rows * B.max_nonzeros_per_row +
    C->num_rows * C->max_nonzeros_per_row) * (sizeof(double) + sizeof(int));
  double peak_mem = total_mem;

  int num_threads = omp_get_max_threads();

  printf("Running with %d OpenMP threads\n", num_threads);
  printf("/n==================\n");
  printf("Profiling Results:\n");
  printf("==================\n");
  printf("Multiplication Time: %.4f seconds\n", mult_time);
  printf("Peak Memory Usage: %.2f MB\n", peak_mem / 1024 / 1024);
  printf("Total FLOPS: %.2e\n", (double)flops);
  printf("Performance: %.2f GFLOPS\n", gflops);
  printf("\n=================\n");
  printf("Matrix Statistics:\n");
  printf("==================\n");
  printf("Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)\n",
         A.num_rows, A.num_cols, a_nonzeros, a_density);
  printf("Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)\n",
         B.num_rows, B.num_cols, b_nonzeros, b_density);
  printf("Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)\n",
         C->num_rows, C->num_cols, c_nonzeros, c_density);
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

  ell_matrix A = ell_from_dense(mat_a, m, n);
  ell_matrix B = ell_from_dense(mat_b, m, n);
  ell_matrix C;

  ell_matrix_mult(A, B, &C);

  clock_t end = clock();
  double total_time = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Initialization Time: %.4f seconds\n", init_time);
  printf("\nTotal Time: %.4f seconds\n", total_time);

  free(A.cols); free(A.values);
  free(B.cols); free(B.values);
  free(C.cols); free(C.values);

  return 0;
}
