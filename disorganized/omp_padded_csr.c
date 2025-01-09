#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define PAD_SIZE 32

typedef struct {
  int num_rows;
  int num_cols;
  int *row_ptr;
  int *col_ind;
  double *val;
} sparse_matrix;

void sparse_matrix_mult(sparse_matrix A, sparse_matrix B, sparse_matrix *C) {
  int m = A.num_rows;
  int n = B.num_cols;

  int *row_counts = (int*) calloc(m, sizeof(int));
  C->num_rows = m;
  C->num_cols = n;
  C->row_ptr = (int*) malloc((m+1) * sizeof(int));

  clock_t start = clock();

  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; j++) {
      int col = A.col_ind[j];
      double A_val = A.val[j];
      for (int k = B.row_ptr[col]; k < B.row_ptr[col+1]; k++) {
        #pragma omp atomic
        row_counts[i]++;
      }
    }
  }

  for (int i = 0; i < m; i++) {
    C->row_ptr[i+1] = C->row_ptr[i] + row_counts[i];
  }

  int padded_nnz = C->row_ptr[m];
  int padding = (PAD_SIZE - (padded_nnz % PAD_SIZE)) % PAD_SIZE;
  padded_nnz += padding;

  C->col_ind = (int*) aligned_alloc(PAD_SIZE, padded_nnz * sizeof(int));
  C->val = (double*) aligned_alloc(PAD_SIZE, padded_nnz * sizeof(double));

  memset(C->col_ind + C->row_ptr[m], 0, padding * sizeof(int));
  memset(C->val + C->row_ptr[m], 0, padding * sizeof(double));

  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    int row_start = C->row_ptr[i];
    int row_end = C->row_ptr[i+1];
    int counter = row_start;

    for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; j++) {
      int col = A.col_ind[j];
      double A_val = A.val[j];
      for (int k = B.row_ptr[col]; k < B.row_ptr[col+1]; k++) {
        int B_col = B.col_ind[k];
        double B_val = B.val[k];

        int l = row_start;
        while (l < counter && C->col_ind[l] < B_col) l++;

        if (l == counter) {
          C->col_ind[counter] = B_col;
          C->val[counter] = A_val * B_val;
          counter++;
        } else {
          #pragma omp atomic
          C->val[l] += A_val * B_val;
        }
      }
    }
  }

  clock_t end = clock();
  double mult_time = ((double) (end - start)) / CLOCKS_PER_SEC;

  int a_nonzeros = A.row_ptr[A.num_rows];
  int b_nonzeros = B.row_ptr[B.num_rows];
  int c_nonzeros = C->row_ptr[C->num_rows];

  double a_density = 100.0 * a_nonzeros / (A.num_rows * A.num_cols);
  double b_density = 100.0 * b_nonzeros / (B.num_rows * B.num_cols);
  double c_density = 100.0 * c_nonzeros / (C->num_rows * C->num_cols);

  int flops = 2 * c_nonzeros;
  double gflops = flops / mult_time / 1e9;
  double total_mem = (a_nonzeros + b_nonzeros + padded_nnz) * (sizeof(double) + sizeof(int));

  int num_threads = omp_get_max_threads();

  printf("Running with %d OpenMP threads\n\n", num_threads);
  printf("Profiling Results:\n");
  printf("==================\n");
  printf("Multiplication time: %f seconds\n", mult_time);
  printf("Memory usage: %.2f MB\n", total_mem / 1024 / 1024);
  printf("Total FLOPS: %d\n", flops);
  printf("GFLOPS: %.2f\n", gflops);
  printf("Matrix Statistics:\n");
  printf("Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)\n", A.num_rows, A.num_cols, a_nonzeros, a_density);
  printf("Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)\n", B.num_rows, B.num_cols, b_nonzeros, b_density);
  printf("Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)\n", C->num_rows, C->num_cols, c_nonzeros, c_density);

  free(row_counts);
}

sparse_matrix padded_csr_from_dense(double *mat, int m, int n) {
  int nnz = 0;
  for (int i = 0; i < m*n; i++) {
    if (mat[i] != 0) nnz++;
  }

  sparse_matrix M;
  M.num_rows = m;
  M.num_cols = n;
  M.row_ptr = (int*) malloc((m+1) * sizeof(int));

  int padded_nnz = nnz;
  int padding = (PAD_SIZE - (nnz % PAD_SIZE)) % PAD_SIZE;
  padded_nnz += padding;

  M.col_ind = (int*) aligned_alloc(PAD_SIZE, padded_nnz * sizeof(int));
  M.val = (double*) aligned_alloc(PAD_SIZE, padded_nnz * sizeof(double));

  int ctr = 0;
  for (int i = 0; i < m; i++) {
    M.row_ptr[i] = ctr;
    for (int j = 0; j < n; j++) {
      if (mat[i*n + j] != 0) {
        M.col_ind[ctr] = j;
        M.val[ctr] = mat[i*n + j];
        ctr++;
      }
    }
  }
  M.row_ptr[m] = nnz;

  memset(M.col_ind + nnz, 0, padding * sizeof(int));
  memset(M.val + nnz, 0, padding * sizeof(double));

  return M;
}

int main() {
  clock_t init_start = clock();

  int m = 1000;
  int n = 1000;
  double sparsity = 0.01;

  double *mat_a = (double*) malloc(m * n * sizeof(double));
  double *mat_b = (double*) malloc(m * n * sizeof(double));

  for (int i = 0; i < m*n; i++) {
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

  sparse_matrix A = padded_csr_from_dense(mat_a, m, n);
  sparse_matrix B = padded_csr_from_dense(mat_b, m, n);
  sparse_matrix C;

  sparse_matrix_mult(A, B, &C);

  clock_t end = clock();
  double total_time = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Initialization time: %f seconds\n", init_time);
  printf("\nTotal time: %f seconds\n", total_time);

  free(A.row_ptr); free(A.col_ind); free(A.val);
  free(B.row_ptr); free(B.col_ind); free(B.val);
  free(C.row_ptr); free(C.col_ind); free(C.val);

  return 0;
}
