#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include "sparse_common_3.h"

typedef struct {
  int rows;
  int cols;
  int nnz;
  double *values;
  int *col_indices;
  int *row_ptr;
} CSRMatrix;

CSRMatrix* create_csr_matrix(int rows, int cols, int nnz) {
  CSRMatrix *mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
  mat->rows = rows;
  mat->cols = cols;
  mat->nnz = nnz;

  mat->values = (double*)calloc(nnz, sizeof(double));
  mat->col_indices = (int*)calloc(nnz, sizeof(int));
  mat->row_ptr = (int*)calloc(rows + 1, sizeof(int));

  return mat;
}

void free_csr_matrix(CSRMatrix *mat) {
  free(mat->values);
  free(mat->col_indices);
  free(mat->row_ptr);
  free(mat);
}

CSRMatrix* multiply_sparse_matrices(CSRMatrix *A, CSRMatrix *B, Profile *profile) {
  if (A->cols != B->rows) {
    printf("Error: Incompatible matrix dimensions\n");
    return NULL;
  }

  profile_start(profile);

  profile->rows_a = A->rows;
  profile->cols_a = A->cols;
  profile->nnz_a = A->nnz;
  profile->density_a = (double)A->nnz / (A->rows * A->cols);

  profile->rows_b = B->rows;
  profile->cols_b = B->cols;
  profile->nnz_b = B->nnz;
  profile->density_b = (double)B->nnz / (B->rows * B->cols);

  int max_nnz = A->nnz * B->nnz;
  CSRMatrix *C = create_csr_matrix(A->rows, B->cols, max_nnz);

  double *temp_row = (double*)calloc(B->cols, sizeof(double));
  int *nnz_per_row = (int*)calloc(A->rows, sizeof(int));

  profile->init_time = (get_time_usec() - start_time) / 1e6;
  start_time = get_time_usec();

  int current_nnz = 0;
  C->row_ptr[0] = 0;

  #pragma omp parallel for reduction(+:current_nnz)
  for (int i = 0; i < A->rows; i++) {
    memset(temp_row, 0, B->cols * sizeof(double));

    for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
      int col_a = A->col_indices[j];
      double val_a = A->values[j];

      for (int k = B->row_ptr[col_a]; k < B->row_ptr[col_a + 1]; k++) {
        int col_b = B->col_indices[k];
        double val_b = B->values[k];
        temp_row[col_b] += val_a * val_b;
        profile->total_flops += 2;
      }
    }

    for (int j = 0; j < B->cols; j++) {
      if (temp_row[j] != 0) {
        C->values[current_nnz] = temp_row[j];
        C->col_indices[current_nnz] = j;
        current_nnz++;
      }
    }
    C->row_ptr[i + 1] = current_nnz;
  }

  profile->mult_time = (get_time_usec() - start_time) / 1e6;

  C->nnz = current_nnz;
  C->values = (double*)realloc(C->values, current_nnz * sizeof(double));
  C->col_indices = (int*)realloc(C->col_indices, current_nnz * sizeof(int));

  profile->memory_usage = (current_nnz * (sizeof(double) + sizeof(int)) +
    (C->rows + 1) * sizeof(int)) / (1024.0 * 1024.0); // in MB

  free(temp_row);
  free(nnz_per_row);

  return C;
}

CSRMatrix* generate_random_sparse_matrix(int rows, int cols, double density) {
  int nnz = (int)(rows * cols * density);
  CSRMatrix *mat = create_csr_matrix(rows, cols, nnz);

  int *positions = (int*)malloc(rows * cols * sizeof(int));
  for (int i = 0; i < rows * cols; i++) {
    positions[i] = i;
  }

  for (int i = rows * cols - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int temp = positions[i];
    positions[i] = positions[j];
    positions[j] = temp;
  }

  for (int i = 0; i < nnz; i++) {
    int row = positions[i] / cols;
    int col = positions[i] % cols;
    mat->values[i] = (double)rand() / RAND_MAX;
    mat->col_indices[i] = col;
    mat->row_ptr[row + 1]++;
  }

  for (int i = 1; i <= rows; i++) {
    mat->row_ptr[i] += mat->row_ptr[i - 1];
  }

  free(positions);
  return mat;
}

int main() {
  srand(time(NULL));

  int size = 1000;
  double density = 0.01;

  CSRMatrix *A = generate_random_sparse_matrix(size, size, density);
  CSRMatrix *B = generate_random_sparse_matrix(size, size, density);

  ProfileData prof = {0};

  CSRMatrix *C = multiply_sparse_matrices(A, B, &prof);

  print_profile_print(&profile, stdout);

  free_csr_matrix(A);
  free_csr_matrix(B);
  free_csr_matrix(C);

  return 0;
}
