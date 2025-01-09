#include <time.h>
#include "simple_profiler.h"
#include <math.h>

typedef struct {
  int rows;
  int cols;
  int nnz;
  double *values;
  int *col_indices;
  int *row_ptr;
} CSRMatrix;

CSRMatrix* create_random_sparse_matrix(int rows, int cols, double density);
void free_csr_matrix(CSRMatrix *matrix);
CSRMatrix* multiply_csr_matrices(CSRMatrix *A, CSRMatrix *B, SimpleProfile *profile);

CSRMatrix* create_random_sparse_matrix(int rows, int cols, double density) {
  int max_nnz = (int)(rows * cols * density);
  CSRMatrix *matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = max_nnz;

  matrix->values = (double*)malloc(max_nnz * sizeof(double));
  matrix->col_indices = (int*)malloc(max_nnz * sizeof(int));
  matrix->row_ptr = (int*)malloc((rows + 1) * sizeof(int));

  for (int i = 0; i <= rows; i++) {
    matrix->row_ptr[i] = (int)((i * (double)max_nnz) / rows);
  }

  srand(time(NULL));
  for (int i = 0; i < max_nnz; i++) {
    matrix->values[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // [-1, 1]
    matrix->col_indices[i] = rand() % cols;
  }

  return matrix;
}

void free_csr_matrix(CSRMatrix *matrix) {
  if (matrix) {
    free(matrix->values);
    free(matrix->col_indices);
    free(matrix->row_ptr);
    free(matrix);
  }
}

CSRMatrix* multiply_csr_matrices(CSRMatrix *A, CSRMatrix *B, SimpleProfile *profile) {
  profile->rows_a = A->rows;
  profile->cols_a = A->cols;
  profile->nnz_a = A->nnz;

  profile->rows_b = B->rows;
  profile->cols_b = B->cols;
  profile->nnz_b = B->nnz;

  profile_update_densities(profile);

  profile_start(profile);

  int max_nnz = (int)(A->nnz * B->nnz / B->rows);
  CSRMatrix *C = (CSRMatrix*)malloc(sizeof(CSRMatrix));
  C->rows = A->rows;
  C->cols = B->cols;
  C->values = (double*)malloc(max_nnz * sizeof(double));
  C->col_indices = (int*)malloc(max_nnz * sizeof(int));
  C->row_ptr = (int*)malloc((A->rows + 1) * sizeof(int));

  int actual_nnz = 0;
  C->row_ptr[0] = 0;

  #pragma omp parallel
  {
    #pragma omp single
    {
      profile->num_threads = omp_get_num_threads();
    }

    #pragma omp for schedule(dynamic, 32)
    for (int i = 0; i < A->rows; i++) {
      double *temp = (double*)calloc(B->cols, sizeof(double));
      int row_nnz = 0;

      for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
        int col = A->col_indices[j];
        for (int k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
          temp[B->col_indices[k]] += A->values[j] * B->values[k];
        }
      }

      for (int j = 0; j < B->cols; j++) {
        if (fabs(temp[j]) > 1e-10) {
          row_nnz++;
        }
      }

      #pragma omp critical
      {
        int pos = actual_nnz;
        actual_nnz += row_nnz;
        C->row_ptr[i + 1] = actual_nnz;

        for (int j = 0; j < B->cols; j++) {
          if (fabs(temp[j]) > 1e-10) {
            C->values[pos] = temp[j];
            C->col_indices[pos] = j;
            pos++;
          }
        }
      }

      free(temp);
    }
  }

  C->nnz = actual_nnz;
  profile->nnz_c = actual_nnz;
  profile->rows_c = C->rows;
  profile->cols_c = C->cols;
  profile_update_densities(profile);

  profile_stop(profile);
  profile->mult_time = profile->total_time;

  profile->total_flops = 2L * actual_nnz;
  profile_update_flops(profile);

  profile->bytes_transferred = (A->nnz + B->nnz + actual_nnz) *
    (sizeof(double) + sizeof(int));
  profile_collect_memory_metrics(profile);

  return C;
}

int main(int argc, char *argv[]) {
  SimpleProfile profile;
  profile_init(&profile);

  profile_start(&profile);

  int rows = 10000;
  int cols = 10000;
  double density = 0.01;

  CSRMatrix *A = create_random_sparse_matrix(rows, cols, density);
  CSRMatrix *B = create_random_sparse_matrix(cols, rows, density);

  profile_stop(&profile);
  profile.init_time = profile.total_time;

  CSRMatrix *C = multiply_csr_matrices(A, B, &profile);

  profile_print(&profile, stdout);

  free_csr_matrix(A);
  free_csr_matrix(B);
  free_csr_matrix(C);

  return 0;
}
