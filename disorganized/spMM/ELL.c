#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

typedef struct {
  size_t rows;
  size_t cols;
  size_t max_nnz_row;
  size_t total_nnz;

  size_t *col_idx;
  double *values;
} ELLMatrix;

typedef struct {
  double load_time;
  double compute_time;
  double memory_usage;
  size_t flop_count;
} Performance;

ELLMatrix* create_ell_matrix(size_t rows, size_t cols, size_t max_nnz_row) {
  ELLMatrix* matrix = (ELLMatrix*)malloc(sizeof(ELLMatrix));
  if (!matrix) return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->max_nnz_row = max_nnz_row;
  matrix->total_nnz = 0;

  size_t total_elements = rows * max_nnz_row;
  matrix->col_idx = (size_t*)aligned_alloc(64, total_elements * sizeof(size_t));
  matrix->values = (double*)aligned_alloc(64, total_elements * sizeof(double));

  if (!matrix->col_idx || !matrix->values) {
    free(matrix->col_idx);
    free(matrix->values);
    free(matrix);
    return NULL;
  }

  for (size_t i = 0; i < total_elements; i++) {
    matrix->col_idx[i] = SIZE_MAX;
  }

  memset(matrix->values, 0, total_elements * sizeof(double));

  return matrix;
}

static inline size_t ell_index(const ELLMatrix* matrix, size_t row, size_t idx) {
  return idx * matrix->rows + row;
}

ELLMatrix* load_matrix_from_mat(const char* filename) { //TODO: Implement this function
  struct timeval start, end;
  gettimeofday(&start, NULL);

  mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
  if (!matfp) {
    fprintf(stderr, "Error opening MAT file: %s\n", filename);
    return NULL;
  }

  matvar_t *matvar = Mat_VarRead(matfp, NULL);
  if (!matvar) {
    fprintf(stderr, "Error reading matrix from MAT file\n");
    Mat_Close(matfp);
    return NULL;
  }

  // Convert from MAT format to ELL
  // First find maximum non-zeros per row
  // Then allocate and fill ELL storage
  // Implementation depends on input format
  // ...

  Mat_VarFree(matvar);
  Mat_Close(matfp);

  gettimeofday(&end, NULL);
  double load_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
  printf("Matrix loaded in %.3f seconds\n", load_time);

  return NULL;
}

ELLMatrix* multiply_ell_matrices(const ELLMatrix* A, const ELLMatrix* B) {
  if (A->cols != B->rows) {
    fprintf(stderr, "Matrix dimensions incompatible for multiplication\n");
    return NULL;
  }

  size_t* row_nnz = (size_t*)calloc(A->rows, sizeof(size_t));

  #pragma omp parallel
  {
    size_t* local_counts = (size_t*)calloc(A->rows, sizeof(size_t));
    size_t* marker = (size_t*)calloc(B->cols, sizeof(size_t));

    #pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < A->rows; i++) {
      size_t count = 0;
      size_t mark = i + 1;

      for (size_t j = 0; j < A->max_nnz_row; j++) {
        size_t a_idx = ell_index(A, i, j);
        size_t col = A->col_idx[a_idx];
        if (col == SIZE_MAX) break;

        for (size_t k = 0; k < B->max_nnz_row; k++) {
          size_t b_idx = ell_index(B, col, k);
          size_t b_col = B->col_idx[b_idx];
          if (b_col == SIZE_MAX) break;

          if (marker[b_col] != mark) {
            marker[b_col] = mark;
            count++;
          }
        }
      }
      local_counts[i] = count;
    }

    #pragma omp critical
    {
      for (size_t i = 0; i < A->rows; i++) {
        if (local_counts[i] > row_nnz[i]) {
          row_nnz[i] = local_counts[i];
        }
      }
    }
    free(local_counts);
    free(marker);
  }

  size_t max_nnz_row = 0;
  for (size_t i = 0; i < A->rows; i++) {
    if (row_nnz[i] > max_nnz_row) {
      max_nnz_row = row_nnz[i];
    }
  }

  ELLMatrix* C = create_ell_matrix(A->rows, B->cols, max_nnz_row);
  if (!C) {
    free(row_nnz);
    return NULL;
  }

  #pragma omp parallel
  {
    double* temp_values = (double*)calloc(B->cols, sizeof(double));
    size_t* temp_marks = (size_t*)calloc(B->cols, sizeof(size_t));
    size_t* temp_cols = (size_t*)calloc(B->cols, sizeof(size_t));

    #pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < A->rows; i++) {
      size_t nnz = 0;
      size_t mark = i + 1;

      for (size_t j = 0; j < A->max_nnz_row; j++) {
        size_t a_idx = ell_index(A, i, j);
        size_t col = A->col_idx[a_idx];
        if (col == SIZE_MAX) break;
        double val = A->values[a_idx];

        for (size_t k = 0; k < B->max_nnz_row; k++) {
          size_t b_idx = ell_index(B, col, k);
          size_t b_col = B->col_idx[b_idx];
          if (b_col == SIZE_MAX) break;

          if (temp_marks[b_col] != mark) {
            temp_marks[b_col] = mark;
            temp_cols[nnz++] = b_col;
            temp_values[b_col] = val * B->values[b_idx];
          } else {
            temp_values[b_col] += val * B->values[b_idx];
          }
        }
      }

      for (size_t j = 0; j < nnz; j++) {
        size_t col = temp_cols[j];
        size_t c_idx = ell_index(C, i, j);
        C->col_idx[c_idx] = col;
        C->values[c_idx] = temp_values[col];
        temp_values[col] = 0.0;
      }

      for (size_t j = nnz; j < C->max_nnz_row; j++) {
        size_t c_idx = ell_index(C, i, j);
        C->col_idx[c_idx] = SIZE_MAX;
        C->values[c_idx] = 0.0;
      }
    }
    free(temp_values);
    free(temp_marks);
    free(temp_cols);
  }

  C->total_nnz = 0;
  for (size_t i = 0; i < A->rows; i++) {
    C->total_nnz += row_nnz[i];
  }

  free(row_nnz);
  return C;
}

Performance benchmark_multiplication(const ELLMatrix* A, const ELLMatrix* B) {
  Performance perf = {0};
  struct timeval start, end;

  ELLMatrix* warmup = multiply_ell_matrices(A, B);
  free_ell_matrix(warmup);

  gettimeofday(&start, NULL);
  ELLMatrix* C = multiply_ell_matrices(A, B);
  gettimeofday(&end, NULL);

  perf.compute_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
  perf.memory_usage = (A->rows * A->max_nnz_row + B->rows * B->max_nnz_row +  C->rows * C->max_nnz_row) *  (sizeof(double) + sizeof(size_t));
  perf.flop_count = 2 * C->total_nnz;

  free_ell_matrix(C);
  return perf;
}

void free_ell_matrix(ELLMatrix* matrix) {
  if (matrix) {
    free(matrix->col_idx);
    free(matrix->values);
    free(matrix);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix_A.mat> <matrix_B.mat>\n", argv[0]);
    return 1;
  }

  ELLMatrix* A = load_matrix_from_mat(argv[1]);
  ELLMatrix* B = load_matrix_from_mat(argv[2]);

  if (!A || !B) {
    fprintf(stderr, "Error loading matrices\n");
    return 1;
  }

  Performance perf = benchmark_multiplication(A, B);

  printf("\nPerformance Results:\n");
  printf("Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("Actual FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);
  printf("Storage efficiency: %.2f%%\n", 100.0 * A->total_nnz / (A->rows * A->max_nnz_row));

  printf("\nMatrix Statistics:\n");
  printf("Maximum non-zeros per row: %zu\n", A->max_nnz_row);
  printf("Average non-zeros per row: %.2f\n", (double)A->total_nnz / A->rows);
  printf("Storage overhead: %.2f%%\n", 100.0 * (1.0 - (double)A->total_nnz / (A->rows * A->max_nnz_row)));

  free_ell_matrix(A);
  free_ell_matrix(B);

  return 0;
}
