#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

typedef struct {
  size_t rows;
  size_t cols;
  size_t nnz;
  size_t *row_ptr;
  size_t *col_idx;
  double *values;
} CSRMatrix;

typedef struct {
  double load_time;
  double compute_time;
  double memory_usage;
  size_t flop_count;
} Performance;

CSRMatrix* create_csr_matrix(size_t rows, size_t cols, size_t nnz) {
  CSRMatrix* matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
  if (!matrix) return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  matrix->row_ptr = (size_t*)aligned_alloc(64, (rows + 1) * sizeof(size_t));
  matrix->col_idx = (size_t*)aligned_alloc(64, nnz * sizeof(size_t));
  matrix->values = (double*)aligned_alloc(64, nnz * sizeof(double));

  if (!matrix->row_ptr || !matrix->col_idx || !matrix->values) {
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    free(matrix);
    return NULL;
  }

  return matrix;
}

CSRMatrix* load_matrix_from_mat(const char* filename) {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
  if (!matfp) {
    fprintf(stderr, "Error opening MAT file: %s\n", filename);
    return NULL;
  }

  matvar_t *rows_var = Mat_VarRead(matfp, "rows");
  matvar_t *cols_var = Mat_VarRead(matfp, "cols");
  matvar_t *nnz_var = Mat_VarRead(matfp, "nnz");

  if (!rows_var || !cols_var || !nnz_var) {
    fprintf(stderr, "Error reading matrix metadata\n");
    Mat_Close(matfp);
    return NULL;
  }

  size_t rows = *(uint64_t*)rows_var->data;
  size_t cols = *(uint64_t*)cols_var->data;
  size_t nnz = *(uint64_t*)nnz_var->data;

  CSRMatrix* matrix = create_csr_matrix(rows, cols, nnz);
  if (!matrix) {
    Mat_VarFree(rows_var);
    Mat_VarFree(cols_var);
    Mat_VarFree(nnz_var);
    Mat_Close(matfp);
    return NULL;
  }

  matvar_t *row_ptr_var = Mat_VarRead(matfp, "row_ptr");
  matvar_t *col_idx_var = Mat_VarRead(matfp, "col_idx");
  matvar_t *values_var = Mat_VarRead(matfp, "values");

  if (!row_ptr_var || !col_idx_var || !values_var) {
    fprintf(stderr, "Error reading matrix arrays\n");
    free_csr_matrix(matrix);
    Mat_Close(matfp);
    return NULL;
  }

  memcpy(matrix->row_ptr, row_ptr_var->data, (rows + 1) * sizeof(size_t));
  memcpy(matrix->col_idx, col_idx_var->data, nnz * sizeof(size_t));
  memcpy(matrix->values, values_var->data, nnz * sizeof(double));

  Mat_VarFree(rows_var);
  Mat_VarFree(cols_var);
  Mat_VarFree(nnz_var);
  Mat_VarFree(row_ptr_var);
  Mat_VarFree(col_idx_var);
  Mat_VarFree(values_var);
  Mat_Close(matfp);

  gettimeofday(&end, NULL);
  double load_time = (end.tv_sec - start.tv_sec) +  (end.tv_usec - start.tv_usec) / 1e6;
  printf("Matrix loaded in %.3f seconds\n", load_time);
  printf("Matrix stats: %zu rows, %zu cols, %zu non-zeros\n",  rows, cols, nnz);

  return matrix;
}

static inline void process_row(const CSRMatrix* A, const CSRMatrix* B, size_t row, double* temp_values, size_t* temp_marks, size_t mark, size_t* result_nnz) {
  for (size_t j = A->row_ptr[row]; j < A->row_ptr[row + 1]; j++) {
    size_t col = A->col_idx[j];
    double val = A->values[j];

    for (size_t k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
      size_t b_col = B->col_idx[k];

      if (temp_marks[b_col] != mark) {
        temp_marks[b_col] = mark;
        (*result_nnz)++;
        temp_values[b_col] = val * B->values[k];
      } else {
        temp_values[b_col] += val * B->values[k];
      }
    }
  }
}

CSRMatrix* multiply_csr_matrices(const CSRMatrix* A, const CSRMatrix* B) {
  if (A->cols != B->rows) {
    fprintf(stderr, "Matrix dimensions incompatible for multiplication\n");
    return NULL;
  }

  size_t* row_nnz = (size_t*)calloc(A->rows, sizeof(size_t));

  #pragma omp parallel
  {
    double* temp_values = (double*)calloc(B->cols, sizeof(double));
    size_t* temp_marks = (size_t*)calloc(B->cols, sizeof(size_t));

    #pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < A->rows; i++) {
      size_t nnz = 0;
      process_row(A, B, i, temp_values, temp_marks, i + 1, &nnz);
      row_nnz[i] = nnz;

      for (size_t j = B->row_ptr[0]; j < B->row_ptr[B->rows]; j++) {
        size_t col = B->col_idx[j];
        if (temp_marks[col] == i + 1) {
          temp_values[col] = 0.0;
        }
      }
    }
    free(temp_values);
    free(temp_marks);
  }

  size_t total_nnz = 0;
  for (size_t i = 0; i < A->rows; i++) {
    total_nnz += row_nnz[i];
  }

  CSRMatrix* C = create_csr_matrix(A->rows, B->cols, total_nnz);
  if (!C) {
    free(row_nnz);
    return NULL;
  }

  C->row_ptr[0] = 0;
  for (size_t i = 0; i < A->rows; i++) {
    C->row_ptr[i + 1] = C->row_ptr[i] + row_nnz[i];
  }

  #pragma omp parallel
  {
    double* temp_values = (double*)calloc(B->cols, sizeof(double));
    size_t* temp_marks = (size_t*)calloc(B->cols, sizeof(size_t));
    size_t* temp_cols = (size_t*)calloc(B->cols, sizeof(size_t));

    #pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < A->rows; i++) {
      size_t next_idx = C->row_ptr[i];
      size_t nnz = 0;

      for (size_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
        size_t col = A->col_idx[j];
        double val = A->values[j];

        for (size_t k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
          size_t b_col = B->col_idx[k];

          if (temp_marks[b_col] != i + 1) {
            temp_marks[b_col] = i + 1;
            temp_cols[nnz++] = b_col;
            temp_values[b_col] = val * B->values[k];
          } else {
            temp_values[b_col] += val * B->values[k];
          }
        }
      }
      for (size_t j = 0; j < nnz; j++) {
        size_t col = temp_cols[j];
        C->col_idx[next_idx] = col;
        C->values[next_idx] = temp_values[col];
        temp_values[col] = 0.0;
        next_idx++;
      }
    }
    free(temp_values);
    free(temp_marks);
    free(temp_cols);
  }
  free(row_nnz);
  return C;
}

Performance benchmark_multiplication(const CSRMatrix* A, const CSRMatrix* B) {
  Performance perf = {0};
  struct timeval start, end;

  CSRMatrix* warmup = multiply_csr_matrices(A, B);
  free_csr_matrix(warmup);

  gettimeofday(&start, NULL);
  CSRMatrix* C = multiply_csr_matrices(A, B);
  gettimeofday(&end, NULL);

  perf.compute_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
  perf.memory_usage = (A->nnz + B->nnz + C->nnz) * (sizeof(double) + sizeof(size_t)) + (A->rows + B->rows + C->rows + 3) * sizeof(size_t);
  perf.flop_count = 2 * C->nnz;

  free_csr_matrix(C);
  return perf;
}

void free_csr_matrix(CSRMatrix* matrix) {
  if (matrix) {
    free(matrix->row_ptr);
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

  CSRMatrix* A = load_matrix_from_mat(argv[1]);
  CSRMatrix* B = load_matrix_from_mat(argv[2]);

  if (!A || !B) {
    fprintf(stderr, "Error loading matrices\n");
    return 1;
  }

  Performance perf = benchmark_multiplication(A, B);

  printf("\nPerformance Results:\n");
  printf("Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);

  free_csr_matrix(A);
  free_csr_matrix(B);

  return 0;
}
