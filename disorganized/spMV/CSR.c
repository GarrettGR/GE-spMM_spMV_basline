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
  size_t size;
  double *values;
} Vector;

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

Vector* create_vector(size_t size) {
  Vector* vec = (Vector*)malloc(sizeof(Vector));
  if (!vec) return NULL;

  vec->size = size;
  vec->values = (double*)aligned_alloc(64, size * sizeof(double));

  if (!vec->values) {
    free(vec);
    return NULL;
  }
  return vec;
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

Vector* create_random_vector(size_t size) {
  Vector* vec = create_vector(size);
  if (!vec) return NULL;

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++) {
    vec->values[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  }

  return vec;
}

Vector* multiply_csr_matrix_vector(const CSRMatrix* A, const Vector* x) {
  if (A->cols != x->size) {
    fprintf(stderr, "Matrix and vector dimensions incompatible for multiplication\n");
    return NULL;
  }

  Vector* y = create_vector(A->rows);
  if (!y) return NULL;

  memset(y->values, 0, A->rows * sizeof(double));

  const size_t CHUNK_SIZE = 32;

  #pragma omp parallel
  {
    double* local_y = (double*)aligned_alloc(64, A->rows * sizeof(double));
    memset(local_y, 0, A->rows * sizeof(double));

    #pragma omp for schedule(dynamic, CHUNK_SIZE)
    for (size_t i = 0; i < A->rows; i++) {
      double sum = 0.0;
      for (size_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
        sum += A->values[j] * x->values[A->col_idx[j]];
      }
      local_y[i] = sum;
    }

    #pragma omp critical
    {
      for (size_t i = 0; i < A->rows; i++) {
        y->values[i] += local_y[i];
      }
    }
    free(local_y);
  }
  return y;
}

Performance benchmark_multiplication(const CSRMatrix* A, const Vector* x) {
  Performance perf = {0};
  struct timeval start, end;

  Vector* warmup = multiply_csr_matrix_vector(A, x);
  free_vector(warmup);

  const int NUM_ITERATIONS = 10;
  gettimeofday(&start, NULL);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    Vector* y = multiply_csr_matrix_vector(A, x);
    free_vector(y);
  }

  gettimeofday(&end, NULL);

  perf.compute_time = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6) / NUM_ITERATIONS;
  perf.memory_usage = A->nnz * (sizeof(double) + sizeof(size_t)) + (A->rows + 1) * sizeof(size_t) + x->size * sizeof(double) + A->rows * sizeof(double);
  perf.flop_count = 2 * A->nnz;

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

void free_vector(Vector* vector) {
  if (vector) {
    free(vector->values);
    free(vector);
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <matrix.mat>\n", argv[0]);
    return 1;
  }

  CSRMatrix* A = load_matrix_from_mat(argv[1]);
  if (!A) {
    fprintf(stderr, "Error loading matrix\n");
    return 1;
  }

  Vector* x = create_random_vector(A->cols);
  if (!x) {
    fprintf(stderr, "Error creating vector\n");
    free_csr_matrix(A);
    return 1;
  }

  Performance perf = benchmark_multiplication(A, x);

  printf("\nPerformance Results:\n");
  printf("Average Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);

  free_csr_matrix(A);
  free_vector(x);

  return 0;
}
