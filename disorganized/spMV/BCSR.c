#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

#define BLOCK_SIZE 4  // 4x4 blocks (??)
#define BLOCK_AREA (BLOCK_SIZE * BLOCK_SIZE)

typedef struct {
  size_t block_rows;
  size_t block_cols;
  size_t rows;
  size_t cols;
  size_t nnz_blocks;
  size_t *block_row_ptr;
  size_t *block_col_idx;
  double *values;
} BCSRMatrix;

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

BCSRMatrix* create_bcsr_matrix(size_t rows, size_t cols, size_t nnz_blocks) {
  BCSRMatrix* matrix = (BCSRMatrix*)malloc(sizeof(BCSRMatrix));
  if (!matrix) return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->block_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  matrix->block_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  matrix->nnz_blocks = nnz_blocks;

  matrix->block_row_ptr = (size_t*)aligned_alloc(64, (matrix->block_rows + 1) * sizeof(size_t));
  matrix->block_col_idx = (size_t*)aligned_alloc(64, nnz_blocks * sizeof(size_t));
  matrix->values = (double*)aligned_alloc(64, nnz_blocks * BLOCK_AREA * sizeof(double));

  if (!matrix->block_row_ptr || !matrix->block_col_idx || !matrix->values) {
    free(matrix->block_row_ptr);
    free(matrix->block_col_idx);
    free(matrix->values);
    free(matrix);
    return NULL;
  }
  return matrix;
}

BCSRMatrix* load_matrix_from_mat(const char* filename) {
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

  matvar_t *row_ptr_var = Mat_VarRead(matfp, "row_ptr");
  matvar_t *col_idx_var = Mat_VarRead(matfp, "col_idx");
  matvar_t *values_var = Mat_VarRead(matfp, "values");

  if (!row_ptr_var || !col_idx_var || !values_var) {
    fprintf(stderr, "Error reading matrix arrays\n");
    Mat_VarFree(rows_var);
    Mat_VarFree(cols_var);
    Mat_VarFree(nnz_var);
    Mat_Close(matfp);
    return NULL;
  }

  const size_t BLOCK_SIZE = 4;  // 4x4 blocks
  size_t block_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t block_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

  size_t* block_counts = (size_t*)calloc(block_rows * block_cols, sizeof(size_t));
  size_t* input_row_ptr = (size_t*)row_ptr_var->data;
  size_t* input_col_idx = (size_t*)col_idx_var->data;
  double* input_values = (double*)values_var->data;

  for (size_t i = 0; i < rows; i++) {
    size_t block_row = i / BLOCK_SIZE;
    for (size_t j = input_row_ptr[i]; j < input_row_ptr[i + 1]; j++) {
      size_t col = input_col_idx[j];
      size_t block_col = col / BLOCK_SIZE;
      block_counts[block_row * block_cols + block_col] = 1;
    }
  }

  size_t nnz_blocks = 0;
  for (size_t i = 0; i < block_rows * block_cols; i++) {
    if (block_counts[i]) nnz_blocks++;
  }

  BCSRMatrix* matrix = (BCSRMatrix*)malloc(sizeof(BCSRMatrix));
  if (!matrix) {
    free(block_counts);
    Mat_VarFree(rows_var);
    Mat_VarFree(cols_var);
    Mat_VarFree(nnz_var);
    Mat_VarFree(row_ptr_var);
    Mat_VarFree(col_idx_var);
    Mat_VarFree(values_var);
    Mat_Close(matfp);
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->block_size = BLOCK_SIZE;
  matrix->block_rows = block_rows;
  matrix->block_cols = block_cols;
  matrix->nnz_blocks = nnz_blocks;

  matrix->block_row_ptr = (size_t*)malloc((block_rows + 1) * sizeof(size_t));
  matrix->block_col_idx = (size_t*)malloc(nnz_blocks * sizeof(size_t));
  matrix->values = (double*)calloc(nnz_blocks * BLOCK_SIZE * BLOCK_SIZE, sizeof(double));

  if (!matrix->block_row_ptr || !matrix->block_col_idx || !matrix->values) {
    free_bcsr_matrix(matrix);
    free(block_counts);
    Mat_VarFree(rows_var);
    Mat_VarFree(cols_var);
    Mat_VarFree(nnz_var);
    Mat_VarFree(row_ptr_var);
    Mat_VarFree(col_idx_var);
    Mat_VarFree(values_var);
    Mat_Close(matfp);
    return NULL;
  }

  size_t* block_offsets = (size_t*)calloc(block_rows, sizeof(size_t));
  matrix->block_row_ptr[0] = 0;

  for (size_t i = 0; i < block_rows; i++) {
    size_t count = 0;
    for (size_t j = 0; j < block_cols; j++) {
      if (block_counts[i * block_cols + j]) count++;
    }
    matrix->block_row_ptr[i + 1] = matrix->block_row_ptr[i] + count;
  }

  memset(block_counts, 0, block_rows * block_cols * sizeof(size_t));

  for (size_t i = 0; i < rows; i++) {
    size_t block_row = i / BLOCK_SIZE;
    size_t local_row = i % BLOCK_SIZE;

    for (size_t j = input_row_ptr[i]; j < input_row_ptr[i + 1]; j++) {
      size_t col = input_col_idx[j];
      size_t block_col = col / BLOCK_SIZE;
      size_t local_col = col % BLOCK_SIZE;

      size_t block_idx = block_row * block_cols + block_col;
      if (!block_counts[block_idx]) {
        size_t pos = matrix->block_row_ptr[block_row] + block_offsets[block_row];
        matrix->block_col_idx[pos] = block_col;
        block_counts[block_idx] = pos + 1;
        block_offsets[block_row]++;
      }

      size_t pos = block_counts[block_idx] - 1;
      matrix->values[pos * BLOCK_SIZE * BLOCK_SIZE + local_row * BLOCK_SIZE + local_col] = input_values[j];
    }
  }

  free(block_counts);
  free(block_offsets);
  Mat_VarFree(rows_var);
  Mat_VarFree(cols_var);
  Mat_VarFree(nnz_var);
  Mat_VarFree(row_ptr_var);
  Mat_VarFree(col_idx_var);
  Mat_VarFree(values_var);
  Mat_Close(matfp);

  gettimeofday(&end, NULL);
  double load_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

  printf("Matrix loaded in %.3f seconds\n", load_time);
  printf("Matrix stats: %zu rows, %zu cols, %zu non-zeros\n", rows, cols, nnz);
  printf("Block structure: %zux%zu blocks (%zux%zu each)\n", block_rows, block_cols, BLOCK_SIZE, BLOCK_SIZE);
  printf("Non-zero blocks: %zu (%.2f%% of total blocks)\n", nnz_blocks, 100.0 * nnz_blocks / (block_rows * block_cols));
  printf("Average non-zeros per block: %.2f\n", (double)nnz / nnz_blocks);

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

Vector* create_random_vector(size_t size) {
  Vector* vec = create_vector(size);
  if (!vec) return NULL;

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++) {
    vec->values[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  }
  return vec;
}

static inline void multiply_block_vector(const double* block, const double* x_segment, double* y_segment) {
  for (int i = 0; i < BLOCK_SIZE; i++) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int j = 0; j < BLOCK_SIZE; j++) {
      sum += block[i * BLOCK_SIZE + j] * x_segment[j];
    }
    y_segment[i] += sum;
  }
}

Vector* multiply_bcsr_matrix_vector(const BCSRMatrix* A, const Vector* x) {
  if (A->cols != x->size) {
    fprintf(stderr, "Matrix and vector dimensions incompatible for multiplication\n");
    return NULL;
  }

  Vector* y = create_vector(A->rows);
  if (!y) return NULL;

  memset(y->values, 0, A->rows * sizeof(double));

  const size_t CHUNK_SIZE = 8;

  #pragma omp parallel
  {
    double* local_y = (double*)aligned_alloc(64, A->rows * sizeof(double));
    memset(local_y, 0, A->rows * sizeof(double));

    #pragma omp for schedule(dynamic, CHUNK_SIZE)
    for (size_t block_row = 0; block_row < A->block_rows; block_row++) {
      size_t row_start = block_row * BLOCK_SIZE;
      size_t row_end = (block_row + 1) * BLOCK_SIZE;
      if (row_end > A->rows) row_end = A->rows;

      for (size_t j = A->block_row_ptr[block_row];
        j < A->block_row_ptr[block_row + 1]; j++) {
          size_t block_col = A->block_col_idx[j];
          size_t col_start = block_col * BLOCK_SIZE;

          const double* block = &A->values[j * BLOCK_AREA];
          const double* x_segment = &x->values[col_start];
          double* y_segment = &local_y[row_start];

          multiply_block_vector(block, x_segment, y_segment);
        }
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

Performance benchmark_multiplication(const BCSRMatrix* A, const Vector* x) {
  Performance perf = {0};
  struct timeval start, end;

  Vector* warmup = multiply_bcsr_matrix_vector(A, x);
  free_vector(warmup);

  const int NUM_ITERATIONS = 10;
  gettimeofday(&start, NULL);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    Vector* y = multiply_bcsr_matrix_vector(A, x);
    free_vector(y);
  }

  gettimeofday(&end, NULL);

  perf.compute_time = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6) / NUM_ITERATIONS;
  perf.memory_usage = A->nnz_blocks * (sizeof(size_t) + BLOCK_AREA * sizeof(double)) + (A->block_rows + 1) * sizeof(size_t) + x->size * sizeof(double) + A->rows * sizeof(double);
  perf.flop_count = A->nnz_blocks * (2 * BLOCK_SIZE * BLOCK_SIZE);

  return perf;
}

void free_bcsr_matrix(BCSRMatrix* matrix) {
  if (matrix) {
    free(matrix->block_row_ptr);
    free(matrix->block_col_idx);
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

  BCSRMatrix* A = load_matrix_from_mat(argv[1]);
  if (!A) {
    fprintf(stderr, "Error loading matrix\n");
    return 1;
  }

  Vector* x = create_random_vector(A->cols);
  if (!x) {
    fprintf(stderr, "Error creating vector\n");
    free_bcsr_matrix(A);
    return 1;
  }

  Performance perf = benchmark_multiplication(A, x);

  printf("\nPerformance Results:\n");
  printf("Average Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);
  printf("Number of blocks: %zu\n", A->nnz_blocks);
  printf("Block density: %.2f%%\n", 100.0 * A->nnz_blocks / (A->block_rows * A->block_cols));

  free_bcsr_matrix(A);
  free_vector(x);

  return 0;
}
