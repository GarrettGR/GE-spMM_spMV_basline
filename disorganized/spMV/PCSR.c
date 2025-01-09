#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

#define PADDING_FACTOR 8  // assuming 64-byte cache line and 8-byte doubles (??)
#define ALIGN_TO(x) (((x) + PADDING_FACTOR - 1) & ~(PADDING_FACTOR - 1))

typedef struct {
  size_t rows;
  size_t cols;
  size_t nnz;
  size_t padded_nnz;
  size_t *row_ptr;
  size_t *col_idx;
  double *values;
} PCSRMatrix;

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

PCSRMatrix* create_pcsr_matrix(size_t rows, size_t cols, size_t nnz) {
  PCSRMatrix* matrix = (PCSRMatrix*)malloc(sizeof(PCSRMatrix));
  if (!matrix) return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;
  matrix->padded_nnz = ALIGN_TO(nnz);

  matrix->row_ptr = (size_t*)aligned_alloc(64, (rows + 1) * sizeof(size_t));
  matrix->col_idx = (size_t*)aligned_alloc(64, matrix->padded_nnz * sizeof(size_t));
  matrix->values = (double*)aligned_alloc(64, matrix->padded_nnz * sizeof(double));

  if (!matrix->row_ptr || !matrix->col_idx || !matrix->values) {
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    free(matrix);
    return NULL;
  }

  return matrix;
}

PCSRMatrix* load_matrix_from_mat(const char* filename) {
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

  size_t* input_row_ptr = (size_t*)row_ptr_var->data;
  size_t max_row_length = 0;
  for (size_t i = 0; i < rows; i++) {
    size_t row_length = input_row_ptr[i + 1] - input_row_ptr[i];
    if (row_length > max_row_length) {
      max_row_length = row_length;
    }
  }

  const size_t SEGMENT_SIZE = 32;
  const float PADDING_FACTOR = 0.2;
  size_t padded_segment_size = SEGMENT_SIZE + (size_t)(SEGMENT_SIZE * PADDING_FACTOR);
  size_t num_segments = (rows + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

  PCSRMatrix* matrix = (PCSRMatrix*)malloc(sizeof(PCSRMatrix));
  if (!matrix) {
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
  matrix->nnz = nnz;
  matrix->num_segments = num_segments;

  matrix->segments = (Segment*)malloc(num_segments * sizeof(Segment));
  size_t total_capacity = nnz + (size_t)(nnz * PADDING_FACTOR);
  matrix->col_idx = (size_t*)malloc(total_capacity * sizeof(size_t));
  matrix->values = (double*)malloc(total_capacity * sizeof(double));
  matrix->row_to_segment = (size_t*)malloc(rows * sizeof(size_t));

  if (!matrix->segments || !matrix->col_idx || !matrix->values || !matrix->row_to_segment) {
    free_pcsr_matrix(matrix);
    Mat_VarFree(rows_var);
    Mat_VarFree(cols_var);
    Mat_VarFree(nnz_var);
    Mat_VarFree(row_ptr_var);
    Mat_VarFree(col_idx_var);
    Mat_VarFree(values_var);
    Mat_Close(matfp);
    return NULL;
  }

  size_t current_pos = 0;
  for (size_t seg = 0; seg < num_segments; seg++) {
    size_t seg_start = seg * SEGMENT_SIZE;
    size_t seg_end = min(seg_start + SEGMENT_SIZE, rows);

    matrix->segments[seg].start_idx = current_pos;
    matrix->segments[seg].count = 0;

    size_t seg_nnz = 0;
    for (size_t i = seg_start; i < seg_end; i++) {
      seg_nnz += input_row_ptr[i + 1] - input_row_ptr[i];
    }
    size_t seg_capacity = seg_nnz + (size_t)(seg_nnz * PADDING_FACTOR);
    matrix->segments[seg].capacity = seg_capacity;

    for (size_t row = seg_start; row < seg_end; row++) {
      matrix->row_to_segment[row] = seg;
      size_t row_start = input_row_ptr[row];
      size_t row_end = input_row_ptr[row + 1];

      for (size_t j = row_start; j < row_end; j++) {
        matrix->col_idx[current_pos] = ((size_t*)col_idx_var->data)[j];
        matrix->values[current_pos] = ((double*)values_var->data)[j];
        current_pos++;
        matrix->segments[seg].count++;
      }
    }
    current_pos += (seg_capacity - matrix->segments[seg].count);
  }

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
  printf("PCSR format: %zu segments, %.2f%% storage efficiency\n", num_segments, (100.0 * nnz) / total_capacity);
  printf("Average segment size: %.2f elements\n", (double)nnz / num_segments);

  return matrix;
}

static inline size_t min(size_t a, size_t b) {
  return (a < b) ? a : b;
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

Vector* multiply_pcsr_matrix_vector(const PCSRMatrix* A, const Vector* x) {
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
      size_t row_start = A->row_ptr[i];
      size_t row_end = A->row_ptr[i + 1];

      size_t j;
      for (j = row_start; j + PADDING_FACTOR <= row_end; j += PADDING_FACTOR) {
        double temp_sum = 0.0;
        #pragma omp simd reduction(+:temp_sum)
        for (size_t k = 0; k < PADDING_FACTOR; k++) {
          size_t col = A->col_idx[j + k];
          if (col < A->cols) {
            temp_sum += A->values[j + k] * x->values[col];
          }
        }
        sum += temp_sum;
      }

      for (; j < row_end; j++) {
        size_t col = A->col_idx[j];
        if (col < A->cols) {
          sum += A->values[j] * x->values[col];
        }
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

Performance benchmark_multiplication(const PCSRMatrix* A, const Vector* x) {
  Performance perf = {0};
  struct timeval start, end;

  Vector* warmup = multiply_pcsr_matrix_vector(A, x);
  free_vector(warmup);

  const int NUM_ITERATIONS = 10;
  gettimeofday(&start, NULL);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    Vector* y = multiply_pcsr_matrix_vector(A, x);
    free_vector(y);
  }

  gettimeofday(&end, NULL);

  perf.compute_time = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6) / NUM_ITERATIONS;
  perf.memory_usage = A->padded_nnz * (sizeof(double) + sizeof(size_t)) + (A->rows + 1) * sizeof(size_t) + x->size * sizeof(double) + A->rows * sizeof(double);
  perf.flop_count = 2 * A->nnz;

  return perf;
}

void free_pcsr_matrix(PCSRMatrix* matrix) {
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

  PCSRMatrix* A = load_matrix_from_mat(argv[1]);
  if (!A) {
    fprintf(stderr, "Error loading matrix\n");
    return 1;
  }

  Vector* x = create_random_vector(A->cols);
  if (!x) {
    fprintf(stderr, "Error creating vector\n");
    free_pcsr_matrix(A);
    return 1;
  }

  Performance perf = benchmark_multiplication(A, x);

  printf("\nPerformance Results:\n");
  printf("Average Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("Actual FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);
  printf("Padding Overhead: %.2f%%\n", ((double)(A->padded_nnz - A->nnz) / A->nnz) * 100.0);

  free_pcsr_matrix(A);
  free_vector(x);

  return 0;
}
