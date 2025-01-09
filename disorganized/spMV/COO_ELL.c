#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

typedef struct {
  size_t nnz;
  size_t *row_idx;
  size_t *col_idx;
  double *values;
} COOComponent;

typedef struct {
  size_t max_nnz_row;
  size_t *col_idx;
  double *values;
} ELLComponent;

typedef struct {
  size_t rows;
  size_t cols;
  size_t total_nnz;
  ELLComponent ell;
  COOComponent coo;
} HybridMatrix;

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

static inline size_t ell_index(const HybridMatrix* matrix, size_t row, size_t idx) {
  return idx * matrix->rows + row;
}

HybridMatrix* create_hybrid_matrix(size_t rows, size_t cols, size_t ell_width, size_t coo_nnz) {
  // ... (Same implementation as matrix-matrix version)
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

Vector* multiply_hybrid_matrix_vector(const HybridMatrix* A, const Vector* x) {
  if (A->cols != x->size) {
    fprintf(stderr, "Matrix and vector dimensions incompatible for multiplication\n");
    return NULL;
  }

  Vector* y = create_vector(A->rows);
  if (!y) return NULL;

  memset(y->values, 0, A->rows * sizeof(double));

  #pragma omp parallel
  {
    double* local_y = (double*)aligned_alloc(64, A->rows * sizeof(double));
    memset(local_y, 0, A->rows * sizeof(double));

    #pragma omp for schedule(static) nowait
    for (size_t j = 0; j < A->ell.max_nnz_row; j++) {
      size_t base_idx = j * A->rows;

      #pragma omp simd
      for (size_t i = 0; i < A->rows; i++) {
        size_t col = A->ell.col_idx[base_idx + i];
        if (col != SIZE_MAX) {
          local_y[i] += A->ell.values[base_idx + i] * x->values[col];
        }
      }
    }

    #pragma omp for schedule(dynamic, 1024)
    for (size_t i = 0; i < A->coo.nnz; i++) {
      size_t row = A->coo.row_idx[i];
      size_t col = A->coo.col_idx[i];
      local_y[row] += A->coo.values[i] * x->values[col];
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

Performance benchmark_multiplication(const HybridMatrix* A, const Vector* x) {
  Performance perf = {0};
  struct timeval start, end;

  Vector* warmup = multiply_hybrid_matrix_vector(A, x);
  free_vector(warmup);

  const int NUM_ITERATIONS = 10;
  gettimeofday(&start, NULL);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    Vector* y = multiply_hybrid_matrix_vector(A, x);
    free_vector(y);
  }

  gettimeofday(&end, NULL);

  perf.compute_time = ((end.tv_sec - start.tv_sec) +  (end.tv_usec - start.tv_usec) / 1e6) / NUM_ITERATIONS;

  perf.memory_usage =  (A->rows * A->ell.max_nnz_row) * (sizeof(double) + sizeof(size_t)) + A->coo.nnz
    * (2 * sizeof(size_t) + sizeof(double)) + x->size * sizeof(double) + A->rows * sizeof(double);

  size_t ell_nnz = 0;
  for (size_t j = 0; j < A->ell.max_nnz_row; j++) {
    for (size_t i = 0; i < A->rows; i++) {
      if (A->ell.col_idx[j * A->rows + i] != SIZE_MAX) {
        ell_nnz++;
      }
    }
  }
  perf.flop_count = 2 * (ell_nnz + A->coo.nnz);

  return perf;
}

void free_hybrid_matrix(HybridMatrix* matrix) {
  if (matrix) {
    free(matrix->ell.col_idx);
    free(matrix->ell.values);

    free(matrix->coo.row_idx);
    free(matrix->coo.col_idx);
    free(matrix->coo.values);

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

  HybridMatrix* A = load_matrix_from_mat(argv[1]);
  if (!A) {
    fprintf(stderr, "Error loading matrix\n");
    return 1;
  }

  Vector* x = create_random_vector(A->cols);
  if (!x) {
    fprintf(stderr, "Error creating vector\n");
    free_hybrid_matrix(A);
    return 1;
  }

  Performance perf = benchmark_multiplication(A, x);

  printf("\nPerformance Results:\n");
  printf("Average Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);

  printf("\nMatrix Structure Analysis:\n");
  printf("Total non-zeros: %zu\n", A->total_nnz);
  printf("ELL portion: %zu entries (%.1f%%)\n", A->rows * A->ell.max_nnz_row, 100.0 * A->rows * A->ell.max_nnz_row / A->total_nnz);
  printf("COO portion: %zu entries (%.1f%%)\n", A->coo.nnz, 100.0 * A->coo.nnz / A->total_nnz);

  printf("\nMemory Access Patterns:\n");
  printf("ELL stride: %zu bytes\n", A->rows * sizeof(double));
  printf("Estimated cache lines accessed: %zu\n", (A->rows * A->ell.max_nnz_row * sizeof(double) + 63) / 64 + (A->coo.nnz * sizeof(double) + 63) / 64);

  size_t bytes_accessed =  (A->rows * A->ell.max_nnz_row) * (sizeof(double) + sizeof(size_t)) +
  A->coo.nnz * (2 * sizeof(size_t) + sizeof(double)) + A->cols * sizeof(double) + A->rows * sizeof(double);

  printf("Memory Bandwidth: %.2f GB/s\n", bytes_accessed / (perf.compute_time * 1e9));

  free_hybrid_matrix(A);
  free_vector(x);

  return 0;
}
