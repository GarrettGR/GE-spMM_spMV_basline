#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

typedef struct {
  size_t row;
  size_t col;
  double value;
} NonZeroElement;

typedef struct {
  size_t rows;
  size_t cols;
  size_t nnz;
  NonZeroElement* elements;
} COOMatrix;

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

COOMatrix* create_coo_matrix(size_t rows, size_t cols, size_t nnz) {
  COOMatrix* matrix = (COOMatrix*)malloc(sizeof(COOMatrix));
  if (!matrix) return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  matrix->elements = (NonZeroElement*)aligned_alloc(64,
    nnz * sizeof(NonZeroElement));

  if (!matrix->elements) {
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

Vector* create_random_vector(size_t size) {
  Vector* vec = create_vector(size);
  if (!vec) return NULL;

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++) {
    vec->values[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  }

  return vec;
}

Vector* multiply_coo_matrix_vector(const COOMatrix* A, const Vector* x) {
  if (A->cols != x->size) {
    fprintf(stderr, "Matrix and vector dimensions incompatible for multiplication\n");
    return NULL;
  }

  Vector* y = create_vector(A->rows);
  if (!y) return NULL;

  memset(y->values, 0, A->rows * sizeof(double));

  const size_t CHUNK_SIZE = 1024;

  #pragma omp parallel
  {
    double* local_y = (double*)aligned_alloc(64, A->rows * sizeof(double));
    memset(local_y, 0, A->rows * sizeof(double));

    #pragma omp for schedule(dynamic, CHUNK_SIZE)
    for (size_t i = 0; i < A->nnz; i++) {
      size_t row = A->elements[i].row;
      size_t col = A->elements[i].col;
      double val = A->elements[i].value;

      local_y[row] += val * x->values[col];
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

Performance benchmark_multiplication(const COOMatrix* A, const Vector* x) {
  Performance perf = {0};
  struct timeval start, end;

  Vector* warmup = multiply_coo_matrix_vector(A, x);
  free_vector(warmup);

  const int NUM_ITERATIONS = 10;
  gettimeofday(&start, NULL);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    Vector* y = multiply_coo_matrix_vector(A, x);
    free_vector(y);
  }

  gettimeofday(&end, NULL);

  perf.compute_time = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6) / NUM_ITERATIONS;
  perf.memory_usage = A->nnz * sizeof(NonZeroElement) + x->size * sizeof(double) + A->rows * sizeof(double);
  perf.flop_count = 2 * A->nnz;

  return perf;
}

void free_coo_matrix(COOMatrix* matrix) {
  if (matrix) {
    free(matrix->elements);
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

  COOMatrix* A = load_matrix_from_mat(argv[1]);
  if (!A) {
    fprintf(stderr, "Error loading matrix\n");
    return 1;
  }

  Vector* x = create_random_vector(A->cols);
  if (!x) {
    fprintf(stderr, "Error creating vector\n");
    free_coo_matrix(A);
    return 1;
  }

  Performance perf = benchmark_multiplication(A, x);

  printf("\nPerformance Results:\n");
  printf("Average Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);
  printf("Matrix density: %.2e%%\n", 100.0 * A->nnz / (A->rows * A->cols));

  free_coo_matrix(A);
  free_vector(x);

  return 0;
}
