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
    size_t size;
    double *values;
} Vector;

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

Vector* multiply_ell_matrix_vector(const ELLMatrix* A, const Vector* x) {
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

        #pragma omp for schedule(static)
        for (size_t j = 0; j < A->max_nnz_row; j++) {
            size_t base_idx = j * A->rows;

            #pragma omp simd
            for (size_t i = 0; i < A->rows; i++) {
                size_t col = A->col_idx[base_idx + i];
                if (col != SIZE_MAX) {  // Check if this is a valid entry
                    local_y[i] += A->values[base_idx + i] * x->values[col];
                }
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

Performance benchmark_multiplication(const ELLMatrix* A, const Vector* x) {
    Performance perf = {0};
    struct timeval start, end;

    Vector* warmup = multiply_ell_matrix_vector(A, x);
    free_vector(warmup);

    const int NUM_ITERATIONS = 10;
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        Vector* y = multiply_ell_matrix_vector(A, x);
        free_vector(y);
    }

    gettimeofday(&end, NULL);

    perf.compute_time = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6) / NUM_ITERATIONS;
    perf.memory_usage = (A->rows * A->max_nnz_row) * (sizeof(double) + sizeof(size_t)) + x->size * sizeof(double) + A->rows * sizeof(double);
    perf.flop_count = 2 * A->total_nnz;

    return perf;
}

void free_ell_matrix(ELLMatrix* matrix) {
    if (matrix) {
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

    ELLMatrix* A = load_matrix_from_mat(argv[1]);
    if (!A) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }

    Vector* x = create_random_vector(A->cols);
    if (!x) {
        fprintf(stderr, "Error creating vector\n");
        free_ell_matrix(A);
        return 1;
    }

    Performance perf = benchmark_multiplication(A, x);

    printf("\nPerformance Results:\n");
    printf("Average Computation Time: %.3f seconds\n", perf.compute_time);
    printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
    printf("Actual FLOP Count: %zu\n", perf.flop_count);
    printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);

    printf("\nMatrix Statistics:\n");
    printf("Maximum non-zeros per row: %zu\n", A->max_nnz_row);
    printf("Average non-zeros per row: %.2f\n", (double)A->total_nnz / A->rows);
    printf("Storage efficiency: %.2f%%\n", 100.0 * A->total_nnz / (A->rows * A->max_nnz_row));
    printf("Memory bandwidth: %.2f GB/s\n", (perf.memory_usage + 2 * A->total_nnz * sizeof(double)) /  (perf.compute_time * 1e9));

    free_ell_matrix(A);
    free_vector(x);

    return 0;
}
