#include "algorithms/spMV/csr_spmv.h"
#include "test_utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef status_t (*spmv_impl_t)(const csr_matrix*, const double*, double*, profile_context*);

static spmv_impl_t current_impl;

static void test_spmv_null_input(void);
static void test_spmv_identity(void);
static void test_spmv_random_small(void);
static void test_spmv_random_large(void);
static void test_spmv_performance(void);

static void test_impl(spmv_impl_t impl, const char* name, int argc, char** argv, const test_case_t* test_cases) {
  printf("\nTesting %s implementation:\n", name);
  current_impl = impl;
  
  if (argc == 1 || strcmp(argv[1], "all") == 0) {
    for (const test_case_t* test = test_cases; test->name != NULL; test++) {
      test->func();
    }
  } else {
    run_test_suite(argc, argv, test_cases);
  }
  
  printf("All %s tests completed\n\n", name);
}

static double measure_time(clock_t start, clock_t end) {
  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static void test_spmv_null_input(void) {
  printf("Testing null input handling...\n");

  double x[1] = {1.0};
  double y[1] = {0.0};

  assert(current_impl(NULL, x, y, NULL) == STATUS_NULL_POINTER);

  csr_matrix *A;
  status_t status = create_test_matrix(1, 1, 1, PATTERN_DIAGONAL, &A);
  assert(status == STATUS_SUCCESS);

  assert(current_impl(A, NULL, y, NULL) == STATUS_NULL_POINTER);
  assert(current_impl(A, x, NULL, NULL) == STATUS_NULL_POINTER);

  csr_free(A, NULL);
  printf("Null input tests passed\n");
}

static void test_spmv_identity(void) {
  printf("Testing identity matrix multiplication...\n");

  const uint64_t size = 10;
  csr_matrix *A;
  status_t status = create_test_matrix(size, size, size, PATTERN_DIAGONAL, &A);
  assert(status == STATUS_SUCCESS);

  double *x = (double *)malloc(size * sizeof(double));
  double *y = (double *)malloc(size * sizeof(double));
  double *expected = (double *)malloc(size * sizeof(double));

  for (uint64_t i = 0; i < size; i++) {
    x[i] = i + 1.0;
    expected[i] = x[i];
  }

  status = current_impl(A, x, y, NULL);
  assert(status == STATUS_SUCCESS);

  assert(compare_vectors(y, expected, size));

  free(x);
  free(y);
  free(expected);
  csr_free(A, NULL);
  printf("Identity matrix tests passed\n");
}

static void test_spmv_random_small(void) {
  printf("Testing random small matrix multiplication...\n");

  const uint64_t rows = 100;
  const uint64_t cols = 80;
  const uint64_t nnz = 500;

  csr_matrix *A;
  status_t status = create_test_matrix(rows, cols, nnz, PATTERN_RANDOM, &A);
  printf("status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  double *x = (double *)malloc(cols * sizeof(double));
  double *y = (double *)malloc(rows * sizeof(double));
  double *expected = (double *)malloc(rows * sizeof(double));

  for (uint64_t i = 0; i < cols; i++) {
    x[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  }

  status = compute_dense_spmv(A, x, expected);
  printf("status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  status = current_impl(A, x, y, NULL);
  printf("status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  assert(compare_vectors(y, expected, rows));

  free(x);
  free(y);
  free(expected);
  csr_free(A, NULL);
  printf("Random small matrix tests passed\n");
}

static void test_spmv_random_large(void) {
  printf("Testing random large matrix multiplication...\n");

  const uint64_t rows = 10000;
  const uint64_t cols = 10000;
  const uint64_t nnz = 100000;

  csr_matrix *A;
  status_t status = create_test_matrix(rows, cols, nnz, PATTERN_RANDOM, &A);
  printf("status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  double *x = (double *)malloc(cols * sizeof(double));
  double *y = (double *)malloc(rows * sizeof(double));
  double *expected = (double *)malloc(rows * sizeof(double));

  for (uint64_t i = 0; i < cols; i++) {
    x[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  }

  status = compute_dense_spmv(A, x, expected);
  printf("status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  clock_t start = clock();
  status = current_impl(A, x, y, NULL);
  clock_t end = clock();
  assert(status == STATUS_SUCCESS);

  assert(compare_vectors(y, expected, rows));

  printf("Large matrix multiplication time: %f seconds\n", measure_time(start, end));

  free(x);
  free(y);
  free(expected);
  csr_free(A, NULL);
  printf("Random large matrix tests passed\n");
}

static void test_spmv_performance(void) {
  printf("Running performance tests...\n");

  const uint64_t sizes[] = {1000, 5000, 10000};
  const double densities[] = {0.001, 0.01, 0.1};

  for (int i = 0; i < 3; i++) {
    uint64_t size = sizes[i];
    for (int j = 0; j < 3; j++) {
      double density = densities[j];
      uint64_t nnz = (uint64_t)(size * size * density);

      printf("\nTesting size=%llu, density=%f, nnz=%llu\n", size, density, nnz);

      csr_matrix *A;
      status_t status = create_dense_matrix(size, size, density, &A);
      assert(status == STATUS_SUCCESS);

      double *x = (double *)malloc(size * sizeof(double));
      double *y = (double *)malloc(size * sizeof(double));

      for (uint64_t k = 0; k < size; k++) {
        x[k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
      }

      clock_t start = clock();
      status = current_impl(A, x, y, NULL);
      clock_t end = clock();
      assert(status == STATUS_SUCCESS);

      double time = measure_time(start, end);
      double gflops = (2.0 * nnz) / (time * 1e9);

      printf("Time: %f seconds\n", time);
      printf("Performance: %f GFlop/s\n", gflops);

      free(x);
      free(y);
      csr_free(A, NULL);
    }
  }
  printf("\nPerformance tests completed\n");
}

static const test_case_t test_cases[] = {
  {"null_input", test_spmv_null_input},
  {"identity", test_spmv_identity},
  {"random_small", test_spmv_random_small},
  {"random_large", test_spmv_random_large},
  {"performance", test_spmv_performance},
  {NULL, NULL}
};

int main(int argc, char **argv) {
  test_impl(csr_spmv_sequential, "Sequential", argc, argv, test_cases);
#ifdef _OPENMP
  test_impl(csr_spmv_openmp, "OpenMP Parallel", argc, argv, test_cases);
#endif  
  return 0;
}