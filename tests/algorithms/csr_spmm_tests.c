#include "test_utils.h"
#include "algorithms/spMM/csr_spmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>

typedef status_t (*spmm_impl_t)(const csr_matrix*, const csr_matrix*, csr_matrix**, profile_context*);

static void test_spmm_null_input(void);
static void test_spmm_identity(void);
static void test_spmm_random_small(void);
static void test_spmm_random_large(void);
static void test_spmm_chain(void);
static void test_spmm_block_diagonal(void);
static void test_spmm_dense(void);
static void test_spmm_performance(void);

static double measure_time(clock_t start, clock_t end) {
  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static spmm_impl_t current_impl;

static void test_impl(spmm_impl_t impl, const char* name, int argc, char** argv, const test_case_t* test_cases) {
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

static void test_spmm_null_input(void) {
  printf("Testing null input handling...\n");
  
  csr_matrix* C = NULL;
  
  assert(current_impl(NULL, NULL, &C, NULL) == STATUS_NULL_POINTER);
  
  csr_matrix* A;
  csr_matrix* B;
  status_t status = create_test_matrix(1, 1, 1, PATTERN_DIAGONAL, &A);
  assert(status == STATUS_SUCCESS);
  status = create_test_matrix(1, 1, 1, PATTERN_DIAGONAL, &B);
  assert(status == STATUS_SUCCESS);
  
  assert(current_impl(A, B, NULL, NULL) == STATUS_NULL_POINTER);
  
  csr_free(A, NULL);
  csr_free(B, NULL);
  printf("Null input tests passed\n");
}

static void test_spmm_identity(void) {
  printf("Testing identity matrix multiplication...\n");
  
  const uint64_t size = 10;
  
  csr_matrix* A;
  csr_matrix* B;
  csr_matrix* C;
  
  status_t status = create_test_matrix(size, size, size, PATTERN_DIAGONAL, &A);
  assert(status == STATUS_SUCCESS);
  status = create_test_matrix(size, size, size, PATTERN_DIAGONAL, &B);
  assert(status == STATUS_SUCCESS);
  
  status = current_impl(A, B, &C, NULL);
  assert(status == STATUS_SUCCESS);
  
  assert(compare_matrices(A, C));
  
  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);
  printf("Identity matrix tests passed\n");
}

static void test_spmm_random_small(void) {
  printf("Testing random small matrix multiplication...\n");
  
  const uint64_t rows = 100;
  const uint64_t inner = 80;
  const uint64_t cols = 60;
  const uint64_t nnz_A = 500;
  const uint64_t nnz_B = 400;
  
  csr_matrix* A;
  csr_matrix* B;
  csr_matrix* C;
  csr_matrix* C_expected;
  
  status_t status = create_test_matrix(rows, inner, nnz_A, PATTERN_RANDOM, &A);
  assert(status == STATUS_SUCCESS);
  status = create_test_matrix(inner, cols, nnz_B, PATTERN_RANDOM, &B);
  assert(status == STATUS_SUCCESS);
  
  status = compute_dense_spmm(A, B, &C_expected);
  assert(status == STATUS_SUCCESS);
  
  status = current_impl(A, B, &C, NULL);
  assert(status == STATUS_SUCCESS);
  assert(compare_matrices(C, C_expected));
  
  for (uint64_t i = 0; i < C->rows; i++) {
    for (uint64_t j = C->row_ptr[i]; j < C->row_ptr[i + 1]; j++) {
      assert(fabs(C->values[j]) > 1e-15);
    }
  }
  
  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);
  csr_free(C_expected, NULL);
  printf("Random small matrix tests passed\n");
}

static void test_spmm_random_large(void) {
  printf("Testing random large matrix multiplication...\n");
  
  const uint64_t rows = 1000;
  const uint64_t inner = 1000;
  const uint64_t cols = 1000;
  const uint64_t nnz_A = 10000;
  const uint64_t nnz_B = 10000;
  
  csr_matrix* A;
  csr_matrix* B;
  csr_matrix* C;
  
  status_t status = create_test_matrix(rows, inner, nnz_A, PATTERN_RANDOM, &A);
  assert(status == STATUS_SUCCESS);
  status = create_test_matrix(inner, cols, nnz_B, PATTERN_RANDOM, &B);
  assert(status == STATUS_SUCCESS);
  
  clock_t start = clock();
  status = current_impl(A, B, &C, NULL);
  clock_t end = clock();
  assert(status == STATUS_SUCCESS);
  
  printf("Large matrix multiplication time: %f seconds\n", measure_time(start, end));
  
  assert(C->rows == rows);
  assert(C->cols == cols);
  assert(C->nnz > 0);
  
  for (uint64_t i = 0; i < C->rows; i++) {
    for (uint64_t j = C->row_ptr[i]; j < C->row_ptr[i + 1]; j++) {
      assert(fabs(C->values[j]) > 1e-15);
    }
  }
  
  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);
  printf("Random large matrix tests passed\n");
}

static void test_spmm_chain(void) {
  printf("Testing chain matrix multiplication...\n");
  
  const uint64_t size = 100;
  const uint64_t nnz = 500;
  
  csr_matrix* A;
  csr_matrix* B;
  csr_matrix* C;
  csr_matrix* AB;
  csr_matrix* BC;
  csr_matrix* ABC1;
  csr_matrix* ABC2;
  
  status_t status = create_test_matrix(size, size, nnz, PATTERN_RANDOM, &A);
  assert(status == STATUS_SUCCESS);
  status = create_test_matrix(size, size, nnz, PATTERN_RANDOM, &B);
  assert(status == STATUS_SUCCESS);
  status = create_test_matrix(size, size, nnz, PATTERN_RANDOM, &C);
  assert(status == STATUS_SUCCESS);
  
  status = current_impl(A, B, &AB, NULL);
  assert(status == STATUS_SUCCESS);
  status = current_impl(AB, C, &ABC1, NULL);
  assert(status == STATUS_SUCCESS);
  
  status = current_impl(B, C, &BC, NULL);
  assert(status == STATUS_SUCCESS);
  status = current_impl(A, BC, &ABC2, NULL);
  assert(status == STATUS_SUCCESS);
  
  assert(compare_matrices(ABC1, ABC2));
  
  for (uint64_t i = 0; i < ABC1->rows; i++) {
    for (uint64_t j = ABC1->row_ptr[i]; j < ABC1->row_ptr[i + 1]; j++) {
      assert(fabs(ABC1->values[j]) > 1e-15);
    }
  }
  for (uint64_t i = 0; i < ABC2->rows; i++) {
    for (uint64_t j = ABC2->row_ptr[i]; j < ABC2->row_ptr[i + 1]; j++) {
      assert(fabs(ABC2->values[j]) > 1e-15);
    }
  }
  
  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);
  csr_free(AB, NULL);
  csr_free(BC, NULL);
  csr_free(ABC1, NULL);
  csr_free(ABC2, NULL);
  printf("Chain multiplication tests passed\n");
}

static void test_spmm_block_diagonal(void) {
  printf("Testing block diagonal matrix multiplication...\n");
  
  const uint64_t size = 30;
  const uint64_t block_size = 10;
  const uint64_t nnz = size * 2;
  
  printf("Creating block diagonal matrices of size %llu with %llu non-zeros\n", size, nnz);
  
  csr_matrix* A;
  csr_matrix* B;
  csr_matrix* C;
  
  status_t status = create_test_matrix(size, size, nnz, PATTERN_BLOCK, &A);
  assert(status == STATUS_SUCCESS);
  printf("Created matrix A with %llu non-zeros\n", A->nnz);
  
  status = create_test_matrix(size, size, nnz, PATTERN_BLOCK, &B);
  assert(status == STATUS_SUCCESS);
  printf("Created matrix B with %llu non-zeros\n", B->nnz);
  
  status = current_impl(A, B, &C, NULL);
  assert(status == STATUS_SUCCESS);
  printf("Completed multiplication, result has %llu non-zeros\n", C->nnz);
  
  for (uint64_t i = 0; i < size; i++) {
    uint64_t block = i / block_size;
    for (uint64_t j = C->row_ptr[i]; j < C->row_ptr[i + 1]; j++) {
      uint64_t col_block = C->col_idx[j] / block_size;
      assert(block == col_block);
    }
  }
  
  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);
  printf("Block diagonal matrix tests passed\n");
}

static void test_spmm_dense(void) {
  printf("Testing dense matrix multiplication...\n");
  
  const uint64_t size = 100;
  const double density = 0.8;
  
  csr_matrix* A;
  csr_matrix* B;
  csr_matrix* C;
  csr_matrix* C_expected;
  
  status_t status = create_dense_matrix(size, size, density, &A);
  assert(status == STATUS_SUCCESS);
  status = create_dense_matrix(size, size, density, &B);
  assert(status == STATUS_SUCCESS);
  
  status = compute_dense_spmm(A, B, &C_expected);
  assert(status == STATUS_SUCCESS);
  
  clock_t start = clock();
  status = current_impl(A, B, &C, NULL);
  clock_t end = clock();
  assert(status == STATUS_SUCCESS);
  
  printf("Dense matrix multiplication time: %f seconds\n", measure_time(start, end));
  
  assert(compare_matrices(C, C_expected));
  
  for (uint64_t i = 0; i < C->rows; i++) {
    for (uint64_t j = C->row_ptr[i]; j < C->row_ptr[i + 1]; j++) {
      assert(fabs(C->values[j]) > 1e-15);
    }
  }
  
  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);
  csr_free(C_expected, NULL);
  printf("Dense matrix tests passed\n");
}

static void test_spmm_performance(void) {
  printf("Running performance tests...\n");
  
  const uint64_t sizes[] = {500, 1000, 10000};
  const double densities[] = {0.001, 0.01, 0.05};
  
  for (int i = 0; i < 3; i++) {
    uint64_t size = sizes[i];
    for (int j = 0; j < 3; j++) {
      double density = densities[j];
      fprintf(stderr, "\nTesting size=%llu, density=%f\n", size, density);
      
      uint64_t max_nnz = size * size;
      if (max_nnz / size != size) {
        fprintf(stderr, "Matrix size would cause overflow\n");
        continue;
      }
      
      uint64_t target_nnz = (uint64_t)(max_nnz * density);
      if (target_nnz > max_nnz) {
        fprintf(stderr, "Density calculation would cause overflow\n");
        continue;
      }
      
      uint64_t min_nnz = size;
      target_nnz = (target_nnz < min_nnz) ? min_nnz : target_nnz;
      
      fprintf(stderr, "Creating matrices with target_nnz=%llu\n", target_nnz);
      
      csr_matrix* A = NULL;
      csr_matrix* B = NULL;
      csr_matrix* C = NULL;
      csr_matrix* C_expected = NULL;
      
      status_t status = create_test_matrix(size, size, target_nnz, PATTERN_RANDOM, &A);
      if (status != STATUS_SUCCESS || !A) {
        fprintf(stderr, "Failed to create matrix A (status=%d)\n", status);
        continue;
      }
      
      status = create_test_matrix(size, size, target_nnz, PATTERN_RANDOM, &B);
      if (status != STATUS_SUCCESS || !B) {
        fprintf(stderr, "Failed to create matrix B (status=%d)\n", status);
        csr_free(A, NULL);
        continue;
      }
 
      fprintf(stderr, "\nCreated the matrices, status: %i\n", status);
 
      status = compute_dense_spmm(A, B, &C_expected);
      assert(status == STATUS_SUCCESS);

      fprintf(stderr, "\nComputed the expected output C, status: %i\n", status);
      
      clock_t start = clock();
      status = current_impl(A, B, &C, NULL);
      clock_t end = clock();
      assert(status == STATUS_SUCCESS);

      fprintf(stderr, "\nRan the multiplication algorithm, status: %i\n", status);
      
      assert(compare_matrices(C, C_expected));

      fprintf(stderr, "\nCompared the matrices\n");
      
      double time = measure_time(start, end);
      double gflops = (2.0 * A->nnz * B->nnz) / (time * 1e9);
      
      printf("Matrix sizes: A(%llux%llu, nnz=%llu) B(%llux%llu, nnz=%llu) C(%llux%llu, nnz=%llu)\n",
             A->rows, A->cols, A->nnz, B->rows, B->cols, B->nnz, C->rows, C->cols, C->nnz);
      printf("Time: %f seconds\n", time);
      printf("Performance: %f GFlop/s\n", gflops);
      
      csr_free(A, NULL);
      csr_free(B, NULL);
      csr_free(C, NULL);
      csr_free(C_expected, NULL);
    }
  }
  printf("\nPerformance tests completed\n");
}

static const test_case_t test_cases[] = {
  {"null_input", test_spmm_null_input},
  {"identity", test_spmm_identity},
  {"random_small", test_spmm_random_small},
  {"random_large", test_spmm_random_large},
  {"chain", test_spmm_chain},
  {"block_diagonal", test_spmm_block_diagonal},
  {"dense", test_spmm_dense},
  {"performance", test_spmm_performance},
  {NULL, NULL}
};

int main(int argc, char **argv) {
  test_impl(csr_spmm_sequential, "Sequential", argc, argv, test_cases);
#ifdef _OPENMP
  test_impl(csr_spmm_openmp, "OpenMP Parallel", argc, argv, test_cases);
#endif
  return 0;
}