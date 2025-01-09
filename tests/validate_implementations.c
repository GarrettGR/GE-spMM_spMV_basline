// #include "test_utils.h"
#include "utils/validator.h"
#include "algorithms/spMM/csr_spmm.h"
#include "algorithms/spMV/csr_spmv.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef status_t (*spmm_impl_t)(const csr_matrix*, const csr_matrix*, csr_matrix**, profile_context*);
typedef status_t (*spmv_impl_t)(const csr_matrix*, const double*, double*, profile_context*);

typedef struct {
  const char* name;
  void (*func)(void);
} test_case_t;

typedef enum {
  IMPL_TYPE_SPMM,
  IMPL_TYPE_SPMV
} impl_type_t;

typedef struct {
  impl_type_t type;
  union {
    spmm_impl_t spmm;
    spmv_impl_t spmv;
  } func;
  const char* name;
} impl_t;

static impl_t current_impl;

static int run_test_suite(int argc, char** argv, const test_case_t* test_cases);
static void test_validate_spmm_random(void);
static void test_validate_spmm_block(void);
static void test_validate_spmm_chain(void);
static void test_validate_spmm_extreme(void);
static void test_validate_spmv_random(void);
static void test_validate_spmv_dense(void);
static void test_validate_spmv_extreme(void);

static void test_impl(impl_t impl, int argc, char** argv, const test_case_t* test_cases) {
  printf("\nTesting %s implementation:\n", impl.name);
  current_impl = impl;
  run_test_suite(argc, argv, test_cases);
  printf("All %s tests completed\n\n", impl.name);
}

static int run_test_suite(int argc, char **argv, const test_case_t *test_cases) {
  const char *test_name = argv[1];

  for (const test_case_t *test = test_cases; test->name != NULL; test++) {
    if (strcmp(test_name, test->name) == 0) {
      if ((current_impl.type == IMPL_TYPE_SPMV && strncmp(test->name, "spmm_", 5) == 0) ||
          (current_impl.type == IMPL_TYPE_SPMM && strncmp(test->name, "spmv_", 5) == 0)) {
        continue;
      }
      test->func();
      return 0;
    }
  }

  if (strcmp(test_name, "all") != 0) {
    printf("Unknown test: %s\n", test_name);
    return 1;
  }

  for (const test_case_t *test = test_cases; test->name != NULL; test++) {
    if ((current_impl.type == IMPL_TYPE_SPMV && strncmp(test->name, "spmm_", 5) == 0) ||
        (current_impl.type == IMPL_TYPE_SPMM && strncmp(test->name, "spmv_", 5) == 0)) {
      continue;
    }
    test->func();
  }

  return 0;
}

static void test_validate_spmm_random(void) {
  printf("Testing SpMM implementation with random matrices...\n");

  struct {
    uint64_t m, n, k;
    uint64_t nnz_a;
    uint64_t nnz_b;
    double density_a;
    double density_b;
  } tests[] = {
    {10, 10, 10, 20, 20, 0.2, 0.2},
    {100, 80, 60, 1000, 800, 0.15, 0.15},
    {500, 500, 500, 5000, 5000, 0.02, 0.02},
    {50, 50, 50, 2000, 2000, 0.8, 0.8}
  };

  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
    printf("\nTest case %zu: %llux%llu * %llux%llu\n", i + 1, tests[i].m, tests[i].k, tests[i].k, tests[i].n);

    csr_matrix *A = NULL, *B = NULL, *C = NULL;

    status_t status = create_test_matrix(tests[i].m, tests[i].k, tests[i].nnz_a, PATTERN_RANDOM, &A);
    assert(status == STATUS_SUCCESS);

    status = create_test_matrix(tests[i].k, tests[i].n, tests[i].nnz_b, PATTERN_RANDOM, &B);
    assert(status == STATUS_SUCCESS);

    status = current_impl.func.spmm(A, B, &C, NULL);
    assert(status == STATUS_SUCCESS);

    bool valid = validate_spmm_with_petsc(A, B, C);
    printf("Validation %s\n", valid ? "PASSED" : "FAILED");
    assert(valid);

    csr_free(A, NULL);
    csr_free(B, NULL);
    csr_free(C, NULL);
  }

  printf("\nAll random SpMM tests passed!\n");
}

static void test_validate_spmm_block(void) {
  printf("Testing SpMM implementation with block matrices...\n");

  const uint64_t sizes[] = {64, 128, 256}; // has to be multiples of block size (32) ???

  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    uint64_t size = sizes[i];
    uint64_t nnz = size * size * 0.01;

    printf("\nTest case %zu: %llux%llu matrices\n", i + 1, size, size);

    csr_matrix *A = NULL, *B = NULL, *C = NULL;

    status_t status = create_test_matrix(size, size, nnz, PATTERN_BLOCK, &A);
    assert(status == STATUS_SUCCESS);

    status = create_test_matrix(size, size, nnz, PATTERN_BLOCK, &B);
    assert(status == STATUS_SUCCESS);

    status = current_impl.func.spmm(A, B, &C, NULL);
    assert(status == STATUS_SUCCESS);

    bool valid = validate_spmm_with_petsc(A, B, C);
    printf("Validation %s\n", valid ? "PASSED" : "FAILED");
    assert(valid);

    csr_free(A, NULL);
    csr_free(B, NULL);
    csr_free(C, NULL);
  }

  printf("\nAll block SpMM tests passed!\n");
}

static void test_validate_spmm_chain() {
  printf("Testing SpMM implementation with chain multiplication...\n");

  const uint64_t size = 100;
  const uint64_t nnz = size * 2;

  csr_matrix *A = NULL, *B = NULL, *C = NULL, *AB = NULL, *BC = NULL, *ABC1 = NULL, *ABC2 = NULL;

  status_t status = create_test_matrix(size, size, nnz, PATTERN_RANDOM, &A);
  assert(status == STATUS_SUCCESS);

  status = create_test_matrix(size, size, nnz, PATTERN_RANDOM, &B);
  assert(status == STATUS_SUCCESS);

  status = create_test_matrix(size, size, nnz, PATTERN_RANDOM, &C);
  assert(status == STATUS_SUCCESS);

  status = current_impl.func.spmm(A, B, &AB, NULL);
  assert(status == STATUS_SUCCESS);

  status = current_impl.func.spmm(AB, C, &ABC1, NULL);
  assert(status == STATUS_SUCCESS);

  status = current_impl.func.spmm(B, C, &BC, NULL);
  assert(status == STATUS_SUCCESS);

  status = current_impl.func.spmm(A, BC, &ABC2, NULL);
  assert(status == STATUS_SUCCESS);

  bool valid = compare_matrices(ABC1, ABC2);
  printf("Validation %s\n", valid ? "PASSED" : "FAILED");
  assert(valid);

  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);
  csr_free(AB, NULL);
  csr_free(BC, NULL);
  csr_free(ABC1, NULL);
  csr_free(ABC2, NULL);

  printf("\nChain multiplication tests passed!\n");
}

static void test_validate_spmv_random(void) {
  printf("Testing SpMV implementation with random matrices...\n");

  struct {
    uint64_t m, n;
    uint64_t nnz;
    double density;
  } tests[] = {
    {10, 10, 20, 0.2},
    {100, 80, 1000, 0.15},
    {1000, 1000, 5000, 0.005},
    {50, 50, 2000, 0.8}
  };

  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
    printf("\nTest case %zu: %llux%llu matrix\n", i + 1, tests[i].m, tests[i].n);

    csr_matrix *A = NULL;
    status_t status = create_test_matrix(tests[i].m, tests[i].n, tests[i].nnz, PATTERN_RANDOM, &A);
    assert(status == STATUS_SUCCESS);

    double *x = (double *)malloc(tests[i].n * sizeof(double));
    double *y = (double *)malloc(tests[i].m * sizeof(double));
    assert(x && y);

    for (uint64_t j = 0; j < tests[i].n; j++) {
      x[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    status = current_impl.func.spmv(A, x, y, NULL); // tests[i].n, tests[i].m
    assert(status == STATUS_SUCCESS);

    bool valid = validate_spmv_with_petsc(A, x, y); // tests[i].n, tests[i].m);
    printf("Validation %s\n", valid ? "PASSED" : "FAILED");
    assert(valid);

    free(x);
    free(y);
    csr_free(A, NULL);
  }

  printf("\nAll random SpMV tests passed!\n");
}

static void test_validate_spmm_extreme(void) {
  printf("Testing SpMM implementation with extreme cases...\n");

  uint64_t size = 100;
  csr_matrix *A = NULL, *B = NULL, *C = NULL;

  status_t status = create_test_matrix(size, size, size, PATTERN_DIAGONAL, &A);
  assert(status == STATUS_SUCCESS);

  status = create_test_matrix(size, size, size * size * 0.5, PATTERN_RANDOM, &B);
  assert(status == STATUS_SUCCESS);

  status = current_impl.func.spmm(A, B, &C, NULL);
  assert(status == STATUS_SUCCESS);

  bool valid = validate_spmm_with_petsc(A, B, C);
  printf("Single non-zero per row test: %s\n", valid ? "PASSED" : "FAILED");
  assert(valid);

  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);

  status = create_test_matrix(size, size, size * 2, PATTERN_RANDOM, &A);
  assert(status == STATUS_SUCCESS);

  status = create_test_matrix(size, size, size * size * 0.9, PATTERN_RANDOM, &B);
  assert(status == STATUS_SUCCESS);

  status = current_impl.func.spmm(A, B, &C, NULL);
  assert(status == STATUS_SUCCESS);

  valid = validate_spmm_with_petsc(A, B, C);
  printf("Sparse * Dense test: %s\n", valid ? "PASSED" : "FAILED");
  assert(valid);

  csr_free(A, NULL);
  csr_free(B, NULL);
  csr_free(C, NULL);

  printf("\nAll extreme case SpMM tests passed!\n");
}

static void test_validate_spmv_dense(void) {
  printf("Testing SpMV implementation with dense matrices...\n");

  struct {
    uint64_t m, n;
    double density;
    const char* desc;
  } tests[] = {
    {50, 50, 0.7, "70% dense square matrix"},
    {100, 100, 0.8, "80% dense square matrix"},
    {80, 120, 0.9, "90% dense rectangular matrix"},
    {200, 150, 0.95, "95% dense rectangular matrix"}
  };

  for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
    printf("\nTest case %zu: %s (%llux%llu)\n", i+1, tests[i].desc, tests[i].m, tests[i].n);

    uint64_t nnz = (uint64_t)(tests[i].m * tests[i].n * tests[i].density);

    csr_matrix* A = NULL;
    status_t status = create_test_matrix(tests[i].m, tests[i].n, nnz, PATTERN_DENSE, &A);
    assert(status == STATUS_SUCCESS);

    double* x = (double*)malloc(tests[i].n * sizeof(double));
    double* y = (double*)malloc(tests[i].m * sizeof(double));
    assert(x && y);

    for (uint64_t j = 0; j < tests[i].n; j++) {
      x[j] = cos(j * 0.1) + sin(j * 0.05);
    }

    status = current_impl.func.spmv(A, x, y, NULL); // tests[i].n, tests[i].m
    assert(status == STATUS_SUCCESS);

    bool valid = validate_spmv_with_petsc(A, x, y); // tests[i].n, tests[i].m);
    printf("Validation %s\n", valid ? "PASSED" : "FAILED");
    assert(valid);

    free(x);
    free(y);
    csr_free(A, NULL);
  }

  printf("\nAll dense SpMV tests passed!\n");
}

static void test_validate_spmv_extreme(void) {
  printf("Testing SpMV implementation with extreme cases...\n");

  // Test case 1: Diagonal matrix
  {
    printf("\nTest case 1: Diagonal matrix\n");
    const uint64_t size = 100;

    csr_matrix* A = NULL;
    status_t status = create_test_matrix(size, size, size, PATTERN_DIAGONAL, &A);
    assert(status == STATUS_SUCCESS);

    double* x = (double*)malloc(size * sizeof(double));
    double* y = (double*)malloc(size * sizeof(double));
    assert(x && y);

    for (uint64_t i = 0; i < size; i++) {
      x[i] = i + 1;
    }

    status = current_impl.func.spmv(A, x, y, NULL);
    assert(status == STATUS_SUCCESS);

    bool valid = validate_spmv_with_petsc(A, x, y); // size, size);
    printf("Diagonal matrix test: %s\n", valid ? "PASSED" : "FAILED");
    assert(valid);

    free(x);
    free(y);
    csr_free(A, NULL);
  }

  // Test case 2: Single non-zero per row
  {
    printf("\nTest case 2: Single non-zero per row\n");
    const uint64_t m = 150;
    const uint64_t n = 200;

    csr_matrix* A = NULL;
    status_t status = csr_create(m, n, m, &A, NULL);
    assert(status == STATUS_SUCCESS);

    A->row_ptr[0] = 0;
    for (uint64_t i = 0; i < m; i++) {
      A->row_ptr[i + 1] = i + 1;
      A->col_idx[i] = rand() % n;
      A->values[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    double* x = (double*) malloc(n * sizeof(double));
    double* y = (double*) malloc(m * sizeof(double));
    assert(x && y);

    for (uint64_t i = 0; i < n; i++) {
      x[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }

    status = current_impl.func.spmv(A, x, y, NULL);
    assert(status == STATUS_SUCCESS);

    bool valid = validate_spmv_with_petsc(A, x, y); // n, m);
    printf("Single non-zero per row test: %s\n", valid ? "PASSED" : "FAILED");
    assert(valid);

    free(x);
    free(y);
    csr_free(A, NULL);
  }

  // Test case 3: Skewed density distribution
  {
    printf("\nTest case 3: Skewed density distribution\n");
    const uint64_t m = 120;
    const uint64_t n = 120;
    const uint64_t dense_rows = 20;
    const uint64_t nnz = dense_rows * n + (m - dense_rows);

    csr_matrix* A = NULL;
    status_t status = csr_create(m, n, nnz, &A, NULL);
    assert(status == STATUS_SUCCESS);

    uint64_t pos = 0;
    A->row_ptr[0] = 0;

    for (uint64_t i = 0; i < dense_rows; i++) {
      for (uint64_t j = 0; j < n; j++) {
        A->col_idx[pos] = j;
        A->values[pos] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        pos++;
      }
      A->row_ptr[i + 1] = pos;
    }

    for (uint64_t i = dense_rows; i < m; i++) {
      A->col_idx[pos] = rand() % n;
      A->values[pos] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
      pos++;
      A->row_ptr[i + 1] = pos;
    }

    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(m * sizeof(double));
    assert(x && y);

    for (uint64_t i = 0; i < n; i++) {
      x[i] = exp(i * 0.1);
    }

    status = current_impl.func.spmv(A, x, y,  NULL);
    assert(status == STATUS_SUCCESS);

    bool valid = validate_spmv_with_petsc(A, x, y); // n, m);
    printf("Skewed density test: %s\n", valid ? "PASSED" : "FAILED");
    assert(valid);

    free(x);
    free(y);
    csr_free(A, NULL);
  }

  printf("\nAll extreme case SpMV tests passed!\n");
}

static const test_case_t test_cases[] = {
  {"spmm_random", test_validate_spmm_random},
  {"spmm_block", test_validate_spmm_block},
  {"spmm_extreme", test_validate_spmm_extreme},
  {"spmm_chain", test_validate_spmm_chain},
  {"spmv_random", test_validate_spmv_random},
  {"spmv_dense", test_validate_spmv_dense},
  {"spmv_extreme", test_validate_spmv_extreme},
  {NULL, NULL}
};

int main(int argc, char **argv) {
  impl_t impls[] = {
    {IMPL_TYPE_SPMM, {.spmm = csr_spmm_sequential}, "SpMM Sequential"},
    {IMPL_TYPE_SPMV, {.spmv = csr_spmv_sequential}, "SpMV Sequential"},
 #ifdef _OPENMP
   , {IMPL_TYPE_SPMM, {.spmm = csr_spmm_openmp}, "SpMM OpenMP"},
   {IMPL_TYPE_SPMV, {.spmv = csr_spmv_openmp}, "SpMV OpenMP"}
#endif
  };

  for (size_t i = 0; i < sizeof(impls)/sizeof(impls[0]); i++) {
    test_impl(impls[i], argc, argv, test_cases);
  }
  return 0;
}
