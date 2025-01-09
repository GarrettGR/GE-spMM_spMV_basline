#include "formats/csr.h"
#include "test_utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_csr_create(void);
static void test_csr_create_from_arrays(void);
static void test_csr_copy(void);
static void test_csr_get_value(void);
static void test_csr_validation(void);
static void test_csr_memory_management(void);
static void test_csr_large_matrix(void);
static void test_csr_edge_cases(void);

static void test_csr_create(void) {
  printf("Testing CSR creation...\n");

  csr_matrix *matrix = NULL;
  status_t status;

  status = csr_create(0, 10, 5, &matrix, NULL);
  assert(status == STATUS_INVALID_DIMENSIONS);

  status = csr_create(10, 0, 5, &matrix, NULL);
  assert(status == STATUS_INVALID_DIMENSIONS);

  status = csr_create(10, 10, 0, &matrix, NULL);
  assert(status == STATUS_INVALID_DIMENSIONS);

  status = csr_create(10, 10, 5, &matrix, NULL);
  assert(status == STATUS_SUCCESS);
  assert(matrix != NULL);
  assert(matrix->rows == 10);
  assert(matrix->cols == 10);
  assert(matrix->nnz == 5);
  assert(matrix->values != NULL);
  assert(matrix->col_idx != NULL);
  assert(matrix->row_ptr != NULL);
  assert(matrix->own_data == 1);

  csr_free(matrix, NULL);
  printf("CSR creation tests passed\n");
}

static void test_csr_create_from_arrays(void) {
  printf("Testing CSR creation from arrays...\n");

  const uint64_t rows = 3;
  const uint64_t cols = 3;
  const uint64_t nnz = 4;

  double *values = (double *)malloc(nnz * sizeof(double));
  uint64_t *col_idx = (uint64_t *)malloc(nnz * sizeof(uint64_t));
  uint64_t *row_ptr = (uint64_t *)malloc((rows + 1) * sizeof(uint64_t));

  values[0] = 1.0;
  values[1] = 2.0;
  values[2] = 3.0;
  values[3] = 4.0;
  col_idx[0] = 0;
  col_idx[1] = 1;
  col_idx[2] = 1;
  col_idx[3] = 2;
  row_ptr[0] = 0;
  row_ptr[1] = 2;
  row_ptr[2] = 3;
  row_ptr[3] = 4;

  csr_matrix *matrix = NULL;
  status_t status = csr_create_from_arrays(rows, cols, nnz, values, col_idx, row_ptr, &matrix, NULL);

  assert(status == STATUS_SUCCESS);
  assert(matrix != NULL);
  assert(matrix->rows == rows);
  assert(matrix->cols == cols);
  assert(matrix->nnz == nnz);
  assert(matrix->values == values);
  assert(matrix->col_idx == col_idx);
  assert(matrix->row_ptr == row_ptr);
  assert(matrix->own_data == 0);

  status = csr_validate(matrix);
  assert(status == STATUS_SUCCESS);

  csr_free(matrix, NULL);
  free(values);
  free(col_idx);
  free(row_ptr);
  printf("CSR creation from arrays tests passed\n");
}

static void test_csr_copy(void) {
  printf("Testing CSR copy...\n");

  const uint64_t rows = 3;
  const uint64_t cols = 3;
  const uint64_t nnz = 4;

  csr_matrix *src = NULL;
  status_t status = csr_create(rows, cols, nnz, &src, NULL);
  assert(status == STATUS_SUCCESS);

  src->values[0] = 1.0;
  src->values[1] = 2.0;
  src->values[2] = 3.0;
  src->values[3] = 4.0;
  src->col_idx[0] = 0;
  src->col_idx[1] = 1;
  src->col_idx[2] = 1;
  src->col_idx[3] = 2;
  src->row_ptr[0] = 0;
  src->row_ptr[1] = 2;
  src->row_ptr[2] = 3;
  src->row_ptr[3] = 4;

  csr_matrix *dest = NULL;
  status = csr_copy(src, &dest, NULL);
  assert(status == STATUS_SUCCESS);

  assert(dest->rows == src->rows);
  assert(dest->cols == src->cols);
  assert(dest->nnz == src->nnz);
  assert(memcmp(dest->values, src->values, nnz * sizeof(double)) == 0);
  assert(memcmp(dest->col_idx, src->col_idx, nnz * sizeof(uint64_t)) == 0);
  assert(memcmp(dest->row_ptr, src->row_ptr, (rows + 1) * sizeof(uint64_t)) == 0);

  assert(dest->values != src->values);
  assert(dest->col_idx != src->col_idx);
  assert(dest->row_ptr != src->row_ptr);

  csr_free(src, NULL);
  csr_free(dest, NULL);
  printf("CSR copy tests passed\n");
}

static void test_csr_get_value(void) {
  printf("Testing CSR value retrieval...\n");

  const uint64_t rows = 3;
  const uint64_t cols = 3;
  const uint64_t nnz = 4;

  csr_matrix *matrix = NULL;
  status_t status = csr_create(rows, cols, nnz, &matrix, NULL);
  assert(status == STATUS_SUCCESS);

  matrix->values[0] = 1.0;
  matrix->values[1] = 2.0;
  matrix->values[2] = 3.0;
  matrix->values[3] = 4.0;
  matrix->col_idx[0] = 0;
  matrix->col_idx[1] = 1;
  matrix->col_idx[2] = 1;
  matrix->col_idx[3] = 2;
  matrix->row_ptr[0] = 0;
  matrix->row_ptr[1] = 2;
  matrix->row_ptr[2] = 3;
  matrix->row_ptr[3] = 4;

  double value;

  status = csr_get_value(matrix, 0, 0, &value, NULL);
  assert(status == STATUS_SUCCESS);
  assert(value == 1.0);

  status = csr_get_value(matrix, 0, 1, &value, NULL);
  assert(status == STATUS_SUCCESS);
  assert(value == 2.0);

  status = csr_get_value(matrix, 0, 2, &value, NULL);
  assert(status == STATUS_SUCCESS);
  assert(value == 0.0);

  status = csr_get_value(matrix, rows, 0, &value, NULL);
  assert(status == STATUS_INVALID_DIMENSIONS);

  status = csr_get_value(matrix, 0, cols, &value, NULL);
  assert(status == STATUS_INVALID_DIMENSIONS);

  csr_free(matrix, NULL);
  printf("CSR value retrieval tests passed\n");
}

static void test_csr_validation(void) {
  printf("Testing CSR validation...\n");

  csr_matrix *matrix = NULL;
  status_t status = csr_create(3, 3, 4, &matrix, NULL);
  assert(status == STATUS_SUCCESS);

  matrix->values[0] = 1.0;
  matrix->values[1] = 2.0;
  matrix->values[2] = 3.0;
  matrix->values[3] = 4.0;
  matrix->col_idx[0] = 0;
  matrix->col_idx[1] = 1;
  matrix->col_idx[2] = 1;
  matrix->col_idx[3] = 2;
  matrix->row_ptr[0] = 0;
  matrix->row_ptr[1] = 2;
  matrix->row_ptr[2] = 3;
  matrix->row_ptr[3] = 4;

  status = csr_validate(matrix);
  assert(status == STATUS_SUCCESS);

  matrix->row_ptr[1] = 4;
  status = csr_validate(matrix);
  assert(status == STATUS_INVALID_SPARSE_STRUCTURE);
  matrix->row_ptr[1] = 2;

  matrix->col_idx[0] = 3;
  status = csr_validate(matrix);
  assert(status == STATUS_INVALID_DIMENSIONS);
  matrix->col_idx[0] = 0;

  matrix->col_idx[0] = 1;
  matrix->col_idx[1] = 0;
  status = csr_validate(matrix);
  assert(status == STATUS_UNSORTED_INDICES);

  csr_free(matrix, NULL);
  printf("CSR validation tests passed\n");
}

static void test_csr_memory_management(void) {
  printf("Testing CSR memory management...\n");

  csr_matrix *matrix1 = NULL;
  status_t status = csr_create(10, 10, 5, &matrix1, NULL);
  assert(status == STATUS_SUCCESS);
  assert(matrix1->own_data == 1);

  // void *values_ptr = matrix1->values;
  // void *col_idx_ptr = matrix1->col_idx;
  // void *row_ptr_ptr = matrix1->row_ptr;

  csr_free(matrix1, NULL);

  double *values = (double *)malloc(5 * sizeof(double));
  uint64_t *col_idx = (uint64_t *)malloc(5 * sizeof(uint64_t));
  uint64_t *row_ptr = (uint64_t *)malloc(11 * sizeof(uint64_t));

  csr_matrix *matrix2 = NULL;
  status = csr_create_from_arrays(10, 10, 5, values, col_idx, row_ptr, &matrix2, NULL);
  assert(status == STATUS_SUCCESS);
  assert(matrix2->own_data == 0);

  csr_free(matrix2, NULL);

  free(values);
  free(col_idx);
  free(row_ptr);

  printf("CSR memory management tests passed\n");
}

static void test_csr_large_matrix(void) {
  printf("Testing CSR with large matrix...\n");

  const uint64_t rows = 1000000;
  const uint64_t cols = 1000000;
  const uint64_t nnz = 5000000;

  csr_matrix *matrix = NULL;
  status_t status = csr_create(rows, cols, nnz, &matrix, NULL);
  assert(status == STATUS_SUCCESS);

  for (uint64_t i = 0; i < rows; i++) {
    matrix->row_ptr[i] = (i * nnz) / rows;
  }
  matrix->row_ptr[rows] = nnz;

  for (uint64_t i = 0; i < nnz; i++) {
    matrix->col_idx[i] = i % cols;
    matrix->values[i] = 1.0;
  }

  status = csr_validate(matrix);
  assert(status == STATUS_SUCCESS);

  csr_free(matrix, NULL);
  printf("CSR large matrix tests passed\n");
}

static void test_csr_edge_cases(void) {
  printf("\n=== Testing CSR edge cases... ===\n");
  status_t status;

  printf("Testing empty matrix creation...\n");
  csr_matrix *empty_matrix = NULL;
  status = csr_create(0, 0, 0, &empty_matrix, NULL);
  printf("Empty matrix creation status: %d\n", status);
  assert(status == STATUS_SUCCESS);
  
  printf("Validating empty matrix...\n");
  status = csr_validate(empty_matrix);
  printf("Empty matrix validation status: %d\n", status);
  assert(status == STATUS_SUCCESS);
  
  printf("Freeing empty matrix...\n");
  csr_free(empty_matrix, NULL);

  printf("\nTesting diagonal matrix creation...\n");
  csr_matrix *matrix1 = NULL;
  status = csr_create(5, 5, 5, &matrix1, NULL);
  printf("Diagonal matrix creation status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  printf("Filling diagonal matrix...\n");
  for (uint64_t i = 0; i < 5; i++) {
    matrix1->row_ptr[i] = i;
    matrix1->col_idx[i] = i;
    matrix1->values[i] = 1.0;
    printf("Set row_ptr[%llu]=%llu, col_idx[%llu]=%llu, values[%llu]=%.1f\n", i, matrix1->row_ptr[i], i, matrix1->col_idx[i], i, matrix1->values[i]);
  }
  matrix1->row_ptr[5] = 5;
  printf("Set final row_ptr[5]=%llu\n", matrix1->row_ptr[5]);

  printf("Validating diagonal matrix...\n");
  status = csr_validate(matrix1);
  printf("Diagonal matrix validation status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  printf("\nTesting matrix with empty rows...\n");
  csr_matrix *matrix2 = NULL;
  status = csr_create(3, 3, 3, &matrix2, NULL);
  printf("Matrix2 creation status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  printf("Setting up matrix2 structure...\n");
  matrix2->row_ptr[0] = 0;
  matrix2->row_ptr[1] = 3;
  matrix2->row_ptr[2] = 3;
  matrix2->row_ptr[3] = 3;
  printf("row_ptr values: [%llu, %llu, %llu, %llu]\n", matrix2->row_ptr[0], matrix2->row_ptr[1], matrix2->row_ptr[2], matrix2->row_ptr[3]);

  matrix2->col_idx[0] = 0;
  matrix2->col_idx[1] = 1;
  matrix2->col_idx[2] = 2;
  printf("col_idx values: [%llu, %llu, %llu]\n", matrix2->col_idx[0], matrix2->col_idx[1], matrix2->col_idx[2]);

  matrix2->values[0] = 1.0;
  matrix2->values[1] = 2.0;
  matrix2->values[2] = 3.0;
  printf("values: [%.1f, %.1f, %.1f]\n", matrix2->values[0], matrix2->values[1], matrix2->values[2]);

  printf("Validating matrix2...\n");
  status = csr_validate(matrix2);
  printf("Matrix2 validation status: %d\n", status);
  assert(status == STATUS_SUCCESS);

  printf("\nTesting NULL pointer handling...\n");
  status = csr_create(1, 1, 1, NULL, NULL);
  printf("NULL pointer test status: %d\n", status);
  assert(status == STATUS_NULL_POINTER);

  printf("\nCleaning up...\n");
  csr_free(matrix1, NULL);
  csr_free(matrix2, NULL);
  printf("=== CSR edge cases tests passed ===\n\n");
}

static const test_case_t test_cases[] = {
  {"create", test_csr_create},
  {"create_from_arrays", test_csr_create_from_arrays},
  {"copy", test_csr_copy},
  {"get_value", test_csr_get_value},
  {"validation", test_csr_validation},
  {"memory_management", test_csr_memory_management},
  {"large_matrix", test_csr_large_matrix},
  {"edge_cases", test_csr_edge_cases}
};

int main(int argc, char **argv) {
  return run_test_suite(argc, argv, test_cases);
}