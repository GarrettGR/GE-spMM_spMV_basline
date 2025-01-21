#include "parsers/parse_mtx.h"
#include "test_utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static status_t
create_specific_test_matrix(const char *filename, uint64_t *row_indices, uint64_t *col_indices, double *values, 
                            uint64_t rows, uint64_t cols, uint64_t nnz, int is_symmetric, int is_pattern, int is_complex) {

  FILE *fp = fopen(filename, "w");
  if (!fp) return STATUS_FILE_ERROR;

  fprintf(fp, "%%%%MatrixMarket matrix coordinate %s %s\n", is_pattern ? "pattern" : (is_complex ? "complex" : "real"), is_symmetric ? "symmetric" : "general");
  fprintf(fp, "%llu %llu %llu\n", rows, cols, nnz);

  for (uint64_t i = 0; i < nnz; i++) {
    if (is_pattern) {
      fprintf(fp, "%llu %llu\n", row_indices[i], col_indices[i]);
    } else if (is_complex) {
      fprintf(fp, "%llu %llu %.17g %.17g\n", row_indices[i], col_indices[i], values[i], 0.0);
    } else {
      fprintf(fp, "%llu %llu %.17g\n", row_indices[i], col_indices[i], values[i]);
    }
  }

  fclose(fp);
  return STATUS_SUCCESS;
}

static void verify_csr_properties(const csr_matrix *matrix) {
  assert(matrix != NULL);
  assert(matrix->values != NULL);
  assert(matrix->col_idx != NULL);
  assert(matrix->row_ptr != NULL);

  assert(matrix->row_ptr[0] == 0);
  assert(matrix->row_ptr[matrix->rows] == matrix->nnz);

  for (uint64_t i = 0; i < matrix->rows; i++) {
    assert(matrix->row_ptr[i] <= matrix->row_ptr[i + 1]);
  }

  for (uint64_t i = 0; i < matrix->rows; i++) {
    for (uint64_t j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1] - 1; j++) {
      assert(matrix->col_idx[j] < matrix->col_idx[j + 1]);
    }
  }

  for (uint64_t i = 0; i < matrix->nnz; i++) {
    assert(matrix->col_idx[i] < matrix->cols);
  }
}

static void test_mtx_null_arguments(void) {
  printf("Testing null argument handling...\n");

  mtx_info_t info;
  assert(mtx_read_info(NULL, &info) == STATUS_NULL_POINTER);
  assert(mtx_read_info("test.mtx", NULL) == STATUS_NULL_POINTER);

  csr_matrix *matrix;
  assert(mtx_to_csr(NULL, &matrix) == STATUS_NULL_POINTER);
  assert(mtx_to_csr("test.mtx", NULL) == STATUS_NULL_POINTER);

  size_t size;
  assert(mtx_get_memory_estimate(NULL, MTX_FORMAT_CSR, &size) == STATUS_NULL_POINTER);
  assert(mtx_get_memory_estimate("test.mtx", MTX_FORMAT_CSR, NULL) == STATUS_NULL_POINTER);

  printf("Null argument tests passed\n");
}

static void test_mtx_diagonal_matrix(void) {
  printf("Testing diagonal matrix handling...\n");

  const char *filename = "diagonal.mtx";
  uint64_t size = 5;
  uint64_t row_indices[] = {1, 2, 3, 4, 5};
  uint64_t col_indices[] = {1, 2, 3, 4, 5};
  double values[] = {1.0, 2.0, 3.0, 4.0, 5.0};

  status_t status = create_specific_test_matrix(filename, row_indices, col_indices, values, size, size, size, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr(filename, &matrix);
  assert(status == STATUS_SUCCESS);

  verify_csr_properties(matrix);

  for (uint64_t i = 0; i < size; i++) {
    assert(matrix->values[i] == (double)(i + 1));
    assert(matrix->col_idx[i] == i);
    assert(matrix->row_ptr[i] == i);
  }
  assert(matrix->row_ptr[size] == size);

  csr_free(matrix, NULL);
  delete_test_file(filename);
  printf("Diagonal matrix tests passed\n");
}

static void test_mtx_symmetric_matrix(void) {
  printf("Testing symmetric matrix handling...\n");

  const char *filename = "symmetric.mtx";
  uint64_t rows = 3, cols = 3, nnz = 4;
  uint64_t row_indices[] = {1, 1, 2, 3};
  uint64_t col_indices[] = {1, 2, 2, 3};
  double values[] = {1.0, 2.0, 3.0, 4.0};

  status_t status = create_specific_test_matrix(filename, row_indices, col_indices, values, rows, cols, nnz, 1, 0, 0);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr(filename, &matrix);
  assert(status == STATUS_SUCCESS);

  verify_csr_properties(matrix);

  double value1, value2;

  status = csr_get_value(matrix, 0, 1, &value1, NULL);
  assert(status == STATUS_SUCCESS);
  status = csr_get_value(matrix, 1, 0, &value2, NULL);
  assert(status == STATUS_SUCCESS);
  assert(value1 == value2);

  csr_free(matrix, NULL);
  delete_test_file(filename);
  printf("Symmetric matrix tests passed\n");
}

static void test_mtx_pattern_matrix(void) {
  printf("Testing pattern matrix handling...\n");

  const char *filename = "pattern.mtx";
  uint64_t rows = 4, cols = 4, nnz = 5;
  uint64_t row_indices[] = {1, 1, 2, 3, 4};
  uint64_t col_indices[] = {1, 2, 2, 3, 4};
  double values[] = {0};

  status_t status = create_specific_test_matrix(filename, row_indices, col_indices, values, rows, cols, nnz, 0, 1, 0);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr(filename, &matrix);
  assert(status == STATUS_SUCCESS);

  verify_csr_properties(matrix);

  for (uint64_t i = 0; i < matrix->nnz; i++) {
    assert(matrix->values[i] == 1.0);
  }

  csr_free(matrix, NULL);
  delete_test_file(filename);
  printf("Pattern matrix tests passed\n");
}

static void test_mtx_complex_matrix(void) {
  printf("Testing complex matrix handling...\n");

  const char *filename = "complex.mtx";
  uint64_t rows = 3, cols = 3, nnz = 4;
  uint64_t row_indices[] = {1, 1, 2, 3};
  uint64_t col_indices[] = {1, 2, 2, 3};
  double values[] = {1.0, 2.0, 3.0, 4.0};

  status_t status = create_specific_test_matrix(filename, row_indices, col_indices, values, rows, cols, nnz, 0, 0, 1);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr(filename, &matrix);
  assert(status == STATUS_SUCCESS);

  verify_csr_properties(matrix);

  for (uint64_t i = 0; i < matrix->nnz; i++) {
    assert(matrix->values[i] == values[i]);
  }

  csr_free(matrix, NULL);
  delete_test_file(filename);
  printf("Complex matrix tests passed\n");
}

static void test_mtx_large_matrix(void) {
  printf("Testing large matrix handling...\n");

  const char *filename = "large.mtx";
  uint64_t rows = 1000, cols = 1000, nnz = 1000;
  uint64_t *row_indices = malloc(nnz * sizeof(uint64_t));
  uint64_t *col_indices = malloc(nnz * sizeof(uint64_t));
  double *values = malloc(nnz * sizeof(double));

  assert(row_indices && col_indices && values);

  for (uint64_t i = 0; i < nnz; i++) {
    row_indices[i] = i + 1;
    col_indices[i] = i + 1;
    values[i] = i + 1.0;
  }

  status_t status = create_specific_test_matrix(filename, row_indices, col_indices, values, rows, cols, nnz, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr(filename, &matrix);
  assert(status == STATUS_SUCCESS);

  verify_csr_properties(matrix);

  csr_free(matrix, NULL);
  delete_test_file(filename);
  free(row_indices);
  free(col_indices);
  free(values);
  printf("Large matrix tests passed\n");
}

static void test_mtx_memory_estimate(void) {
  printf("Testing memory estimation...\n");

  const char *filename = "estimate.mtx";
  uint64_t rows = 100, cols = 100, nnz = 500;
  uint64_t *row_indices = malloc(nnz * sizeof(uint64_t));
  uint64_t *col_indices = malloc(nnz * sizeof(uint64_t));
  double *values = malloc(nnz * sizeof(double));

  assert(row_indices && col_indices && values);

  for (uint64_t i = 0; i < nnz; i++) {
    row_indices[i] = (i % rows) + 1;
    col_indices[i] = (i % cols) + 1;
    values[i] = i + 1.0;
  }

  status_t status = create_specific_test_matrix(filename, row_indices, col_indices, values, rows, cols, nnz, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  size_t size;
  status = mtx_get_memory_estimate(filename, MTX_FORMAT_CSR, &size);
  assert(status == STATUS_SUCCESS);

  size_t min_size = sizeof(csr_matrix) + (nnz * sizeof(double)) + (nnz * sizeof(uint64_t)) + ((rows + 1) * sizeof(uint64_t));

  assert(size >= min_size);

  free(row_indices);
  free(col_indices);
  free(values);
  delete_test_file(filename);
  printf("Memory estimation tests passed\n");
}

static void test_mtx_edge_cases(void) {
  printf("Testing edge cases...\n");

  const char *filename = "empty.mtx";
  status_t status = create_specific_test_matrix(filename, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  mtx_info_t info;
  status = mtx_read_info(filename, &info);
  assert(status == STATUS_SUCCESS);
  assert(info.rows == 0 && info.cols == 0 && info.nnz == 0);

  delete_test_file(filename);

  uint64_t row_indices[] = {1};
  uint64_t col_indices[] = {1};
  double values[] = {1.0};

  status = create_specific_test_matrix("single.mtx", row_indices, col_indices, values, 1, 1, 1, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr("single.mtx", &matrix);
  assert(status == STATUS_SUCCESS);

  verify_csr_properties(matrix);
  assert(matrix->nnz == 1);
  assert(matrix->values[0] == 1.0);

  csr_free(matrix, NULL);
  delete_test_file("single.mtx");
  printf("Edge case tests passed\n");
}

static void test_mtx_invalid_format(void) {
  printf("Testing invalid format handling...\n");

  const char *filename = "invalid.mtx";
  FILE *fp = fopen(filename, "w");
  assert(fp != NULL);
  fprintf(fp, "Invalid file format\n");
  fclose(fp);

  mtx_info_t info;
  status_t status = mtx_read_info(filename, &info);
  assert(status == STATUS_INVALID_FORMAT);

  delete_test_file(filename);
  printf("Invalid format tests passed\n");
}

static const test_case_t test_cases[] = {
  {"null_arguments", test_mtx_null_arguments},
  {"invalid_format", test_mtx_invalid_format},
  {"diagonal_matrix", test_mtx_diagonal_matrix},
  {"symmetric_matrix", test_mtx_symmetric_matrix},
  {"pattern_matrix", test_mtx_pattern_matrix},
  {"complex_matrix", test_mtx_complex_matrix},
  {"large_matrix", test_mtx_large_matrix},
  {"memory_estimate", test_mtx_memory_estimate},
  {"edge_cases", test_mtx_edge_cases},
  {NULL, NULL}
};

int main(int argc, char **argv) {
  return run_test_suite(argc, argv, test_cases);
}
