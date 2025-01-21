#include "parsers/parse_mtx.h"
#include "test_utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_FILENAME "test_matrix.mtx"

static void test_mtx_invalid_file(void) {
  printf("Testing invalid file handling...\n");

  mtx_info_t info;
  assert(mtx_read_info("nonexistent.mtx", &info) == STATUS_FILE_ERROR);
  assert(mtx_read_info(NULL, &info) == STATUS_NULL_POINTER);
  assert(mtx_read_info(TEST_FILENAME, NULL) == STATUS_NULL_POINTER);

  printf("Invalid file tests passed\n");
}

static void test_mtx_basic_read(void) {
  printf("Testing basic matrix reading...\n");

  status_t status = create_test_mtx_file(TEST_FILENAME, 10, 10, 20, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  mtx_info_t info;
  status = mtx_read_info(TEST_FILENAME, &info);
  assert(status == STATUS_SUCCESS);
  assert(info.rows == 10);
  assert(info.cols == 10);
  assert(info.nnz == 20);
  assert(!info.is_symmetric);
  assert(!info.is_pattern);
  assert(!info.is_complex);

  delete_test_file(TEST_FILENAME);
  printf("Basic read tests passed\n");
}

static void test_mtx_pattern_matrix(void) {
  printf("Testing pattern matrix handling...\n");

  status_t status = create_test_mtx_file(TEST_FILENAME, 5, 5, 10, 0, 1, 0);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr(TEST_FILENAME, &matrix);
  assert(status == STATUS_SUCCESS);

  for (uint64_t i = 0; i < matrix->nnz; i++) {
    assert(matrix->values[i] == 1.0);
  }

  csr_free(matrix, NULL);
  delete_test_file(TEST_FILENAME);
  printf("Pattern matrix tests passed\n");
}

static void test_mtx_symmetric_matrix(void) {
  printf("Testing symmetric matrix handling...\n");

  status_t status = create_test_mtx_file(TEST_FILENAME, 4, 4, 6, 1, 0, 0);
  assert(status == STATUS_SUCCESS);

  mtx_info_t info;
  status = mtx_read_info(TEST_FILENAME, &info);
  assert(status == STATUS_SUCCESS);
  assert(info.is_symmetric);

  delete_test_file(TEST_FILENAME);
  printf("Symmetric matrix tests passed\n");
}

static void test_mtx_memory_estimate(void) {
  printf("Testing memory estimation...\n");

  status_t status = create_test_mtx_file(TEST_FILENAME, 100, 100, 500, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  size_t size;
  status = mtx_get_memory_estimate(TEST_FILENAME, MTX_FORMAT_CSR, &size);
  assert(status == STATUS_SUCCESS);
  assert(size > 0);

  status = mtx_get_memory_estimate(TEST_FILENAME, MTX_FORMAT_ELL, &size); // NOTE: this test is temporary until the ELL format is implemented
  assert(status == STATUS_NOT_IMPLEMENTED);

  delete_test_file(TEST_FILENAME);
  printf("Memory estimation tests passed\n");
}

static void test_mtx_format_conversion(void) {
  printf("Testing format conversion...\n");

  status_t status = create_test_mtx_file(TEST_FILENAME, 50, 50, 100, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  void *matrix;
  status = mtx_to_format(TEST_FILENAME, MTX_FORMAT_CSR, &matrix);
  assert(status == STATUS_SUCCESS);

  csr_free((csr_matrix *)matrix, NULL);

  status = mtx_to_format(TEST_FILENAME, MTX_FORMAT_ELL, &matrix); // NOTE: this test is temporary until the ELL format is implemented
  assert(status == STATUS_NOT_IMPLEMENTED);

  delete_test_file(TEST_FILENAME);
  printf("Format conversion tests passed\n");
}

static void test_mtx_edge_cases(void) {
  printf("Testing edge cases...\n");

  status_t status = create_test_mtx_file(TEST_FILENAME, 0, 0, 0, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  mtx_info_t info;
  status = mtx_read_info(TEST_FILENAME, &info);
  assert(status == STATUS_SUCCESS);

  status = create_test_mtx_file(TEST_FILENAME, 1, 1, 1, 0, 0, 0);
  assert(status == STATUS_SUCCESS);

  csr_matrix *matrix;
  status = mtx_to_csr(TEST_FILENAME, &matrix);
  assert(status == STATUS_SUCCESS);

  csr_free(matrix, NULL);
  delete_test_file(TEST_FILENAME);
  printf("Edge case tests passed\n");
}

static const test_case_t test_cases[] = {
  {"invalid_file", test_mtx_invalid_file},
  {"basic_read", test_mtx_basic_read},
  {"pattern_matrix", test_mtx_pattern_matrix},
  {"symmetric_matrix", test_mtx_symmetric_matrix},
  {"memory_estimate", test_mtx_memory_estimate},
  {"format_conversion", test_mtx_format_conversion},
  {"edge_cases", test_mtx_edge_cases},
  {NULL, NULL}
};

int main(int argc, char **argv) {
  return run_test_suite(argc, argv, test_cases);
}
