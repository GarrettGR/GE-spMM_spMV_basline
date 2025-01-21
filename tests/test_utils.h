#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "formats/csr.h"
#include <stdbool.h>

/**
 * @brief Test case structure
 */
typedef struct {
  const char* name;
  void (*func)(void);
} test_case_t;

/**
 * @brief Matrix pattern types
 */
typedef enum {
  PATTERN_DIAGONAL,
  PATTERN_RANDOM,
  PATTERN_DENSE,
  PATTERN_BANDED,
  PATTERN_BLOCK
} matrix_pattern_t;

/**
 * @brief Create a test matrix with specified pattern
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zero elements
 * @param pattern Pattern type for matrix generation
 * @param[out] matrix Pointer to store created matrix
 *
 * @return status_t Status code
 */
status_t create_test_matrix(uint64_t rows, uint64_t cols, uint64_t nnz, matrix_pattern_t pattern, csr_matrix** matrix);

/**
 * @brief Create a dense matrix with specified density
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param density Density of non-zero elements
 * @param[out] matrix Pointer to store created matrix
 *
 * @return status_t Status code
 */
status_t create_dense_matrix(uint64_t rows, uint64_t cols, double density, csr_matrix** matrix);


/**
 * @brief Compute SpMV on dense matrix
 *
 * @param A Input matrix
 * @param x Input vector
 * @param y Output vector
 *
 * @return status_t Status code
*/
status_t compute_dense_spmv(const csr_matrix* A, const double* x, double* y);

/**
 * @brief Compute SpMM on dense matrix
 * 
 * @param A First input matrix
 * @param B Second input matrix
 * @param[out] C Result matrix
 *
 * @return status_t Status code
*/
status_t compute_dense_spmm(const csr_matrix* A, const csr_matrix* B, csr_matrix** C);

/**
 * @brief Compare two matrices for equality within tolerance
 *
 * @param A First matrix
 * @param B Second matrix
 *
 * @return bool True if matrices match within tolerance
 */
bool compare_matrices(const csr_matrix* A, const csr_matrix* B);

/**
 * @brief Compare two vectors for equality within tolerance
 *
 * @param x First vector
 * @param y Second vector
 * @param size Size of vectors
 *
 * @return bool True if vectors match within tolerance
 */
bool compare_vectors(const double* x, const double* y, uint64_t size);

/**
 * @brief Run a test suite
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @param test_cases Array of test cases
 *
 * @return int Exit code
 */
int run_test_suite(int argc, char** argv, const test_case_t* test_cases);

/**
 * @brief Print a matrix in CSR format
 *
 * @param matrix Pointer to the matrix
 */
void print_matrix(const csr_matrix* matrix);

/**
 * @brief Create a temporary matrix market (.mtx) file for testing
 *
 * @param filename Output filename
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 * @param is_symmetric Whether matrix is symmetric
 * @param is_pattern Whether matrix is pattern only
 * @param is_complex Whether matrix has complex values
 * @return status_t Status code
 */
status_t create_test_mtx_file(const char* filename, uint64_t rows, uint64_t cols, uint64_t nnz, int is_symmetric, int is_pattern, int is_complex);

/**
 * @brief Delete a temporary test file
 */
void delete_test_file(const char* filename);

#endif // TEST_HELPERS_H