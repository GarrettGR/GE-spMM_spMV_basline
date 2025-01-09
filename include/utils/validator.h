#ifndef VALIDATOR_H
#define VALIDATOR_H

#include "formats/csr.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Matrix pattern types
 */
typedef enum {
  PATTERN_RANDOM,
  PATTERN_DIAGONAL,
  PATTERN_DENSE,
  PATTERN_BANDED,
  PATTERN_BLOCK
} matrix_pattern_t;

/**
 * @brief Matrix pattern types
 */

/**
 * @brief Create a test CSR matrix with specified pattern
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zero elements
 * @param pattern Pattern type for matrix generation
 * @param[out] matrix Pointer to store created matrix
 * @return status_t Status code
 */
status_t create_test_matrix(uint64_t rows, uint64_t cols, uint64_t nnz, matrix_pattern_t pattern, csr_matrix** matrix);

/**
 * @brief Validate SpMM result against PETSc
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Result matrix to validate
 * @return bool True if results match
 */
bool validate_spmm_with_petsc(const csr_matrix* A, const csr_matrix* B, const csr_matrix* C);

/**
 * @brief Validate SpMV result against PETSc
 *
 * @param A Input matrix
 * @param x Input vector
 * @param y Result vector to validate
 * @return bool True if results match
 */
bool validate_spmv_with_petsc(const csr_matrix* A, const double* x, const double* y);

/**
 * @brief Compare two matrices for equality within tolerance
 *
 * @param A First matrix
 * @param B Second matrix
 * @return bool True if matrices match within tolerance
 */
bool compare_matrices(const csr_matrix* A, const csr_matrix* B);

/**
 * @brief Compare two vectors for equality within tolerance
 *
 * @param x First vector
 * @param y Second vector
 * @param size Vector size
 * @return bool True if vectors match within tolerance
 */
bool compare_vectors(const double* x, const double* y, uint64_t size);

#endif // VALIDATOR_H
