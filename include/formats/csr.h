#ifndef CSR_H
#define CSR_H

// #include <stddef.h>
#include <stdint.h>
#include "utils/profiler.h"
#include "utils/status.h"

/**
 * @brief Compressed Sparse Row (CSR) matrix format
 */
typedef struct {
  uint64_t rows;
  uint64_t cols;
  uint64_t nnz;

  double* values;
  uint64_t* col_idx;
  uint64_t* row_ptr;

  uint8_t own_data:1;
} csr_matrix;

/**
 * @brief Create a new CSR matrix with allocated memory
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zero elements
 * @param[out] matrix Pointer to matrix structure to initialize
 * @param prof Optional profiler context (can be NULL)
 *
 * @return status_t Status code
 */
status_t csr_create(uint64_t rows, uint64_t cols, uint64_t nnz, csr_matrix** matrix, profile_context* prof);

/**
 * @brief Create a CSR matrix from existing arrays (zero-copy)
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zero elements
 * @param values Array of non-zero values (nnz entries)
 * @param col_idx Array of column indices (nnz entries)
 * @param row_ptr Array of row pointers (rows + 1 entries)
 * @param[out] matrix Pointer to matrix structure to initialize
 * @param prof Optional profiler context (can be NULL)
 *
 * @return status_t Status code
 */
status_t csr_create_from_arrays(uint64_t rows, uint64_t cols, uint64_t nnz, double* values,
  uint64_t* col_idx, uint64_t* row_ptr, csr_matrix** matrix, profile_context* prof);

/**
 * @brief Free memory associated with a CSR matrix
 *
 * @param matrix Matrix to free
 * @param prof Optional profiler context (can be NULL)
 */
void csr_free(csr_matrix* matrix, profile_context* prof);

/**
 * @brief Copy a CSR matrix
 *
 * @param src Source matrix
 * @param[out] dest Pointer to destination matrix
 * @param prof Optional profiler context (can be NULL)
 *
 * @return status_t Status code
 */
status_t csr_copy(const csr_matrix* src, csr_matrix** dest, profile_context* prof);

/**
 * @brief Get value at specific row and column
 *
 * @param matrix Input matrix
 * @param row Row index
 * @param col Column index
 * @param[out] value Pointer to store the value
 * @param prof Optional profiler context (can be NULL)
 *
 * @return status_t Status code
 */
status_t csr_get_value(const csr_matrix* matrix, uint64_t row, uint64_t col, double* value, profile_context* prof);

/**
 * @brief Set value at specific row and column
 *
 * @param matrix Input matrix
 * @param row Row index
 * @param col Column index
 * @param value Value to set
 * @param prof Optional profiler context (can be NULL)
 *
 * @return status_t Status code
 */
status_t csr_set_value(csr_matrix* matrix, uint64_t row, uint64_t col, double value, profile_context* prof);

/**
 * @brief Verify that a CSR matrix is valid
 *
 * @param matrix Matrix to validate
 *
 * @return status_t Status code
 */
status_t csr_validate(const csr_matrix* matrix);

#endif // CSR_H
