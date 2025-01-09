#ifndef CSR_SPMV_H
#define CSR_SPMV_H

#include <stdint.h>
#include "formats/csr.h"
#include "utils/profiler.h"
#include "utils/status.h"

// status_t csr_spmv(const csr_matrix* matrix, const double* x, double* y, profile_context* prof);


/**
 * @brief Perform sparse matrix-vector multiplication Ax = B where A is in CSR format
 *
 * @param matrix Input CSR matrix A
 * @param x Input dense vector x
 * @param[out] y Output dense vector y
 * @param prof Optional profiler context (can be NULL)
 * @return status_t Status code
 */
status_t csr_spmv_sequential(const csr_matrix* matrix, const double* x, double* y, profile_context* prof);

/**
 * @brief Perform sparse matrix-vector multiplication Ax = B where A is in CSR format
 *
 * @param matrix Input CSR matrix A
 * @param x Input dense vector x
 * @param[out] y Output dense vector y
 * @param prof Optional profiler context (can be NULL)
 * @return status_t Status code
 */
status_t csr_spmv_openmp(const csr_matrix* matrix, const double* x, double* y, profile_context* prof);

/**
 * @brief Validate inputs for SpMV operation
 *
 * @param matrix Input CSR matrix
 * @param x Input vector
 * @param y Output vector
 * @param x_size Size of input vector
 * @param y_size Size of output vector
 * @return status_t Status code
 */
status_t csr_spmv_validate(const csr_matrix* matrix, const double* x, const double* y, uint64_t x_size, uint64_t y_size);

#endif // CSR_SPMV_H
