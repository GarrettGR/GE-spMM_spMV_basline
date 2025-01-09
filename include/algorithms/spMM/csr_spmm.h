#ifndef CSR_SPMM_H
#define CSR_SPMM_H

#include "formats/csr.h"
#include "utils/profiler.h"
#include "utils/status.h"
#include <stdint.h>

// status_t csr_spmm(const csr_matrix *A, const csr_matrix *B, csr_matrix **C, profile_context *prof);

/**
 * @brief Perform sparse matrix-matrix multiplication A*B = C where all matrices are in CSR format
 *
 * @param A First input CSR matrix
 * @param B Second input CSR matrix
 * @param[out] C Output CSR matrix
 * @param prof Optional profiler context (can be NULL)
 * @return status_t Status code
 */
status_t csr_spmm_sequential(const csr_matrix *A, const csr_matrix *B, csr_matrix **C, profile_context *prof);

/**
 * @brief Perform sparse matrix-matrix multiplication A*B = C where all matrices are in CSR format
 *
 * @param A First input CSR matrix
 * @param B Second input CSR matrix
 * @param[out] C Output CSR matrix
 * @param prof Optional profiler context (can be NULL)
 * @return status_t Status code
 */
status_t csr_spmm_openmp(const csr_matrix *A, const csr_matrix *B, csr_matrix **C, profile_context *prof);

/**
 * @brief Validate inputs for SpMM operation
 */
status_t csr_spmm_validate(const csr_matrix *A, const csr_matrix *B);

#endif // CSR_SPMM_H
