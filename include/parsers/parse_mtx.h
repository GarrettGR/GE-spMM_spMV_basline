#ifndef PARSE_MTX_H
#define PARSE_MTX_H

#include <stdint.h>
#include "utils/status.h"
#include "formats/csr.h"
#include "formats/pcsr.h"
#include "formats/bcsr.h"
#include "formats/coo.h"
#include "formats/ell.h"
#include "formats/hyb.h"
#include "formats/dia.h"

/**
 * @brief Types of supported matrix format to convert .mtx to
 */
typedef enum mtx_format {
  MTX_FORMAT_CSR,
  MTX_FORMAT_PCSR,
  MTX_FORMAT_BCSR,
  MTX_FORMAT_COO,
  MTX_FORMAT_ELL,
  MTX_FORMAT_HYB,
  MTX_FORMAT_DIA
} mtx_format_t;

/**
 * @brief Data structure to hold metadata for a matrix
 */
typedef struct mtx_info {
  uint64_t rows;
  uint64_t cols;
  uint64_t nnz;
  int is_symmetric;
  int is_pattern;
  int is_complex;
} mtx_info_t;

/**
 * @brief Read the metadata of a matrix from a file in Matrix Market format.
 * 
 * @param filename The name of the file to read.
 * @param info A pointer to a mtx_info_t structure where the metadata will be stored.
 *
 * @return STATUS_SUCCESS on success, or an error code on failure.
 */
status_t mtx_read_info(const char* filename, mtx_info_t* info);

/**
 * @brief Count the number of diagonal entries in a symmetric matrix
 * 
 * @param filename The name of the file to analyze
 * @param info Pointer to matrix info structure
 * @return uint64_t Number of diagonal entries
 */
uint64_t count_diagonal_entries(const char* filename, const mtx_info_t* info);

/**
 * @brief Validate the format of a matrix file.
 * 
 * @param filename The name of the file to validate.
 *
 * @return STATUS_SUCCESS if the file is in a valid format, or an error code on failure.
 */
status_t mtx_validate_format(const char* filename);

/**
 * @brief Read a matrix from a file in Matrix Market format and convert it to the specified format.
 * 
 * @param filename The name of the file to read.
 * @param format The format to convert the matrix to.
 * @param matrix A pointer to a pointer to the matrix structure where the read matrix will be stored.
 *
 * @return STATUS_SUCCESS on success, or an error code on failure.
 */
status_t mtx_to_format(const char* filename, mtx_format_t format, void** matrix);

/**
 * @brief Read a matrix from a file in Matrix Market format and convert it to CSR format.
 * 
 * @param filename The name of the file to read.
 * @param matrix A pointer to a pointer to a csr_matrix structure where the read matrix will be stored.
 *
 * @return STATUS_SUCCESS on success, or an error code on failure.
 */
status_t mtx_to_csr(const char* filename, csr_matrix** matrix);

// status_t mtx_to_pcsr(const char* filename, pcsr_matrix** matrix);
// status_t mtx_to_bcsr(const char* filename, bcsr_matrix** matrix);
// status_t mtx_to_coo(const char* filename, coo_matrix** matrix);
// status_t mtx_to_ell(const char* filename, ell_matrix** matrix);
// status_t mtx_to_hyb(const char* filename, hyb_matrix** matrix);
// status_t mtx_to_dia(const char* filename, dia_matrix** matrix);

/**
 * @brief Convert a mtx_format_t enum value to a string.
 *
 * @param format The mtx_format_t value to convert.
 *
 * @return A string representation of the format.
 */
const char* mtx_format_to_string(mtx_format_t format);

/**
 * @brief Get an estimate of the memory required to store a matrix in a specified format.
 *
 * @param filename The name of the file containing the matrix.
 * @param format The format to estimate memory for.
 * @param size A pointer to a size_t variable where the estimated size will be stored.
 *
 * @return STATUS_SUCCESS on success, or an error code on failure.
 */
status_t mtx_get_memory_estimate(const char* filename, mtx_format_t format, size_t* size);

#endif // PARSE_MTX_H
