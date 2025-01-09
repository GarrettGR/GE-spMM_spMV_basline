#include "formats/csr.h"
#include "utils/profiler.h"
#include "utils/status.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static status_t calculate_memory_size(uint64_t rows, uint64_t cols, uint64_t nnz, size_t* total_size);

status_t csr_create(uint64_t rows, uint64_t cols, uint64_t nnz, csr_matrix** matrix, profile_context* prof) {
  if (!matrix) return STATUS_NULL_POINTER;
  if ( !(rows == 0 && cols == 0 && nnz == 0) && (rows == 0 || cols == 0 || nnz == 0)) return STATUS_INVALID_DIMENSIONS;

  if (prof) profile_start_init(prof);

  size_t total_size;
  status_t status = calculate_memory_size(rows, cols, nnz, &total_size);
  if (status != STATUS_SUCCESS) return status;

  if (prof) profile_record_memory(prof, total_size);

  *matrix = (csr_matrix*)malloc(sizeof(csr_matrix));
  if (!*matrix) return STATUS_ALLOCATION_FAILED;

  (*matrix)->rows = rows;
  (*matrix)->cols = cols;
  (*matrix)->nnz = nnz;
  (*matrix)->own_data = 1;

  size_t values_size = nnz * sizeof(double);
  size_t col_idx_size = nnz * sizeof(uint64_t);
  size_t row_ptr_size = (rows + 1) * sizeof(uint64_t);

#ifdef _WIN32
  (*matrix)->values = (double*)_aligned_malloc(values_size, 64);
  (*matrix)->col_idx = (uint64_t*)_aligned_malloc(col_idx_size, 64);
  (*matrix)->row_ptr = (uint64_t*)_aligned_malloc(row_ptr_size, 64);
#else
  if (posix_memalign((void**)&(*matrix)->values, 64, values_size) ||
    posix_memalign((void**)&(*matrix)->col_idx, 64, col_idx_size) ||
    posix_memalign((void**)&(*matrix)->row_ptr, 64, row_ptr_size)) {
    csr_free(*matrix, prof);
    return STATUS_ALLOCATION_FAILED;
  }
#endif

  if (!(*matrix)->values || !(*matrix)->col_idx || !(*matrix)->row_ptr) {
    csr_free(*matrix, prof);
    return STATUS_ALLOCATION_FAILED;
  }

  memset((*matrix)->row_ptr, 0, row_ptr_size);

  if (prof) profile_end_init(prof);
  return STATUS_SUCCESS;
}

status_t csr_create_from_arrays(uint64_t rows, uint64_t cols, uint64_t nnz, double* values, uint64_t* col_idx, 
                                uint64_t* row_ptr, csr_matrix** matrix, profile_context* prof) {
  if (!matrix) return STATUS_NULL_POINTER;
  if (!values || !col_idx || !row_ptr) return STATUS_NULL_POINTER;
  if (rows == 0 || cols == 0 || nnz == 0) return STATUS_INVALID_DIMENSIONS;

  if (prof) profile_start_init(prof);

  *matrix = (csr_matrix*)malloc(sizeof(csr_matrix));
  if (!*matrix) return STATUS_ALLOCATION_FAILED;

  (*matrix)->rows = rows;
  (*matrix)->cols = cols;
  (*matrix)->nnz = nnz;
  (*matrix)->values = values;
  (*matrix)->col_idx = col_idx;
  (*matrix)->row_ptr = row_ptr;
  (*matrix)->own_data = 0;

  if (prof) profile_record_memory(prof, sizeof(csr_matrix));  // NOTE: This will only be the struct size ???

  if (prof) profile_end_init(prof);
  return STATUS_SUCCESS;
}

void csr_free(csr_matrix* matrix, profile_context* prof) {
  if (!matrix) return;

  if (prof) profile_start_init(prof);

  if (matrix->own_data) {
#ifdef _WIN32
    _aligned_free(matrix->values);
    _aligned_free(matrix->col_idx);
    _aligned_free(matrix->row_ptr);
#else
    free(matrix->values);
    free(matrix->col_idx);
    free(matrix->row_ptr);
#endif
  }

  free(matrix);

  if (prof) {
    profile_record_memory(prof, 0);
    profile_end_init(prof);
  }
}

status_t csr_copy(const csr_matrix* src, csr_matrix** dest, profile_context* prof) {
  if (!src || !dest) return STATUS_NULL_POINTER;

  if (prof) profile_start_init(prof);

  status_t status = csr_create(src->rows, src->cols, src->nnz, dest, NULL);
  if (status != STATUS_SUCCESS) return status;

  memcpy((*dest)->values, src->values, src->nnz * sizeof(double));
  memcpy((*dest)->col_idx, src->col_idx, src->nnz * sizeof(uint64_t));
  memcpy((*dest)->row_ptr, src->row_ptr, (src->rows + 1) * sizeof(uint64_t));

  if (prof) {
    size_t total_size;
    calculate_memory_size(src->rows, src->cols, src->nnz, &total_size);
    profile_record_memory(prof, total_size);
    profile_end_init(prof);
  }

  return STATUS_SUCCESS;
}

status_t csr_get_value(const csr_matrix* matrix, uint64_t row, uint64_t col, double* value, profile_context* prof) {
  if (!matrix || !value) return STATUS_NULL_POINTER;
  if (row >= matrix->rows || col >= matrix->cols) return STATUS_INVALID_DIMENSIONS;

  uint64_t row_start = matrix->row_ptr[row];
  uint64_t row_end = matrix->row_ptr[row + 1];
  uint64_t left = row_start;
  uint64_t right = row_end;

  while (left < right) {
    uint64_t mid = left + (right - left) / 2;
    if (matrix->col_idx[mid] == col) {
      *value = matrix->values[mid];
      return STATUS_SUCCESS;
    }
    if (matrix->col_idx[mid] < col) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  *value = 0.0;
  return STATUS_SUCCESS;
}

status_t csr_set_value(csr_matrix* matrix, uint64_t row, uint64_t col, double value, profile_context* prof) {
  if (!matrix) return STATUS_NULL_POINTER;
  if (row >= matrix->rows || col >= matrix->cols) return STATUS_INVALID_DIMENSIONS;

  uint64_t row_start = matrix->row_ptr[row];
  uint64_t row_end = matrix->row_ptr[row + 1];
  uint64_t left = row_start;
  uint64_t right = row_end;
  
  while (left < right) {
    uint64_t mid = left + (right - left) / 2;
    if (matrix->col_idx[mid] == col) {
      matrix->values[mid] = value;
      return STATUS_SUCCESS;
    }
    if (matrix->col_idx[mid] < col) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  if (value != 0.0) return STATUS_INVALID_SPARSE_STRUCTURE;

  return STATUS_SUCCESS;
}

status_t csr_validate(const csr_matrix* matrix) {
  if (!matrix || !matrix->values || !matrix->col_idx || !matrix->row_ptr) return STATUS_NULL_POINTER;
  if (matrix->rows == 0 && matrix->cols == 0 && matrix->nnz == 0) return STATUS_SUCCESS;
  if (matrix->rows == 0 || matrix->cols == 0 || matrix->nnz == 0) return STATUS_INVALID_DIMENSIONS;

  if (matrix->row_ptr[0] != 0) return STATUS_INVALID_SPARSE_STRUCTURE;
  if (matrix->row_ptr[matrix->rows] != matrix->nnz) return STATUS_INVALID_SPARSE_STRUCTURE;
    
  for (uint64_t i = 0; i < matrix->rows; i++) {
    if (matrix->row_ptr[i] > matrix->row_ptr[i + 1]) return STATUS_INVALID_SPARSE_STRUCTURE;
    if (matrix->row_ptr[i] > matrix->nnz) return STATUS_INVALID_SPARSE_STRUCTURE;
  }
  for (uint64_t i = 0; i < matrix->rows; i++) {
    uint64_t row_start = matrix->row_ptr[i];
    uint64_t row_end = matrix->row_ptr[i + 1];
    for (uint64_t j = row_start; j < row_end; j++) {
      if (matrix->col_idx[j] >= matrix->cols) return STATUS_INVALID_DIMENSIONS;
      if (j > row_start && matrix->col_idx[j] <= matrix->col_idx[j-1]) return STATUS_UNSORTED_INDICES;
    }
  }

  return STATUS_SUCCESS;
}

static inline int size_multiply_overflow(size_t a, size_t b, size_t* result) {
  if (a > 0 && b > SIZE_MAX / a) return 1;
  *result = a * b;
  return 0;
}

static status_t calculate_memory_size(uint64_t rows, uint64_t cols, uint64_t nnz, size_t* total_size) {
  size_t values_size, col_idx_size, row_ptr_size;

  if (size_multiply_overflow(nnz, sizeof(double), &values_size) ||
    size_multiply_overflow(nnz, sizeof(uint64_t), &col_idx_size) ||
    size_multiply_overflow(rows + 1, sizeof(uint64_t), &row_ptr_size)) {
    return STATUS_SIZE_OVERFLOW;
  }

  *total_size = values_size + col_idx_size + row_ptr_size + sizeof(csr_matrix);
  return STATUS_SUCCESS;
}
