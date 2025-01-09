#include "algorithms/spMV/csr_spmv.h"
#include "utils/profiler.h"
#include <string.h>

status_t csr_spmv_validate(const csr_matrix* matrix, const double* x, const double* y, uint64_t x_size, uint64_t y_size) {
  if (!matrix || !x || !y) return STATUS_NULL_POINTER;
    
  status_t status = csr_validate(matrix);
  if (status != STATUS_SUCCESS) return status;
    
  if (x_size != matrix->cols) return STATUS_INVALID_DIMENSIONS;
  if (y_size != matrix->rows) return STATUS_INVALID_DIMENSIONS;
    
  return STATUS_SUCCESS;
}

status_t csr_spmv_sequential(const csr_matrix* matrix, const double* x, double* y, profile_context* prof) {
  if (!matrix || !x || !y) return STATUS_NULL_POINTER;
  
  status_t status = csr_validate(matrix);
  if (status != STATUS_SUCCESS) return status;

  // if (prof) profile_start_compute(prof);

  memset(y, 0, matrix->rows * sizeof(double));

  for (uint64_t i = 0; i < matrix->rows; i++) {
    double sum = 0.0;
    for (uint64_t j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; j++) {
      sum += matrix->values[j] * x[matrix->col_idx[j]];
    }
    y[i] = sum;
  }

  if (prof) {
    // profile_record_flops(prof, matrix->nnz * 2); // NOTE: one multiply and one add per non-zero --- is this right for number of floating point operations ???
    // profile_end_compute(prof);
  }

  return STATUS_SUCCESS;
}
