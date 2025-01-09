#include "formats/csr.h"
#include "algorithms/spMM/csr_spmm.h"
#include "utils/status.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define MIN_PARALLEL_ROWS 1000
#define CHUNK_SIZE 32

status_t csr_spmm_openmp(const csr_matrix *A, const csr_matrix *B, csr_matrix **C, profile_context *prof) {
  if (!A || !B || !C) return STATUS_NULL_POINTER;
  if (A->cols != B->rows) return STATUS_INCOMPATIBLE_DIMENSIONS;

  uint64_t m = A->rows;
  // uint64_t k = A->cols;
  uint64_t n = B->cols;

  uint64_t *row_nnz = (uint64_t *)calloc(m + 1, sizeof(uint64_t));
  if (!row_nnz) return STATUS_ALLOCATION_FAILED;

#pragma omp parallel if (m >= MIN_PARALLEL_ROWS)
  {
    uint64_t *local_workspace = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (local_workspace) {
#pragma omp for schedule(dynamic, CHUNK_SIZE)
      for (uint64_t i = 0; i < m; i++) {
        memset(local_workspace, 0, n * sizeof(uint64_t));

        for (uint64_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
          uint64_t col = A->col_idx[j];
          for (uint64_t k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
            local_workspace[B->col_idx[k]] = 1;
          }
        }

        uint64_t count = 0;
        for (uint64_t j = 0; j < n; j++) {
          count += local_workspace[j];
        }
        row_nnz[i] = count;
      }
      free(local_workspace);
    }
  }

  uint64_t total_nnz = 0;
  for (uint64_t i = 0; i < m; i++) {
    uint64_t current = row_nnz[i];
    row_nnz[i] = total_nnz;
    total_nnz += current;
  }
  row_nnz[m] = total_nnz;

  status_t status = csr_create(m, n, total_nnz, C, NULL);
  if (status != STATUS_SUCCESS) {
    free(row_nnz);
    return status;
  }

  memcpy((*C)->row_ptr, row_nnz, (m + 1) * sizeof(uint64_t));
  free(row_nnz);

#pragma omp parallel if (m >= MIN_PARALLEL_ROWS)
  {
    double *value_accumulator = (double *)calloc(n, sizeof(double));
    uint64_t *col_marker = (uint64_t *)calloc(n, sizeof(uint64_t));

    if (value_accumulator && col_marker) {
#pragma omp for schedule(dynamic, CHUNK_SIZE)
      for (uint64_t i = 0; i < m; i++) {
        memset(value_accumulator, 0, n * sizeof(double));
        memset(col_marker, 0, n * sizeof(uint64_t));

        uint64_t nnz_idx = (*C)->row_ptr[i];

        for (uint64_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
          double a_val = A->values[j];
          uint64_t col = A->col_idx[j];

          for (uint64_t k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
            uint64_t b_col = B->col_idx[k];
            value_accumulator[b_col] += a_val * B->values[k];
            col_marker[b_col] = 1;
          }
        }

        for (uint64_t j = 0; j < n; j++) {
          if (col_marker[j]) {
            (*C)->col_idx[nnz_idx] = j;
            (*C)->values[nnz_idx] = value_accumulator[j];
            nnz_idx++;
          }
        }
      }
    }

    free(value_accumulator);
    free(col_marker);
  }

  return STATUS_SUCCESS;
}
