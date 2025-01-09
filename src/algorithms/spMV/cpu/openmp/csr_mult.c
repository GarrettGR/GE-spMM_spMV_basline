#include "algorithms/spMV/csr_spmv.h"
#include "utils/status.h"
#include "utils/profiler.h"
#include "formats/csr.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define MIN_PARALLEL_ROWS 1000
#define CHUNK_SIZE 32

status_t csr_spmv_openmp(const csr_matrix *A, const double *x, double *y, profile_context *prof) {
    if (!A || !x || !y) return STATUS_NULL_POINTER;
    
    status_t status = csr_validate(A);
    if (status != STATUS_SUCCESS) return status;

    const uint64_t m = A->rows;
    memset(y, 0, m * sizeof(double));

    #pragma omp parallel if(m >= MIN_PARALLEL_ROWS)
    {
      double* local_y = (double*) calloc(m, sizeof(double));
      if (local_y) {
        #pragma omp for schedule(dynamic, CHUNK_SIZE)
        for (uint64_t i = 0; i < m; i++) {
          double sum = 0.0;
          #pragma omp simd reduction(+:sum)
          for (uint64_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            sum += A->values[j] * x[A->col_idx[j]];
          }
          local_y[i] = sum;
        }

        #pragma omp critical
        {
          for (uint64_t i = 0; i < m; i++) {
            y[i] += local_y[i];
          }
        }
        free(local_y);
      }
    }

    return STATUS_SUCCESS;
}
