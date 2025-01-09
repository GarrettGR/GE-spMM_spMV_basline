#include "algorithms/spMM/csr_spmm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --------------------------------------------------------------------------------------------------------------------
// Helper data structures and functions:

typedef struct {
  double *values;
  uint32_t *markers;
  uint32_t *cols;
  uint32_t col_count;
  uint32_t size;
} spa_t;

static void quicksort(uint32_t *arr, int left, int right);
static status_t spa_init(spa_t *spa, uint32_t size);
static void spa_add(spa_t *spa, uint32_t col, double val);
static void spa_sort_and_compact(spa_t *spa, csr_matrix *C, uint64_t *pos);

static status_t spa_init(spa_t *spa, uint32_t size) { // use uint64_t instead of uint32_t ???
  if (size > UINT32_MAX) return STATUS_INVALID_INPUT;
  spa->size = size;
  spa->col_count = 0;

  spa->values = (double *)malloc(size * sizeof(double));
  spa->markers = (uint32_t *)malloc(size * sizeof(uint32_t));
  spa->cols = (uint32_t *)malloc(size * sizeof(uint32_t));

  if (!spa->values || !spa->markers || !spa->cols) {
    free(spa->values);
    free(spa->markers);
    free(spa->cols);
    return STATUS_ALLOCATION_FAILED;
  }

  memset(spa->markers, 0, size * sizeof(uint32_t));
  return STATUS_SUCCESS;
}

static void spa_add(spa_t *spa, uint32_t col, double val) {
  if (fabs(val) < 1e-15) return;
  if (col >= spa->size) return;
  
  if (spa->markers[col] == 0) {
    spa->markers[col] = 1;
    spa->values[col] = val;
    spa->cols[spa->col_count++] = col;
  } else {
    double new_val = spa->values[col] + val;
    if (fabs(new_val) > 1e-15) {
      spa->values[col] = new_val;
    } else {
      spa->markers[col] = 0;
      for (uint32_t i = 0; i < spa->col_count; i++) {
        if (spa->cols[i] == col) {
          spa->cols[i] = spa->cols[--spa->col_count];
          break;
        }
      }
    }
  }
}

static void spa_sort_and_compact(spa_t *spa, csr_matrix *C, uint64_t *pos) {
  quicksort(spa->cols, 0, spa->col_count - 1);
  
  for (uint32_t i = 0; i < spa->col_count; i++) {
    uint32_t col = spa->cols[i];
    double val = spa->values[col];
    if (fabs(val) > 1e-15) {
      C->col_idx[*pos] = col;
      C->values[*pos] = val;
      (*pos)++;
    }
    spa->markers[col] = 0;
  }
  spa->col_count = 0;
}

static void quicksort(uint32_t *arr, int left, int right) {
  if (left >= right) return;

  int i = left, j = right;
  uint32_t pivot = arr[(left + right) / 2];

  while (i <= j) {
    while (arr[i] < pivot) i++;
    while (arr[j] > pivot) j--;

    if (i <= j) {
      uint32_t temp = arr[i];
      arr[i] = arr[j];
      arr[j] = temp;
      i++;
      j--;
    }
  }

  if (left < j) quicksort(arr, left, j);
  if (i < right) quicksort(arr, i, right);
}

// --------------------------------------------------------------------------------------------------------------------
// Main functions:

status_t csr_spmm_validate(const csr_matrix *A, const csr_matrix *B) {
  if (!A || !B) return STATUS_NULL_POINTER;

  status_t status = csr_validate(A);
  if (status != STATUS_SUCCESS) return status;
  status = csr_validate(B);
  if (status != STATUS_SUCCESS) return status;

  if (A->cols != B->rows) return STATUS_INVALID_DIMENSIONS;

  return STATUS_SUCCESS;
}

status_t csr_spmm_sequential(const csr_matrix *A, const csr_matrix *B, csr_matrix **C, profile_context *prof) {
  status_t status = csr_spmm_validate(A, B);
  if (status != STATUS_SUCCESS) return status;

  spa_t spa;
  status = spa_init(&spa, B->cols);
  if (status != STATUS_SUCCESS) return status;

  uint64_t *row_nnz = (uint64_t *)calloc(A->rows, sizeof(uint64_t));
  if (!row_nnz) {
    free(spa.values);
    free(spa.markers);
    free(spa.cols);
    return STATUS_ALLOCATION_FAILED;
  }

  uint64_t total_nnz = 0;
  for (uint64_t i = 0; i < A->rows; i++) {
    spa.col_count = 0;
    memset(spa.values, 0, B->cols * sizeof(double));
    memset(spa.markers, 0, B->cols * sizeof(uint32_t));

    for (uint64_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
      uint64_t k = A->col_idx[j];
      double val_a = A->values[j];
      for (uint64_t l = B->row_ptr[k]; l < B->row_ptr[k + 1]; l++) {
        spa_add(&spa, B->col_idx[l], val_a * B->values[l]);
      }
    }

    row_nnz[i] = spa.col_count;
    total_nnz += spa.col_count;

    for (uint32_t j = 0; j < spa.col_count; j++) {
      spa.markers[spa.cols[j]] = 0;
    }
  }

  status = csr_create(A->rows, B->cols, total_nnz, C, prof);
  if (status != STATUS_SUCCESS) {
    free(spa.values);
    free(spa.markers);
    free(spa.cols);
    free(row_nnz);
    return status;
  }

  (*C)->row_ptr[0] = 0;
  for (uint64_t i = 0; i < A->rows; i++) {
    (*C)->row_ptr[i + 1] = (*C)->row_ptr[i] + row_nnz[i];
  }

  uint64_t pos = 0;
  for (uint64_t i = 0; i < A->rows; i++) {
    spa.col_count = 0;
    memset(spa.values, 0, B->cols * sizeof(double));
    memset(spa.markers, 0, B->cols * sizeof(uint32_t));

    for (uint64_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
      uint64_t k = A->col_idx[j];
      double val_a = A->values[j];
      for (uint64_t l = B->row_ptr[k]; l < B->row_ptr[k + 1]; l++) {
        spa_add(&spa, B->col_idx[l], val_a * B->values[l]);
      }
    }

    spa_sort_and_compact(&spa, *C, &pos);
  }

  free(spa.values);
  free(spa.markers);
  free(spa.cols);
  free(row_nnz);
  return STATUS_SUCCESS;
}