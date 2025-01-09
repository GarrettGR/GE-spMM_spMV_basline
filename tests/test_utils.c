#include "test_utils.h"
#include "utils/status.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define TOLERANCE 1e-10 // tolerance for floating point comparisons

status_t create_test_matrix(uint64_t rows, uint64_t cols, uint64_t nnz, matrix_pattern_t pattern, csr_matrix **matrix) {
  if (!matrix) return STATUS_NULL_POINTER;

  status_t status = csr_create(rows, cols, nnz, matrix, NULL);
  if (status != STATUS_SUCCESS) return status;

  switch (pattern) {
    case PATTERN_DIAGONAL: {
      uint64_t num_entries = (rows < cols) ? rows : cols;
      num_entries = (num_entries < nnz) ? num_entries : nnz;
      
      for (uint64_t i = 0; i < rows; i++) {
        (*matrix)->row_ptr[i] = (i < num_entries) ? i : num_entries;
      }
      (*matrix)->row_ptr[rows] = num_entries;

      for (uint64_t i = 0; i < num_entries; i++) {
        (*matrix)->col_idx[i] = i;
        (*matrix)->values[i] = 1.0;
      }
      break;
    }

    case PATTERN_RANDOM: {
      uint64_t base_per_row = nnz / rows;
      uint64_t extra = nnz % rows;
      uint64_t current_pos = 0;
      
      for (uint64_t i = 0; i < rows; i++) {
        (*matrix)->row_ptr[i] = current_pos;
        uint64_t entries = base_per_row + (i < extra ? 1 : 0);
        uint64_t* used = (uint64_t*)calloc(cols, sizeof(uint64_t));
        
        if (!used) {
          csr_free(*matrix, NULL);
          return STATUS_ALLOCATION_FAILED;
        }
        
        for (uint64_t j = 0; j < entries; j++) {
          uint64_t col;
          do {
            col = (uint64_t)(((double)rand() / RAND_MAX) * cols);
          } while (used[col]);
          used[col] = 1;
          
          (*matrix)->col_idx[current_pos + j] = col;
          (*matrix)->values[current_pos + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }

        for (uint64_t j = 0; j < entries - 1; j++) {
          for (uint64_t k = 0; k < entries - j - 1; k++) {
            if ((*matrix)->col_idx[current_pos + k] > (*matrix)->col_idx[current_pos + k + 1]) {
              uint64_t temp_col = (*matrix)->col_idx[current_pos + k];
              (*matrix)->col_idx[current_pos + k] = (*matrix)->col_idx[current_pos + k + 1];
              (*matrix)->col_idx[current_pos + k + 1] = temp_col;
              
              double temp_val = (*matrix)->values[current_pos + k];
              (*matrix)->values[current_pos + k] = (*matrix)->values[current_pos + k + 1];
              (*matrix)->values[current_pos + k + 1] = temp_val;
            }
          }
        }
        free(used);
        current_pos += entries;
      }
      (*matrix)->row_ptr[rows] = nnz;
      break;
    }

    case PATTERN_DENSE:
    case PATTERN_BANDED:
      return STATUS_NOT_IMPLEMENTED;
    case PATTERN_BLOCK: {
      uint64_t block_size = (rows < cols) ? rows : cols;
      block_size = block_size / 3;
      uint64_t num_blocks = rows / block_size;
      uint64_t entries_per_row = nnz / rows;
      
      uint64_t current_pos = 0;
      for (uint64_t i = 0; i < rows; i++) {
        (*matrix)->row_ptr[i] = current_pos;
        uint64_t block = i / block_size;
        
        if (block < num_blocks) {
          uint64_t block_start = block * block_size;
          uint64_t block_end = (block + 1) * block_size;
          if (block_end > cols) block_end = cols;

          uint64_t max_entries = block_end - block_start;
          uint64_t entries = (entries_per_row < max_entries) ? entries_per_row : max_entries;
          
          for (uint64_t j = 0; j < entries; j++) {
            uint64_t col = block_start + j;
            if (col >= block_end) break;
            
            (*matrix)->col_idx[current_pos] = col;
            (*matrix)->values[current_pos] = 1.0;
            current_pos++;
          }
        }
      }
      (*matrix)->row_ptr[rows] = current_pos;
      (*matrix)->nnz = current_pos;
      break;
    }
  }

  return STATUS_SUCCESS;
}

status_t create_dense_matrix(uint64_t rows, uint64_t cols, double density, csr_matrix **matrix) {
  if (!matrix) return STATUS_NULL_POINTER;
  if (density <= 0.0 || density > 1.0) return STATUS_INVALID_INPUT;

  uint64_t min_nnz = rows;
  uint64_t max_nnz = rows * cols;
  uint64_t nnz = (uint64_t)(rows * cols * density);
  nnz = nnz < min_nnz || nnz > max_nnz ? min_nnz : nnz;
  
  status_t status = csr_create(rows, cols, nnz, matrix, NULL);
  if (status != STATUS_SUCCESS) return status;

  uint64_t base_per_row = nnz / rows;
  uint64_t extra = nnz % rows;
  uint64_t current_pos = 0;

  for (uint64_t i = 0; i < rows; i++) {
    (*matrix)->row_ptr[i] = current_pos;
    uint64_t entries = base_per_row + (i < extra ? 1 : 0);

    uint64_t* used = (uint64_t*)calloc(cols, sizeof(uint64_t));
    if (!used) {
      csr_free(*matrix, NULL);
      return STATUS_ALLOCATION_FAILED;
    }

    for (uint64_t j = 0; j < entries; j++) {
      uint64_t col;
      do {
        col = (uint64_t)(((double)rand() / RAND_MAX) * cols);
      } while (used[col]);
      used[col] = 1;
      
      (*matrix)->col_idx[current_pos + j] = col;
      (*matrix)->values[current_pos + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    for (uint64_t j = 0; j < entries - 1; j++) {
      for (uint64_t k = 0; k < entries - j - 1; k++) {
        if ((*matrix)->col_idx[current_pos + k] > (*matrix)->col_idx[current_pos + k + 1]) {
          uint64_t temp_col = (*matrix)->col_idx[current_pos + k];
          (*matrix)->col_idx[current_pos + k] = (*matrix)->col_idx[current_pos + k + 1];
          (*matrix)->col_idx[current_pos + k + 1] = temp_col;

          double temp_val = (*matrix)->values[current_pos + k];
          (*matrix)->values[current_pos + k] = (*matrix)->values[current_pos + k + 1];
          (*matrix)->values[current_pos + k + 1] = temp_val;
        }
      }
    }

    free(used);
    current_pos += entries;
  }
  (*matrix)->row_ptr[rows] = nnz;

  return STATUS_SUCCESS;
}

bool compare_matrices(const csr_matrix *A, const csr_matrix *B) {
  if (!A || !B) return false;
  if (A->rows != B->rows || A->cols != B->cols || A->nnz != B->nnz) return false;

  for (uint64_t i = 0; i <= A->rows; i++) {
    if (A->row_ptr[i] != B->row_ptr[i]) return false;
  }

  for (uint64_t i = 0; i < A->nnz; i++) {
    if (A->col_idx[i] != B->col_idx[i]) return false;
    if (fabs(A->values[i] - B->values[i]) > TOLERANCE) return false;
  }

  return true;
}

bool compare_vectors(const double *x, const double *y, uint64_t size) {
  if (!x || !y) return false;

  for (uint64_t i = 0; i < size; i++) {
    if (fabs(x[i] - y[i]) > TOLERANCE) return false;
  }

  return true;
}

status_t compute_dense_spmm(const csr_matrix *A, const csr_matrix *B, csr_matrix **C) {
  if (!A || !B || !C) return STATUS_NULL_POINTER;
  if (A->cols != B->rows) return STATUS_INVALID_DIMENSIONS;

  double *dense_C = (double *) calloc(A->rows * B->cols, sizeof(double));
  if (!dense_C) return STATUS_ALLOCATION_FAILED;

  for (uint64_t i = 0; i < A->rows; i++) {
    for (uint64_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
      uint64_t k = A->col_idx[j];
      double val_a = A->values[j];
      
      for (uint64_t l = B->row_ptr[k]; l < B->row_ptr[k + 1]; l++) {
        uint64_t col = B->col_idx[l];
        dense_C[i * B->cols + col] += val_a * B->values[l];
      }
    }
  }

  uint64_t nnz = 0;
  for (uint64_t i = 0; i < A->rows * B->cols; i++) {
    if (fabs(dense_C[i]) > 1e-15) nnz++;
  }

  status_t status = csr_create(A->rows, B->cols, nnz, C, NULL);
  if (status != STATUS_SUCCESS) {
    free(dense_C);
    return status;
  }

  (*C)->row_ptr[0] = 0;
  uint64_t pos = 0;
  for (uint64_t i = 0; i < A->rows; i++) {
    for (uint64_t j = 0; j < B->cols; j++) {
      double val = dense_C[i * B->cols + j];
      if (fabs(val) > 1e-15) {
        (*C)->col_idx[pos] = j;
        (*C)->values[pos] = val;
        pos++;
      }
    }
    (*C)->row_ptr[i + 1] = pos;
  }

  free(dense_C);
  return STATUS_SUCCESS;
}

status_t compute_dense_spmv(const csr_matrix *A, const double *x, double *y) {
  if (!A || !x || !y) return STATUS_NULL_POINTER;
  // if (x_size != A->cols || y_size != A->rows) return STATUS_INVALID_DIMENSIONS;

  memset(y, 0, A->rows * sizeof(double));

  for (uint64_t i = 0; i < A->rows; i++) {
    double sum = 0.0;
    for (uint64_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
      sum += A->values[j] * x[A->col_idx[j]];
    }
    y[i] = sum;
  }

  return STATUS_SUCCESS;
}

int run_test_suite(int argc, char** argv, const test_case_t* test_cases) {
  
  if (argc != 2) {
    printf("Usage: %s <test_name>\n", argv[0]); 
    printf("Available tests:\n");
    printf("  all (runs all tests)\n");
    for (const test_case_t* test = test_cases; test->name != NULL; test++) {
      printf("  %s\n", test->name);
    }
    return 1;
  }
  
  const char* test_name = argv[1];
  
  if (strcmp(test_name, "all") == 0) {
    srand(12345);
    for (const test_case_t* test = test_cases; test->name != NULL; test++) {
      test->func();
    }
    printf("All tests completed successfully\n");
    return 0;
  }

  for (const test_case_t* test = test_cases; test->name != NULL; test++) {
    if (strcmp(test_name, test->name) == 0) {
      srand(12345);
      test->func();
      printf("Test '%s' completed successfully\n", test_name);
      return 0;
    }
  }
  
  printf("Unknown test: %s\n", test_name);
  return 1;
}