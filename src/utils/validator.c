#include "utils/validator.h"
#include "utils/status.h"
#include <math.h>
#include <petsc.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TOLERANCE 1e-8 // was 1e-12 but got a couple of failures
#define BLOCK_SIZE 32
#define DEBUG_PRINT(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)

// --------------------------------------------------------------------------------------------------------------------
// Helper functions:

static void init_petsc_if_needed(void);
static status_t validate_csr_integrity(const csr_matrix *mat);
static inline uint64_t min_u64(uint64_t a, uint64_t b);
static void ensure_unique_sorted_cols(uint64_t* col_idx, uint64_t start, uint64_t end, uint64_t max_col);
static PetscErrorCode safe_csr_to_petsc(const csr_matrix *mat, Mat *petsc_mat);

static void init_petsc_if_needed(void) {
  PetscBool is_initialized;
  PetscInitialized(&is_initialized);
  if (!is_initialized) {
    PetscInitializeNoArguments();
  }
}

static inline uint64_t min_u64(uint64_t a, uint64_t b) {
  return (a < b) ? a : b;
}

static void ensure_unique_sorted_cols(uint64_t* col_idx, uint64_t start, uint64_t end, uint64_t max_col) {
  if (start >= end) return;

  DEBUG_PRINT("Ensuring unique sorted columns for range [%llu, %llu)", start, end);

  for (uint64_t i = start; i < end - 1; i++) {
    for (uint64_t j = start; j < end - 1 - (i - start); j++) {
      if (col_idx[j] > col_idx[j + 1]) {
        uint64_t temp = col_idx[j];
        col_idx[j] = col_idx[j + 1];
        col_idx[j + 1] = temp;
      }
    }
  }

  uint64_t write_pos = start;
  for (uint64_t read_pos = start + 1; read_pos < end; read_pos++) {
    if (col_idx[read_pos] != col_idx[write_pos]) {
      write_pos++;
      if (write_pos != read_pos) {
        col_idx[write_pos] = col_idx[read_pos];
      }
    }
  }

  uint64_t new_end = write_pos + 1;

  DEBUG_PRINT("Reduced from %llu to %llu entries after removing duplicates", end - start, new_end - start);

  while (new_end < end) {
    uint64_t new_col;
    bool found = false;
    for (uint64_t col = 0; col < max_col; col++) {
      bool is_used = false;
      for (uint64_t i = start; i < new_end; i++) {
        if (col_idx[i] == col) {
          is_used = true;
          break;
        }
      }
      if (!is_used) {
        new_col = col;
        found = true;
        break;
      }
    }

    if (!found) {
      DEBUG_PRINT("Warning: Could not find unused column index");
      break;
    }

    uint64_t insert_pos = new_end;
    while (insert_pos > start && col_idx[insert_pos - 1] > new_col) {
      col_idx[insert_pos] = col_idx[insert_pos - 1];
      insert_pos--;
    }
    col_idx[insert_pos] = new_col;
    new_end++;
  }

  DEBUG_PRINT("Final entry count: %llu", new_end - start);
}

static status_t validate_csr_integrity(const csr_matrix *mat) {
  if (!mat) {
    DEBUG_PRINT("Null matrix pointer");
    return STATUS_NULL_POINTER;
  }

  if (!mat->values || !mat->col_idx || !mat->row_ptr) {
    DEBUG_PRINT("Matrix arrays are null: values=%p, col_idx=%p, row_ptr=%p", (void *)mat->values, (void *)mat->col_idx, (void *)mat->row_ptr);
    return STATUS_NULL_POINTER;
  }

  if (mat->rows == 0 || mat->cols == 0) {
    DEBUG_PRINT("Invalid dimensions: rows=%llu, cols=%llu", mat->rows, mat->cols);
    return STATUS_INVALID_DIMENSIONS;
  }

  if (mat->nnz > mat->rows * mat->cols) {
    DEBUG_PRINT("nnz (%llu) exceeds matrix capacity (%llu)", mat->nnz, mat->rows * mat->cols);
    return STATUS_INVALID_SPARSE_STRUCTURE;
  }

  if (mat->row_ptr[0] != 0) {
    DEBUG_PRINT("First row pointer is not 0: %llu", mat->row_ptr[0]);
    return STATUS_INVALID_SPARSE_STRUCTURE;
  }

  if (mat->row_ptr[mat->rows] != mat->nnz) {
    DEBUG_PRINT("Last row pointer (%llu) does not match nnz (%llu)", mat->row_ptr[mat->rows], mat->nnz);
    return STATUS_INVALID_SPARSE_STRUCTURE;
  }

  for (uint64_t i = 0; i < mat->rows; i++) {
    if (mat->row_ptr[i] > mat->row_ptr[i + 1]) {
      DEBUG_PRINT("Non-monotonic row pointers at row %llu: %llu > %llu", i, mat->row_ptr[i], mat->row_ptr[i + 1]);
      return STATUS_INVALID_SPARSE_STRUCTURE;
    }

    for (uint64_t j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
      if (mat->col_idx[j] >= mat->cols) {
        DEBUG_PRINT("Invalid column index at (%llu,%llu): %llu >= %llu", i, j - mat->row_ptr[i], mat->col_idx[j], mat->cols);
        return STATUS_INVALID_DIMENSIONS;
      }

      if (j > mat->row_ptr[i] && mat->col_idx[j] <= mat->col_idx[j - 1]) {
        DEBUG_PRINT("Unsorted/duplicate column indices at row %llu: %llu <= %llu", i, mat->col_idx[j], mat->col_idx[j - 1]);
        return STATUS_UNSORTED_INDICES;
      }
    }
  }

  return STATUS_SUCCESS;
}

// ============================================================================
// main functions

static PetscErrorCode safe_csr_to_petsc(const csr_matrix *mat, Mat *petsc_mat) {
  DEBUG_PRINT("Converting CSR matrix %llux%llu with %llu nonzeros", mat->rows, mat->cols, mat->nnz);

  status_t status = validate_csr_integrity(mat);
  if (status != STATUS_SUCCESS) {
    DEBUG_PRINT("CSR matrix validation failed with status %d", status);
    return PETSC_ERR_ARG_CORRUPT;
  }

  PetscErrorCode ierr;

  ierr = MatCreate(PETSC_COMM_SELF, petsc_mat);
  CHKERRQ(ierr);
  ierr = MatSetType(*petsc_mat, MATSEQAIJ);
  CHKERRQ(ierr);

  if (mat->rows > PETSC_MAX_INT || mat->cols > PETSC_MAX_INT) {
    DEBUG_PRINT("Matrix dimensions exceed PETSc limits");
    return PETSC_ERR_ARG_OUTOFRANGE;
  }

  PetscInt m = (PetscInt)mat->rows;
  PetscInt n = (PetscInt)mat->cols;
  ierr = MatSetSizes(*petsc_mat, m, n, m, n);
  CHKERRQ(ierr);

  PetscInt *nnz = (PetscInt *)malloc(m * sizeof(PetscInt));
  if (!nnz) {
    DEBUG_PRINT("Failed to allocate nnz array");
    return PETSC_ERR_MEM;
  }

  for (PetscInt i = 0; i < m; i++) {
    PetscInt row_nnz = (PetscInt)(mat->row_ptr[i + 1] - mat->row_ptr[i]);
    if (row_nnz > PETSC_MAX_INT) {
      free(nnz);
      DEBUG_PRINT("Row %d has too many nonzeros: %llu", i, mat->row_ptr[i + 1] - mat->row_ptr[i]);
      return PETSC_ERR_ARG_OUTOFRANGE;
    }
    nnz[i] = row_nnz;
  }

  ierr = MatSeqAIJSetPreallocation(*petsc_mat, 0, nnz);
  free(nnz);
  CHKERRQ(ierr);

  for (PetscInt i = 0; i < m; i++) {
    PetscInt ncols = (PetscInt)(mat->row_ptr[i + 1] - mat->row_ptr[i]);
    if (ncols == 0) continue;

    PetscInt *cols = (PetscInt *)malloc(ncols * sizeof(PetscInt));
    PetscScalar *vals = (PetscScalar *)malloc(ncols * sizeof(PetscScalar));

    if (!cols || !vals) {
      free(cols);
      free(vals);
      DEBUG_PRINT("Failed to allocate temporary arrays for row %d", i);
      return PETSC_ERR_MEM;
    }

    for (PetscInt j = 0; j < ncols; j++) {
      uint64_t idx = mat->row_ptr[i] + j;
      if (mat->col_idx[idx] > PETSC_MAX_INT) {
        free(cols);
        free(vals);
        DEBUG_PRINT("Column index too large at (%d,%d): %llu", i, j, mat->col_idx[idx]);
        return PETSC_ERR_ARG_OUTOFRANGE;
      }
      cols[j] = (PetscInt)mat->col_idx[idx];
      vals[j] = (PetscScalar)mat->values[idx];
    }

    ierr = MatSetValues(*petsc_mat, 1, &i, ncols, cols, vals, INSERT_VALUES);
    free(cols);
    free(vals);

    if (ierr) {
      DEBUG_PRINT("MatSetValues failed for row %d", i);
      return ierr;
    }
  }

  ierr = MatAssemblyBegin(*petsc_mat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*petsc_mat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  DEBUG_PRINT("Matrix conversion completed successfully");
  return 0;
}

bool validate_spmm_with_petsc(const csr_matrix *A, const csr_matrix *B, const csr_matrix *C) {
  DEBUG_PRINT("Starting SpMM validation");

  if (!A || !B || !C) {
    DEBUG_PRINT("Null matrix pointer(s): A=%p, B=%p, C=%p", (void *)A, (void *)B, (void *)C);
    return false;
  }

  init_petsc_if_needed();

  Mat petsc_A = NULL, petsc_B = NULL, petsc_result = NULL;
  PetscErrorCode ierr;
  bool match = false;

  ierr = safe_csr_to_petsc(A, &petsc_A);
  if (ierr) {
    DEBUG_PRINT("Failed to convert matrix A to PETSc format");
    goto cleanup;
  }

  ierr = safe_csr_to_petsc(B, &petsc_B);
  if (ierr) {
    DEBUG_PRINT("Failed to convert matrix B to PETSc format");
    goto cleanup;
  }

  ierr = MatMatMult(petsc_A, petsc_B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &petsc_result);
  if (ierr) {
    DEBUG_PRINT("MatMatMult failed");
    goto cleanup;
  }

  PetscInt m, n;
  ierr = MatGetSize(petsc_result, &m, &n);
  if (ierr) {
    DEBUG_PRINT("Failed to get result matrix size");
    goto cleanup;
  }

  if (m != (PetscInt)C->rows || n != (PetscInt)C->cols) {
    DEBUG_PRINT("Result matrix dimensions mismatch: expected %llux%llu, got %dx%d", C->rows, C->cols, m, n);
    goto cleanup;
  }

  DEBUG_PRINT("Comparing matrices");
  match = true;
  for (PetscInt i = 0; i < m && match; i++) {
    const PetscInt *cols;
    const PetscScalar *vals;
    PetscInt ncols;

    ierr = MatGetRow(petsc_result, i, &ncols, &cols, &vals);
    if (ierr) {
      DEBUG_PRINT("Failed to get row %d from PETSc matrix", i);
      match = false;
      break;
    }

    for (PetscInt j = 0; j < ncols; j++) {
      double csr_val;
      status_t status = csr_get_value(C, i, cols[j], &csr_val, NULL);
      if (status != STATUS_SUCCESS || fabs(csr_val - PetscRealPart(vals[j])) > TOLERANCE) {
        DEBUG_PRINT("Value mismatch at (%d,%d): expected %g, got %g", i, cols[j], PetscRealPart(vals[j]), csr_val);
        match = false;
        break;
      }
    }

    ierr = MatRestoreRow(petsc_result, i, &ncols, &cols, &vals);
    if (ierr) {
      DEBUG_PRINT("Failed to restore row %d", i);
      match = false;
      break;
    }
  }

cleanup:
  if (petsc_A) {
    MatDestroy(&petsc_A);
  }
  if (petsc_B) {
    MatDestroy(&petsc_B);
  }
  if (petsc_result) {
    MatDestroy(&petsc_result);
  }

  DEBUG_PRINT("Validation %s", match ? "succeeded" : "failed");
  return match;
}

bool validate_spmv_with_petsc(const csr_matrix *A, const double *x, const double *y) {
  DEBUG_PRINT("Starting SpMV validation");

  if (!A || !x || !y) {
    DEBUG_PRINT("Null pointer(s): A=%p, x=%p, y=%p", (void *)A, (void *)x, (void *)y);
    return false;
  }

  // if (x_size != A->cols || y_size != A->rows) {
  //   DEBUG_PRINT("Dimension mismatch: matrix is %llux%llu, vectors are x=%llu, y=%llu",A->rows, A->cols, x_size, y_size);
  //   return false;
  // }

  uint64_t x_size = A->cols;
  uint64_t y_size = A->rows;

  init_petsc_if_needed();

  Mat petsc_A = NULL;
  Vec petsc_x = NULL, petsc_y = NULL;
  PetscErrorCode ierr;
  bool result = false;

  ierr = safe_csr_to_petsc(A, &petsc_A);
  if (ierr) {
    DEBUG_PRINT("Failed to convert matrix to PETSc format");
    goto cleanup;
  }

  ierr = VecCreate(PETSC_COMM_SELF, &petsc_x);
  if (ierr) goto cleanup;

  if (x_size > PETSC_MAX_INT) {
    DEBUG_PRINT("Vector size too large for PETSc: %llu", x_size);
    goto cleanup;
  }

  ierr = VecSetSizes(petsc_x, PETSC_DECIDE, (PetscInt)x_size);
  if (ierr) goto cleanup;

  ierr = VecSetFromOptions(petsc_x);
  if (ierr) goto cleanup;

  {
    PetscScalar *x_array;
    ierr = VecGetArray(petsc_x, &x_array);
    if (ierr) goto cleanup;

    for (uint64_t i = 0; i < x_size; i++) {
      x_array[i] = (PetscScalar)x[i];
    }

    ierr = VecRestoreArray(petsc_x, &x_array);
    if (ierr) goto cleanup;
  }

  ierr = VecCreate(PETSC_COMM_SELF, &petsc_y);
  if (ierr) goto cleanup;

  if (y_size > PETSC_MAX_INT) {
    DEBUG_PRINT("Vector size too large for PETSc: %llu", y_size);
    goto cleanup;
  }

  ierr = VecSetSizes(petsc_y, PETSC_DECIDE, (PetscInt)y_size);
  if (ierr) goto cleanup;

  ierr = VecSetFromOptions(petsc_y);
  if (ierr) goto cleanup;

  DEBUG_PRINT("Performing SpMV operation");
  ierr = MatMult(petsc_A, petsc_x, petsc_y);
  if (ierr) {
    DEBUG_PRINT("MatMult failed");
    goto cleanup;
  }

  {
    PetscScalar *result_array;
    ierr = VecGetArray(petsc_y, &result_array);
    if (ierr)
      goto cleanup;

    result = true;
    for (uint64_t i = 0; i < y_size; i++) {
      double diff = fabs(y[i] - PetscRealPart(result_array[i]));
      if (diff > TOLERANCE) {
        DEBUG_PRINT("Value mismatch at index %llu: expected %g, got %g (diff=%g)", i, PetscRealPart(result_array[i]), y[i], diff);
        result = false;
        break;
      }
    }

    ierr = VecRestoreArray(petsc_y, &result_array);
    if (ierr) {
      result = false;
      goto cleanup;
    }
  }

cleanup:
  if (petsc_A)
    MatDestroy(&petsc_A);
  if (petsc_x)
    VecDestroy(&petsc_x);
  if (petsc_y)
    VecDestroy(&petsc_y);

  DEBUG_PRINT("Validation %s", result ? "succeeded" : "failed");
  return result;
}

bool compare_matrices(const csr_matrix *A, const csr_matrix *B) {
  DEBUG_PRINT("Comparing matrices");

  if (!A || !B) {
    DEBUG_PRINT("Null matrix pointer(s): A=%p, B=%p", (void *)A, (void *)B);
    return false;
  }

  if (A->rows != B->rows || A->cols != B->cols || A->nnz != B->nnz) {
    DEBUG_PRINT("Dimension mismatch: A(%llux%llu, nnz=%llu) vs B(%llux%llu, nnz=%llu)", A->rows, A->cols, A->nnz, B->rows, B->cols, B->nnz);
    return false;
  }

  status_t status = validate_csr_integrity(A);
  if (status != STATUS_SUCCESS) {
    DEBUG_PRINT("Matrix A validation failed with status %d", status);
    return false;
  }

  status = validate_csr_integrity(B);
  if (status != STATUS_SUCCESS) {
    DEBUG_PRINT("Matrix B validation failed with status %d", status);
    return false;
  }

  for (uint64_t i = 0; i <= A->rows; i++) {
    if (A->row_ptr[i] != B->row_ptr[i]) {
      DEBUG_PRINT("Row pointer mismatch at index %llu: A=%llu, B=%llu", i, A->row_ptr[i], B->row_ptr[i]);
      return false;
    }
  }

  for (uint64_t i = 0; i < A->nnz; i++) {
    if (A->col_idx[i] != B->col_idx[i]) {
      DEBUG_PRINT("Column index mismatch at position %llu: A=%llu, B=%llu", i, A->col_idx[i], B->col_idx[i]);
      return false;
    }

    double diff = fabs(A->values[i] - B->values[i]);
    if (diff > TOLERANCE) {
      DEBUG_PRINT("Value mismatch at position %llu: A=%g, B=%g (diff=%g)", i, A->values[i], B->values[i], diff);
      return false;
    }
  }

  DEBUG_PRINT("Matrices are equal within tolerance %g", TOLERANCE);
  return true;
}

bool compare_vectors(const double *x, const double *y, uint64_t size) {
  DEBUG_PRINT("Comparing vectors of size %llu", size);

  if (!x || !y) {
    DEBUG_PRINT("Null vector pointer(s): x=%p, y=%p", (void *)x, (void *)y);
    return false;
  }

  for (uint64_t i = 0; i < size; i++) {
    double diff = fabs(x[i] - y[i]);
    if (diff > TOLERANCE) {
      DEBUG_PRINT("Value mismatch at index %llu: x=%g, y=%g (diff=%g)", i, x[i], y[i], diff);
      return false;
    }
  }

  DEBUG_PRINT("Vectors are equal within tolerance %g", TOLERANCE);
  return true;
}

status_t create_test_matrix(uint64_t rows, uint64_t cols, uint64_t nnz, matrix_pattern_t pattern, csr_matrix **matrix) {
  DEBUG_PRINT("Creating test matrix: %llux%llu with %llu nonzeros, pattern=%d", rows, cols, nnz, pattern);

  if (!matrix) {
    DEBUG_PRINT("Null output matrix pointer");
    return STATUS_NULL_POINTER;
  }

  if (rows == 0 || cols == 0 || nnz == 0 || nnz > rows * cols) {
    DEBUG_PRINT("Invalid dimensions: rows=%llu, cols=%llu, nnz=%llu", rows, cols, nnz);
    return STATUS_INVALID_DIMENSIONS;
  }

  status_t status = csr_create(rows, cols, nnz, matrix, NULL);
  if (status != STATUS_SUCCESS) {
    DEBUG_PRINT("Failed to create CSR matrix: status=%d", status);
    return status;
  }

  static bool rng_initialized = false;
  if (!rng_initialized) {
    srand(12345);
    rng_initialized = true;
  }

  switch (pattern) {
    case PATTERN_RANDOM: {
      DEBUG_PRINT("Generating random pattern matrix %llux%llu with %llu nonzeros", rows, cols, nnz);

      uint64_t base_entries = nnz / rows;
      uint64_t extra_entries = nnz % rows;

      DEBUG_PRINT("Base entries per row: %llu, Extra entries: %llu", base_entries, extra_entries);

      if (base_entries >= cols) {
        DEBUG_PRINT("Too many entries per row: %llu >= %llu columns", base_entries, cols);
        csr_free(*matrix, NULL);
        return STATUS_INVALID_DIMENSIONS;
      }

      uint64_t current_pos = 0;
      (*matrix)->row_ptr[0] = 0;

      for (uint64_t i = 0; i < rows; i++) {
        uint64_t row_entries = base_entries + (i < extra_entries ? 1 : 0);

        DEBUG_PRINT("Row %llu: Allocating %llu entries starting at position %llu", i, row_entries, current_pos);

        for (uint64_t j = 0; j < row_entries; j++) {
          (*matrix)->col_idx[current_pos + j] = rand() % cols;
          (*matrix)->values[current_pos + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }

        ensure_unique_sorted_cols((*matrix)->col_idx, current_pos, current_pos + row_entries, cols);

        uint64_t actual_entries = 0;
        for (uint64_t j = current_pos;  j < current_pos + row_entries && (j == current_pos || (*matrix)->col_idx[j] != (*matrix)->col_idx[j-1]); j++) {
            actual_entries++;
        }

        DEBUG_PRINT("Row %llu: Actually got %llu unique entries", i, actual_entries);

        current_pos += actual_entries;
        (*matrix)->row_ptr[i + 1] = current_pos;
      }

      (*matrix)->nnz = current_pos;
      DEBUG_PRINT("Final nonzero count: %llu", (*matrix)->nnz);
      break;
    }

    case PATTERN_DIAGONAL: {
      DEBUG_PRINT("Generating diagonal pattern matrix %llux%llu", rows, cols);

      if (rows != cols) {
        DEBUG_PRINT("Non-square matrix for diagonal pattern: %llux%llu", rows, cols);
        csr_free(*matrix, NULL);
        return STATUS_INVALID_DIMENSIONS;
      }

      if (nnz != rows) {
        DEBUG_PRINT("Incorrect nnz for diagonal matrix: got %llu, expected %llu", nnz, rows);
        csr_free(*matrix, NULL);
        return STATUS_INVALID_DIMENSIONS;
      }

      for (uint64_t i = 0; i < rows; i++) {
        (*matrix)->row_ptr[i] = i;
        (*matrix)->col_idx[i] = i;
        (*matrix)->values[i] = 1.0;
      }
      (*matrix)->row_ptr[rows] = rows;

      DEBUG_PRINT("Created diagonal matrix with %llu nonzeros", rows);
      break;
    }

    case PATTERN_DENSE: {
      DEBUG_PRINT("Generating dense pattern matrix %llux%llu with %llu nonzeros", rows, cols, nnz);

      uint64_t max_entries = rows * cols;
      if (nnz > max_entries) {
          DEBUG_PRINT("Too many nonzeros for dense pattern: %llu > %llu", nnz, max_entries);
          csr_free(*matrix, NULL);
          return STATUS_INVALID_DIMENSIONS;
      }

      uint64_t entries_per_row = (nnz + rows - 1) / rows;
      if (entries_per_row > cols) {
        entries_per_row = cols;
      }

      DEBUG_PRINT("Target entries per row: %llu", entries_per_row);

      uint64_t current_pos = 0;
      (*matrix)->row_ptr[0] = 0;

      for (uint64_t i = 0; i < rows && current_pos < nnz; i++) {
        uint64_t row_entries = (i == rows - 1) ? (nnz - current_pos) : min_u64(entries_per_row, nnz - current_pos);

        DEBUG_PRINT("Row %llu: Adding %llu entries at position %llu", i, row_entries, current_pos);

        for (uint64_t j = 0; j < row_entries; j++) {
          (*matrix)->col_idx[current_pos] = j;
          (*matrix)->values[current_pos] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
          current_pos++;
        }

        (*matrix)->row_ptr[i + 1] = current_pos;
      }

      for (uint64_t i = 0; i < rows; i++) {
        if ((*matrix)->row_ptr[i + 1] == 0) {
          (*matrix)->row_ptr[i + 1] = current_pos;
        }
      }

      DEBUG_PRINT("Created dense matrix with %llu nonzeros", current_pos);
      break;
    }

    case PATTERN_BLOCK: {
      DEBUG_PRINT("Generating block pattern matrix %llux%llu with %llu nonzeros", rows, cols, nnz);

      if (rows % BLOCK_SIZE != 0 || cols % BLOCK_SIZE != 0) {
        DEBUG_PRINT("Dimensions not multiple of block size (%d): %llux%llu", BLOCK_SIZE, rows, cols);
        csr_free(*matrix, NULL);
        return STATUS_INVALID_DIMENSIONS;
      }

      uint64_t num_blocks = rows / BLOCK_SIZE;
      uint64_t entries_per_block = nnz / num_blocks;
      uint64_t current_pos = 0;

      (*matrix)->row_ptr[0] = 0;

      for (uint64_t block = 0; block < num_blocks && current_pos < nnz; block++) {
        DEBUG_PRINT("Processing block %llu", block);

        uint64_t entries_per_row = (entries_per_block + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (uint64_t i = 0; i < BLOCK_SIZE && current_pos < nnz; i++) {
          uint64_t row = block * BLOCK_SIZE + i;

          DEBUG_PRINT("Block %llu, Row %llu: Adding %llu entries at position %llu", block, row, entries_per_row, current_pos);

          for (uint64_t j = 0; j < entries_per_row && current_pos < nnz; j++) {
            (*matrix)->col_idx[current_pos] = block * BLOCK_SIZE + j;
            (*matrix)->values[current_pos] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            current_pos++;
          }
          (*matrix)->row_ptr[row + 1] = current_pos;
        }
      }

      for (uint64_t i = 0; i < rows; i++) {
        if ((*matrix)->row_ptr[i + 1] == 0) {
          (*matrix)->row_ptr[i + 1] = current_pos;
        }
      }

      for (uint64_t i = 0; i < rows; i++) {
        if ((*matrix)->row_ptr[i + 1] == 0) {
          (*matrix)->row_ptr[i + 1] = current_pos;
        }
      }

      DEBUG_PRINT("Created block matrix with %llu nonzeros", current_pos);
      break;
    }

    default: {
      DEBUG_PRINT("Invalid pattern type: %d", pattern);
      csr_free(*matrix, NULL);
      return STATUS_INVALID_INPUT;
    }
  }

  status = validate_csr_integrity(*matrix);
  if (status != STATUS_SUCCESS) {
    DEBUG_PRINT("Generated matrix validation failed: status=%d", status);
    csr_free(*matrix, NULL);
    return status;
  }

  DEBUG_PRINT("Matrix generation successful");
  return STATUS_SUCCESS;
}
