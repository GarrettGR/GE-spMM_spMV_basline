#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define BLOCK_SIZE 4 // 4x4 blocks (??)
#define BLOCK_AREA (BLOCK_SIZE * BLOCK_SIZE)

typedef struct {
  size_t block_rows;
  size_t block_cols;
  size_t rows;
  size_t cols;
  size_t nnz_blocks;
  size_t *block_row_ptr;
  size_t *block_col_idx;
  double *values;
} BCSRMatrix;

typedef struct {
  double load_time;
  double compute_time;
  double memory_usage;
  size_t flop_count;
} Performance;

BCSRMatrix *create_bcsr_matrix(size_t rows, size_t cols, size_t nnz_blocks) {
  BCSRMatrix *matrix = (BCSRMatrix *)malloc(sizeof(BCSRMatrix));
  if (!matrix)
    return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->block_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  matrix->block_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  matrix->nnz_blocks = nnz_blocks;

  matrix->block_row_ptr =
      (size_t *)aligned_alloc(64, (matrix->block_rows + 1) * sizeof(size_t));
  matrix->block_col_idx =
      (size_t *)aligned_alloc(64, nnz_blocks * sizeof(size_t));
  matrix->values =
      (double *)aligned_alloc(64, nnz_blocks * BLOCK_AREA * sizeof(double));

  if (!matrix->block_row_ptr || !matrix->block_col_idx || !matrix->values) {
    free(matrix->block_row_ptr);
    free(matrix->block_col_idx);
    free(matrix->values);
    free(matrix);
    return NULL;
  }
  return matrix;
}

static inline void multiply_blocks(const double *A_block, const double *B_block,
                                   double *C_block) {
  for (int i = 0; i < BLOCK_SIZE; i++) {
    for (int k = 0; k < BLOCK_SIZE; k++) {
      double a_ik = A_block[i * BLOCK_SIZE + k];
      for (int j = 0; j < BLOCK_SIZE; j++) {
        C_block[i * BLOCK_SIZE + j] += a_ik * B_block[k * BLOCK_SIZE + j];
      }
    }
  }
}

static inline void accumulate_blocks(double *dest_block,
                                     const double *src_block) {
  for (int i = 0; i < BLOCK_AREA; i++) {
    dest_block[i] += src_block[i];
  }
}

BCSRMatrix *load_matrix_from_mtx(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open file %s\n", filename);
    return NULL;
  }

  char line[1024];
  do {
    if (!fgets(line, sizeof(line), f)) {
      fclose(f);
      return NULL;
    }
  } while (line[0] == '%');

  size_t rows, cols, nnz;
  if (sscanf(line, "%zu %zu %zu", &rows, &cols, &nnz) != 3) {
    fclose(f);
    return NULL;
  }

  size_t block_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t block_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t max_blocks = nnz;

  BCSRMatrix *matrix = create_bcsr_matrix(rows, cols, max_blocks);
  if (!matrix) {
    fclose(f);
    return NULL;
  }

  double **blocks = (double **)calloc(block_rows * block_cols, sizeof(double *));
  size_t *block_nnz = (size_t *)calloc(block_rows * block_cols, sizeof(size_t));

  for (size_t i = 0; i < nnz; i++) {
    size_t row, col;
    double value;
    if (fscanf(f, "%zu %zu %lf", &row, &col, &value) != 3) {
      free(blocks);
      free(block_nnz);
      free_bcsr_matrix(matrix);
      fclose(f);
      return NULL;
    }
    row--;
    col--;

    size_t block_row = row / BLOCK_SIZE;
    size_t block_col = col / BLOCK_SIZE;
    size_t local_row = row % BLOCK_SIZE;
    size_t local_col = col % BLOCK_SIZE;
    size_t block_idx = block_row * block_cols + block_col;

    if (!blocks[block_idx]) {
      blocks[block_idx] = (double *)calloc(BLOCK_AREA, sizeof(double));
    }
    blocks[block_idx][local_row * BLOCK_SIZE + local_col] = value;
    block_nnz[block_idx]++;
  }
  fclose(f);

  size_t current_block = 0;
  matrix->block_row_ptr[0] = 0;

  for (size_t i = 0; i < block_rows; i++) {
    for (size_t j = 0; j < block_cols; j++) {
      size_t block_idx = i * block_cols + j;
      if (block_nnz[block_idx] > 0) {
        memcpy(&matrix->values[current_block * BLOCK_AREA], blocks[block_idx],
               BLOCK_AREA * sizeof(double));
        matrix->block_col_idx[current_block] = j;
        current_block++;
      }
    }
    matrix->block_row_ptr[i + 1] = current_block;
  }

  matrix->nnz_blocks = current_block;

  for (size_t i = 0; i < block_rows * block_cols; i++) {
    free(blocks[i]);
  }
  free(blocks);
  free(block_nnz);

  return matrix;
}

BCSRMatrix *multiply_bcsr_matrices(const BCSRMatrix *A, const BCSRMatrix *B) {
  if (A->cols != B->rows) {
    fprintf(stderr, "Matrix dimensions incompatible for multiplication\n");
    return NULL;
  }

  size_t *block_row_nnz = (size_t *)calloc(A->block_rows, sizeof(size_t));

#pragma omp parallel
  {
    double *temp_blocks =
        (double *)calloc(B->block_cols * BLOCK_AREA, sizeof(double));
    size_t *temp_marks = (size_t *)calloc(B->block_cols, sizeof(size_t));

#pragma omp for schedule(dynamic, 8)
    for (size_t i = 0; i < A->block_rows; i++) {
      size_t nnz = 0;

      for (size_t j = A->block_row_ptr[i]; j < A->block_row_ptr[i + 1]; j++) {
        size_t block_col = A->block_col_idx[j];

        for (size_t k = B->block_row_ptr[block_col];
             k < B->block_row_ptr[block_col + 1]; k++) {
          size_t b_block_col = B->block_col_idx[k];

          if (temp_marks[b_block_col] != i + 1) {
            temp_marks[b_block_col] = i + 1;
            nnz++;
          }
        }
      }
      block_row_nnz[i] = nnz;
    }
    free(temp_blocks);
    free(temp_marks);
  }

  size_t total_blocks = 0;
  for (size_t i = 0; i < A->block_rows; i++) {
    total_blocks += block_row_nnz[i];
  }

  BCSRMatrix *C = create_bcsr_matrix(A->rows, B->cols, total_blocks);
  if (!C) {
    free(block_row_nnz);
    return NULL;
  }

  C->block_row_ptr[0] = 0;
  for (size_t i = 0; i < A->block_rows; i++) {
    C->block_row_ptr[i + 1] = C->block_row_ptr[i] + block_row_nnz[i];
  }

#pragma omp parallel
  {
    double *temp_blocks = (double *)aligned_alloc(
        64, B->block_cols * BLOCK_AREA * sizeof(double));
    size_t *temp_marks = (size_t *)calloc(B->block_cols, sizeof(size_t));
    memset(temp_blocks, 0, B->block_cols * BLOCK_AREA * sizeof(double));

#pragma omp for schedule(dynamic, 8)
    for (size_t i = 0; i < A->block_rows; i++) {
      size_t next_idx = C->block_row_ptr[i];
      size_t nnz = 0;

      for (size_t j = A->block_row_ptr[i]; j < A->block_row_ptr[i + 1]; j++) {
        size_t block_col = A->block_col_idx[j];
        const double *A_block = &A->values[j * BLOCK_AREA];

        for (size_t k = B->block_row_ptr[block_col];
             k < B->block_row_ptr[block_col + 1]; k++) {
          size_t b_block_col = B->block_col_idx[k];
          const double *B_block = &B->values[k * BLOCK_AREA];

          if (temp_marks[b_block_col] != i + 1) {
            temp_marks[b_block_col] = i + 1;
            C->block_col_idx[next_idx + nnz] = b_block_col;
            multiply_blocks(A_block, B_block,
                            &temp_blocks[b_block_col * BLOCK_AREA]);
            nnz++;
          } else {
            double temp_block[BLOCK_AREA] = {0};
            multiply_blocks(A_block, B_block, temp_block);
            accumulate_blocks(&temp_blocks[b_block_col * BLOCK_AREA],
                              temp_block);
          }
        }
      }

      for (size_t j = 0; j < nnz; j++) {
        size_t idx = next_idx + j;
        size_t b_block_col = C->block_col_idx[idx];
        memcpy(&C->values[idx * BLOCK_AREA],
               &temp_blocks[b_block_col * BLOCK_AREA],
               BLOCK_AREA * sizeof(double));
        memset(&temp_blocks[b_block_col * BLOCK_AREA], 0,
               BLOCK_AREA * sizeof(double));
      }
    }
    free(temp_blocks);
    free(temp_marks);
  }
  free(block_row_nnz);
  return C;
}

Performance benchmark_multiplication(const BCSRMatrix *A, const BCSRMatrix *B) {
  Performance perf = {0};
  struct timeval start, end;

  BCSRMatrix *warmup = multiply_bcsr_matrices(A, B);
  free_bcsr_matrix(warmup);

  gettimeofday(&start, NULL);
  BCSRMatrix *C = multiply_bcsr_matrices(A, B);
  gettimeofday(&end, NULL);

  perf.compute_time =
      (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
  perf.memory_usage =
      (A->nnz_blocks + B->nnz_blocks + C->nnz_blocks) *
          (sizeof(size_t) + BLOCK_AREA * sizeof(double)) +
      (A->block_rows + B->block_rows + C->block_rows + 3) * sizeof(size_t);
  perf.flop_count = C->nnz_blocks * (2 * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);

  free_bcsr_matrix(C);
  return perf;
}

void free_bcsr_matrix(BCSRMatrix *matrix) {
  if (matrix) {
    free(matrix->block_row_ptr);
    free(matrix->block_col_idx);
    free(matrix->values);
    free(matrix);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix_A> <matrix_B>\n", argv[0]);
    return 1;
  }

  BCSRMatrix *A, *B;

  // Check file extensions and load accordingly
  char *ext_a = strrchr(argv[1], '.');
  char *ext_b = strrchr(argv[2], '.');

  if (ext_a && strcmp(ext_a, ".mtx") == 0) {
    A = load_matrix_from_mtx(argv[1]);
  } else {
    A = load_matrix_from_mat(argv[1]);
  }

  if (ext_b && strcmp(ext_b, ".mtx") == 0) {
    B = load_matrix_from_mtx(argv[2]);
  } else {
    B = load_matrix_from_mat(argv[2]);
  }

  if (!A || !B) {
    fprintf(stderr, "Error loading matrices\n");
    return 1;
  }

  Performance perf = benchmark_multiplication(A, B);

  printf("\nPerformance Results:\n");
  printf("Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);
  printf("Number of blocks: %zu\n", A->nnz_blocks);
  printf("Block density: %.2f%%\n",
         100.0 * A->nnz_blocks / (A->block_rows * A->block_cols));

  free_bcsr_matrix(A);
  free_bcsr_matrix(B);

  return 0;
}
