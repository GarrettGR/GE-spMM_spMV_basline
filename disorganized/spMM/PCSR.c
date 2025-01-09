#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

#define PADDING_FACTOR 8
#define ALIGN_TO(x) (((x) + PADDING_FACTOR - 1) & ~(PADDING_FACTOR - 1))

typedef struct {
  size_t rows;
  size_t cols;
  size_t nnz;
  size_t padded_nnz;
  size_t *row_ptr;
  size_t *col_idx;
  double *values;
} PCSRMatrix;

typedef struct {
  double load_time;
  double compute_time;
  double memory_usage;
  size_t flop_count;
} Performance;

PCSRMatrix* create_pcsr_matrix(size_t rows, size_t cols, size_t nnz) {
  PCSRMatrix* matrix = (PCSRMatrix*)malloc(sizeof(PCSRMatrix));
  if (!matrix) return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;
  matrix->padded_nnz = ALIGN_TO(nnz);

  matrix->row_ptr = (size_t*)aligned_alloc(64, (rows + 1) * sizeof(size_t));
  matrix->col_idx = (size_t*)aligned_alloc(64, matrix->padded_nnz * sizeof(size_t));
  matrix->values = (double*)aligned_alloc(64, matrix->padded_nnz * sizeof(double));

  if (!matrix->row_ptr || !matrix->col_idx || !matrix->values) {
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    free(matrix);
    return NULL;
  }

  return matrix;
}

PCSRMatrix* load_matrix_from_mat(const char* filename) {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
    if (!matfp) {
        fprintf(stderr, "Error opening MAT file: %s\n", filename);
        return NULL;
    }

    matvar_t *rows_var = Mat_VarRead(matfp, "rows");
    matvar_t *cols_var = Mat_VarRead(matfp, "cols");
    matvar_t *nnz_var = Mat_VarRead(matfp, "nnz");
    if (!rows_var || !cols_var || !nnz_var) {
        fprintf(stderr, "Error reading matrix metadata\n");
        Mat_Close(matfp);
        return NULL;
    }

    size_t rows = *(uint64_t*)rows_var->data;
    size_t cols = *(uint64_t*)cols_var->data;
    size_t nnz = *(uint64_t*)nnz_var->data;

    matvar_t *row_ptr_var = Mat_VarRead(matfp, "row_ptr");
    matvar_t *col_idx_var = Mat_VarRead(matfp, "col_idx");
    matvar_t *values_var = Mat_VarRead(matfp, "values");

    if (!row_ptr_var || !col_idx_var || !values_var) {
        fprintf(stderr, "Error reading matrix arrays\n");
        Mat_VarFree(rows_var);
        Mat_VarFree(cols_var);
        Mat_VarFree(nnz_var);
        Mat_Close(matfp);
        return NULL;
    }

    size_t* input_row_ptr = (size_t*)row_ptr_var->data;
    size_t max_row_length = 0;
    for (size_t i = 0; i < rows; i++) {
        size_t row_length = input_row_ptr[i + 1] - input_row_ptr[i];
        if (row_length > max_row_length) {
            max_row_length = row_length;
        }
    }

    const size_t SEGMENT_SIZE = 32;
    const float PADDING_FACTOR = 0.2;
    size_t padded_segment_size = SEGMENT_SIZE + (size_t)(SEGMENT_SIZE * PADDING_FACTOR);
    size_t num_segments = (rows + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

    PCSRMatrix* matrix = (PCSRMatrix*)malloc(sizeof(PCSRMatrix));
    if (!matrix) {
        Mat_VarFree(rows_var);
        Mat_VarFree(cols_var);
        Mat_VarFree(nnz_var);
        Mat_VarFree(row_ptr_var);
        Mat_VarFree(col_idx_var);
        Mat_VarFree(values_var);
        Mat_Close(matfp);
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->nnz = nnz;
    matrix->num_segments = num_segments;

    matrix->segments = (Segment*)malloc(num_segments * sizeof(Segment));
    size_t total_capacity = nnz + (size_t)(nnz * PADDING_FACTOR);
    matrix->col_idx = (size_t*)malloc(total_capacity * sizeof(size_t));
    matrix->values = (double*)malloc(total_capacity * sizeof(double));
    matrix->row_to_segment = (size_t*)malloc(rows * sizeof(size_t));

    if (!matrix->segments || !matrix->col_idx || !matrix->values || !matrix->row_to_segment) {
        free_pcsr_matrix(matrix);
        Mat_VarFree(rows_var);
        Mat_VarFree(cols_var);
        Mat_VarFree(nnz_var);
        Mat_VarFree(row_ptr_var);
        Mat_VarFree(col_idx_var);
        Mat_VarFree(values_var);
        Mat_Close(matfp);
        return NULL;
    }

    size_t current_pos = 0;
    for (size_t seg = 0; seg < num_segments; seg++) {
        size_t seg_start = seg * SEGMENT_SIZE;
        size_t seg_end = min(seg_start + SEGMENT_SIZE, rows);

        matrix->segments[seg].start_idx = current_pos;
        matrix->segments[seg].count = 0;

        size_t seg_nnz = 0;
        for (size_t i = seg_start; i < seg_end; i++) {
            seg_nnz += input_row_ptr[i + 1] - input_row_ptr[i];
        }
        size_t seg_capacity = seg_nnz + (size_t)(seg_nnz * PADDING_FACTOR);
        matrix->segments[seg].capacity = seg_capacity;

        for (size_t row = seg_start; row < seg_end; row++) {
            matrix->row_to_segment[row] = seg;
            size_t row_start = input_row_ptr[row];
            size_t row_end = input_row_ptr[row + 1];

            for (size_t j = row_start; j < row_end; j++) {
                matrix->col_idx[current_pos] = ((size_t*)col_idx_var->data)[j];
                matrix->values[current_pos] = ((double*)values_var->data)[j];
                current_pos++;
                matrix->segments[seg].count++;
            }
        }
        current_pos += (seg_capacity - matrix->segments[seg].count);
    }

    Mat_VarFree(rows_var);
    Mat_VarFree(cols_var);
    Mat_VarFree(nnz_var);
    Mat_VarFree(row_ptr_var);
    Mat_VarFree(col_idx_var);
    Mat_VarFree(values_var);
    Mat_Close(matfp);

    gettimeofday(&end, NULL);
    double load_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    printf("Matrix loaded in %.3f seconds\n", load_time);
    printf("Matrix stats: %zu rows, %zu cols, %zu non-zeros\n", rows, cols, nnz);
    printf("PCSR format: %zu segments, %.2f%% storage efficiency\n", num_segments, (100.0 * nnz) / total_capacity);
    printf("Average segment size: %.2f elements\n", (double)nnz / num_segments);

    return matrix;
}

static inline size_t min(size_t a, size_t b) {
    return (a < b) ? a : b;
}

PCSRMatrix* multiply_pcsr_matrices(const PCSRMatrix* A, const PCSRMatrix* B) {
  if (A->cols != B->rows) {
    fprintf(stderr, "Matrix dimensions incompatible for multiplication\n");
    return NULL;
  }

  size_t* row_nnz = (size_t*)calloc(A->rows, sizeof(size_t));

  #pragma omp parallel
  {
    double* temp_values = (double*)calloc(B->cols, sizeof(double));
    size_t* temp_marks = (size_t*)calloc(B->cols, sizeof(size_t));

    #pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < A->rows; i++) {
      size_t mark = i + 1;
      size_t nnz = 0;

      for (size_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
        size_t col = A->col_idx[j];
        if (col >= B->rows) continue;

        for (size_t k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
          size_t b_col = B->col_idx[k];
          if (b_col >= B->cols) continue;

          if (temp_marks[b_col] != mark) {
            temp_marks[b_col] = mark;
            nnz++;
          }
        }
      }
      row_nnz[i] = ALIGN_TO(nnz);
    }
    free(temp_values);
    free(temp_marks);
  }

  size_t total_nnz = 0;
  for (size_t i = 0; i < A->rows; i++) {
    total_nnz += row_nnz[i];
  }

  PCSRMatrix* C = create_pcsr_matrix(A->rows, B->cols, total_nnz);
  if (!C) {
    free(row_nnz);
    return NULL;
  }

  C->row_ptr[0] = 0;
  for (size_t i = 0; i < A->rows; i++) {
    C->row_ptr[i + 1] = C->row_ptr[i] + row_nnz[i];
  }

  #pragma omp parallel
  {
    double* temp_values = (double*)calloc(B->cols, sizeof(double));
    size_t* temp_marks = (size_t*)calloc(B->cols, sizeof(size_t));
    size_t* temp_cols = (size_t*)calloc(B->cols, sizeof(size_t));

    #pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < A->rows; i++) {
      size_t next_idx = C->row_ptr[i];
      size_t nnz = 0;
      size_t mark = i + 1;

      for (size_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
        size_t col = A->col_idx[j];
        if (col >= B->rows) continue;
        double val = A->values[j];

        for (size_t k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
          size_t b_col = B->col_idx[k];
          if (b_col >= B->cols) continue;

          if (temp_marks[b_col] != mark) {
            temp_marks[b_col] = mark;
            temp_cols[nnz++] = b_col;
            temp_values[b_col] = val * B->values[k];
          } else {
            temp_values[b_col] += val * B->values[k];
          }
        }
      }

      size_t end_idx = C->row_ptr[i + 1];
      for (size_t j = 0; j < nnz; j++) {
        size_t col = temp_cols[j];
        C->col_idx[next_idx] = col;
        C->values[next_idx] = temp_values[col];
        temp_values[col] = 0.0;
        next_idx++;
      }

      for (; next_idx < end_idx; next_idx++) {
        C->col_idx[next_idx] = SIZE_MAX;
        C->values[next_idx] = 0.0;
      }
    }
    free(temp_values);
    free(temp_marks);
    free(temp_cols);
  }
  free(row_nnz);
  return C;
}

Performance benchmark_multiplication(const PCSRMatrix* A, const PCSRMatrix* B) {
  Performance perf = {0};
  struct timeval start, end;

  PCSRMatrix* warmup = multiply_pcsr_matrices(A, B);
  free_pcsr_matrix(warmup);

  gettimeofday(&start, NULL);
  PCSRMatrix* C = multiply_pcsr_matrices(A, B);
  gettimeofday(&end, NULL);

  perf.compute_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
  perf.memory_usage = (A->padded_nnz + B->padded_nnz + C->padded_nnz) * (sizeof(double) + sizeof(size_t)) + (A->rows + B->rows + C->rows + 3) * sizeof(size_t);
  perf.flop_count = 2 * C->nnz;

  free_pcsr_matrix(C);
  return perf;
}

void free_pcsr_matrix(PCSRMatrix* matrix) {
  if (matrix) {
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    free(matrix);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix_A.mat> <matrix_B.mat>\n", argv[0]);
    return 1;
  }

  PCSRMatrix* A = load_matrix_from_mat(argv[1]);
  PCSRMatrix* B = load_matrix_from_mat(argv[2]);

  if (!A || !B) {
    fprintf(stderr, "Error loading matrices\n");
    return 1;
  }

  Performance perf = benchmark_multiplication(A, B);

  printf("\nPerformance Results:\n");
  printf("Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("Actual FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);
  printf("Padding Overhead: %.2f%%\n", ((double)(A->padded_nnz - A->nnz) / A->nnz + (double)(B->padded_nnz - B->nnz) / B->nnz) * 50.0);

  free_pcsr_matrix(A);
  free_pcsr_matrix(B);

  return 0;
}
