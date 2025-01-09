#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

typedef struct {
  int rows;
  int cols;
  int nnz;
  double *values;
  int *col_indices;
  int *row_ptr;
} CSRMatrix;

typedef struct {
  double init_time;
  double mult_time;
  double total_time;
  long memory_usage;
  double flops;
  double gflops;
} Profile;

CSRMatrix* create_csr_matrix(int rows, int cols, int nnz);
void free_csr_matrix(CSRMatrix *matrix);
CSRMatrix* multiply_csr_matrices(CSRMatrix *A, CSRMatrix *B, Profile *profile);
void print_profile(Profile *profile, CSRMatrix *A, CSRMatrix *B, CSRMatrix *C);
long get_memory_usage();

CSRMatrix* create_csr_matrix(int rows, int cols, int nnz) {
  CSRMatrix *matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
  if (!matrix) {
    fprintf(stderr, "Memory allocation failed for matrix structure\n");
    exit(1);
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  matrix->values = (double*)malloc(nnz * sizeof(double));
  matrix->col_indices = (int*)malloc(nnz * sizeof(int));
  matrix->row_ptr = (int*)malloc((rows + 1) * sizeof(int));

  if (!matrix->values || !matrix->col_indices || !matrix->row_ptr) {
    fprintf(stderr, "Memory allocation failed for matrix arrays\n");
    free_csr_matrix(matrix);
    exit(1);
  }

  return matrix;
}

void free_csr_matrix(CSRMatrix *matrix) {
  if (matrix) {
    free(matrix->values);
    free(matrix->col_indices);
    free(matrix->row_ptr);
    free(matrix);
  }
}

long get_memory_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss * 1024L;
}

CSRMatrix* multiply_csr_matrices(CSRMatrix *A, CSRMatrix *B, Profile *profile) {
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  if (A->cols != B->rows) {
    fprintf(stderr, "Matrix dimensions incompatible for multiplication\n");
    exit(1);
  }

  int *nnz_per_row = (int*)calloc(A->rows, sizeof(int));

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic, 32)
    for (int i = 0; i < A->rows; i++) {
      int *mask = (int*)calloc(B->cols, sizeof(int));
      for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
        int col = A->col_indices[j];
        for (int k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
          mask[B->col_indices[k]] = 1;
        }
      }

      for (int j = 0; j < B->cols; j++) {
        if (mask[j]) nnz_per_row[i]++;
      }
      free(mask);
    }
  }

  int total_nnz = 0;
  for (int i = 0; i < A->rows; i++) {
    total_nnz += nnz_per_row[i];
  }

  CSRMatrix *C = create_csr_matrix(A->rows, B->cols, total_nnz);

  C->row_ptr[0] = 0;
  for (int i = 0; i < A->rows; i++) {
    C->row_ptr[i + 1] = C->row_ptr[i] + nnz_per_row[i];
  }

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic, 32)
    for (int i = 0; i < A->rows; i++) {
      double *temp = (double*)calloc(B->cols, sizeof(double));
      int *next_idx = (int*)malloc(B->cols * sizeof(int));
      int next = C->row_ptr[i];

      for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
        int col = A->col_indices[j];
        double val = A->values[j];

        for (int k = B->row_ptr[col]; k < B->row_ptr[col + 1]; k++) {
          temp[B->col_indices[k]] += val * B->values[k];
        }
      }

      for (int j = 0; j < B->cols; j++) {
        if (temp[j] != 0.0) {
          C->values[next] = temp[j];
          C->col_indices[next] = j;
          next++;
        }
      }

      free(temp);
      free(next_idx);
    }
  }

  gettimeofday(&end_time, NULL);
  profile->mult_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

  profile->flops = 2.0 * total_nnz;
  profile->gflops = profile->flops / (profile->mult_time * 1e9);
  profile->memory_usage = get_memory_usage();

  free(nnz_per_row);
  return C;
}

void print_profile(Profile *profile, CSRMatrix *A, CSRMatrix *B, CSRMatrix *C) {
  printf("\n==================\n");
  printf("Profiling Results:\n");
  printf("==================\n");
  printf("Initialization Time: %.4f seconds\n", profile->init_time);
  printf("Multiplication Time: %.4f seconds\n", profile->mult_time);
  printf("Total Time: %.4f seconds\n", profile->total_time);
  printf("Peak Memory Usage: %.2f MB\n", profile->memory_usage / (1024.0 * 1024.0));
  printf("Total FLOPS: %.2e\n", profile->flops);
  printf("Performance: %.2f GFLOPS\n", profile->gflops);
  printf("\n==================\n");
  printf("Matrix Statistics:\n");
  printf("==================\n");
  printf("Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)\n",
         A->rows, A->cols, A->nnz, (100.0 * A->nnz) / (A->rows * A->cols));
  printf("Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)\n",
         B->rows, B->cols, B->nnz, (100.0 * B->nnz) / (B->rows * B->cols));
  printf("Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)\n",
         C->rows, C->cols, C->nnz, (100.0 * C->nnz) / (C->rows * C->cols));
}

int main(int argc, char *argv[]) {
  struct timeval total_start, total_end;
  gettimeofday(&total_start, NULL);

  Profile profile = {0};
  struct timeval init_start, init_end;
  gettimeofday(&init_start, NULL);

  int num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  printf("Running with %d OpenMP threads\n", num_threads);

  CSRMatrix *A = create_csr_matrix(1000, 1000, 10000);
  CSRMatrix *B = create_csr_matrix(1000, 1000, 10000);

  srand(42);
  for (int i = 0; i < A->nnz; i++) {
    A->values[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    A->col_indices[i] = rand() % A->cols;
  }
  for (int i = 0; i < B->nnz; i++) {
    B->values[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    B->col_indices[i] = rand() % B->cols;
  }

  for (int i = 0; i <= A->rows; i++) {
    A->row_ptr[i] = (i * A->nnz) / A->rows;
  }
  for (int i = 0; i <= B->rows; i++) {
    B->row_ptr[i] = (i * B->nnz) / B->rows;
  }

  gettimeofday(&init_end, NULL);
  profile.init_time = (init_end.tv_sec - init_start.tv_sec) +
    (init_end.tv_usec - init_start.tv_usec) / 1000000.0;

  CSRMatrix *C = multiply_csr_matrices(A, B, &profile);

  gettimeofday(&total_end, NULL);
  profile.total_time = (total_end.tv_sec - total_start.tv_sec) +
                       (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  print_profile(&profile, A, B, C);

  free_csr_matrix(A);
  free_csr_matrix(B);
  free_csr_matrix(C);

  return 0;
}
