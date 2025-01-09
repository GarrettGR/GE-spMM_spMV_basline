#ifndef SPARSE_COMMON_H
#define SPARSE_COMMON_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

typedef struct {
  int rows;
  int cols;
  int nnz;
  double *values;
  int *col_indices;
  int *row_ptr;
} CSRMatrix;

typedef struct {
  int rows;
  int cols;
  int nnz;
  int *row_indices;
  int *col_indices;
  double *values;
} COOMatrix;

typedef struct {
  int rows;
  int cols;
  int max_nnz_per_row;
  int *cols;
  double *values;
} ELLMatrix;

typedef struct {
  double init_time;
  double convert_time;
  double mult_time;
  double total_time;
  long memory_usage;
  double flops;
  double gflops;
  int num_threads;
} Profile;

static inline double get_time_usec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1e6 + tv.tv_usec;
}

static inline double get_time_sec() { return get_time_usec() / 1e6; }

static inline long get_peak_memory_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss * 1024L;
}

static inline double calculate_density(int nnz, int rows, int cols) {
  return (100.0 * nnz) / (rows * cols);
}

static inline void print_profile_results(const char *format_name, Profile *prof, int a_nnz, int b_nnz, int c_nnz, int rows, int cols) {
  printf("\nProfiling Results for %s format:\n", format_name);
  printf("=====================================\n");
  printf("Running with %d OpenMP threads\n", prof->num_threads);
  printf("Initialization time:  %.6f seconds\n", prof->init_time);
  printf("Conversion time:      %.6f seconds\n", prof->convert_time);
  printf("Multiplication time:  %.6f seconds\n", prof->mult_time);
  printf("Total time:          %.6f seconds\n", prof->total_time);
  printf("Memory usage:        %.2f MB\n", prof->memory_usage / (1024.0 * 1024.0));
  printf("Total FLOPS:         %.2e\n", prof->flops);
  printf("Performance:         %.2f GFLOPS\n", prof->gflops);
  printf("\nMatrix Statistics:\n");
  printf("Input Matrix A:      %d non-zeros (%.2f%% dense)\n", a_nnz, calculate_density(a_nnz, rows, cols));
  printf("Input Matrix B:      %d non-zeros (%.2f%% dense)\n", b_nnz, calculate_density(b_nnz, rows, cols));
  printf("Result Matrix C:     %d non-zeros (%.2f%% dense)\n", c_nnz, calculate_density(c_nnz, rows, cols));
}

static inline void generate_random_sparse_data(double *mat, int rows, int cols, double sparsity) {
  for (int i = 0; i < rows * cols; i++) {
    if ((double)rand() / RAND_MAX < sparsity) {
      mat[i] = (double)rand() / RAND_MAX;
    } else {
      mat[i] = 0.0;
    }
  }
}

static inline void init_profile(Profile *prof) {
  memset(prof, 0, sizeof(Profile));
  prof->num_threads = omp_get_max_threads();
}

#endif
