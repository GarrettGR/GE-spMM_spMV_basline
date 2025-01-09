#ifndef SPARSE_COMMON_H
#define SPARSE_COMMON_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

typedef struct {
  double init_time;
  double convert_time;
  double mult_time;
  double memory_usage;
  double total_flops;
  int cache_misses;
  double bandwidth;
} ProfileData;

static inline double get_time_usec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1e6 + tv.tv_usec;
}

void print_profile_results(const char *format_name, ProfileData *prof, int rows, int cols, int nnz) {
  printf("\nProfiling Results for %s format:\n", format_name);
  printf("=====================================\n");
  printf("Initialization time:    %.6f seconds\n", prof->init_time);
  printf("Conversion time:        %.6f seconds\n", prof->convert_time);
  printf("Multiplication time:    %.6f seconds\n", prof->mult_time);
  printf("Total time:             %.6f seconds\n", prof->init_time + prof->convert_time + prof->mult_time);
  printf("Memory usage:           %.2f MB\n", prof->memory_usage);
  printf("Total FLOPS:            %.0f\n", prof->total_flops);
  printf("GFLOPS:                 %.2f\n", prof->total_flops / (prof->mult_time * 1e9));
  printf("Cache misses:           %d\n", prof->cache_misses);
  printf("Memory bandwidth:       %.2f GB/s\n", prof->bandwidth);
  printf("Matrix density:         %.2f%%\n", (100.0 * nnz) / (rows * cols));
}

void generate_random_pattern(int rows, int cols, int nnz, int *row_indices, int *col_indices) {
  int *positions = (int *)malloc(rows * cols * sizeof(int));

  for (int i = 0; i < rows * cols; i++) {
    positions[i] = i;
  }

  for (int i = rows * cols - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int temp = positions[i];
    positions[i] = positions[j];
    positions[j] = temp;
  }

  for (int i = 0; i < nnz; i++) {
    row_indices[i] = positions[i] / cols;
    col_indices[i] = positions[i] % cols;
  }

  for (int i = 0; i < nnz - 1; i++) {
    for (int j = 0; j < nnz - i - 1; j++) {
      if (row_indices[j] > row_indices[j + 1]) {
        int temp = row_indices[j];
        row_indices[j] = row_indices[j + 1];
        row_indices[j + 1] = temp;

        temp = col_indices[j];
        col_indices[j] = col_indices[j + 1];
        col_indices[j + 1] = temp;
      }
    }
  }

  free(positions);
}

#endif // SPARSE_COMMON_H
