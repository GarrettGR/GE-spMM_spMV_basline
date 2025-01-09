#ifndef SIMPLE_PROFILER_H
#define SIMPLE_PROFILER_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

typedef struct {
  // Matrix statistics
  int rows_a, cols_a, nnz_a;
  int rows_b, cols_b, nnz_b;
  int rows_c, cols_c, nnz_c;
  double density_a;
  double density_b;
  double density_c;

  // Timing
  double init_time;
  double conv_time;
  double mult_time;
  double total_time;

  // Performance metrics
  long memory_usage;
  double memory_bandwidth;
  long cache_misses;
  long total_flops;
  double gflops;
  long bytes_transferred;

  // Timing helpers
  struct timeval start_time;
  struct timeval end_time;

  // Threading
  int num_threads;
} SimpleProfile;

void profile_init(SimpleProfile *prof) {
  memset(prof, 0, sizeof(SimpleProfile));

#ifdef _OPENMP
  prof->num_threads = omp_get_max_threads();
#elif defined(PTHREADS)
  prof->num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#else
  prof->num_threads = 1;
#endif
}

void profile_start(SimpleProfile *prof) {
  gettimeofday(&prof->start_time, NULL);
}

void profile_stop(SimpleProfile *prof) {
  gettimeofday(&prof->end_time, NULL);
  prof->total_time = (prof->end_time.tv_sec - prof->start_time.tv_sec) + (prof->end_time.tv_usec - prof->start_time.tv_usec) * 1e-6;
}

long get_memory_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss * 1024L;
}

void profile_collect_memory_metrics(SimpleProfile *prof) {
  prof->memory_usage = get_memory_usage();

  // calculate memory bandwidth (only if multiplication time is available)
  if (prof->mult_time > 0) {
    prof->memory_bandwidth = prof->bytes_transferred / (prof->mult_time * 1e9);
  }
}

void profile_update_densities(SimpleProfile *prof) {
  prof->density_a = (double)prof->nnz_a / (prof->rows_a * prof->cols_a) * 100.0;
  prof->density_b = (double)prof->nnz_b / (prof->rows_b * prof->cols_b) * 100.0;
  prof->density_c = (double)prof->nnz_c / (prof->rows_c * prof->cols_c) * 100.0;
}

void profile_update_flops(SimpleProfile *prof) {
  if (prof->mult_time > 0) {
    prof->gflops = prof->total_flops / (prof->mult_time * 1e9);
  }
}

void profile_print(const SimpleProfile *prof, FILE *output) {
  fprintf(output, "\nRunning with %d threads\n", prof->num_threads);

  fprintf(output, "\n==================\n");
  fprintf(output, "Profiling Results:\n");
  fprintf(output, "==================\n");
  fprintf(output, "Multiplication time: %.6f seconds\n", prof->mult_time);
  fprintf(output, "Memory usage: %.2f MB\n", prof->memory_usage / (1024.0 * 1024.0));
  fprintf(output, "Memory Bandwidth: %.2f GB/s\n", prof->memory_bandwidth);
  fprintf(output, "Cache Misses: %ld\n", prof->cache_misses);
  fprintf(output, "Total FLOPS: %ld\n", prof->total_flops);
  fprintf(output, "Performance: %.2f GFLOPS\n", prof->gflops);

  fprintf(output, "\n==================\n");
  fprintf(output, "Matrix Statistics:\n");
  fprintf(output, "==================\n");
  fprintf(output, "Initialization Time: %.6f seconds\n", prof->init_time);
  fprintf(output, "Conversion Time: %.6f seconds\n", prof->conv_time);
  fprintf(output, "Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)\n", prof->rows_a, prof->cols_a, prof->nnz_a, prof->density_a);
  fprintf(output, "Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)\n", prof->rows_b, prof->cols_b, prof->nnz_b, prof->density_b);
  fprintf(output, "Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)\n", prof->rows_c, prof->cols_c, prof->nnz_c, prof->density_c);

  fprintf(output, "\nTotal Time: %.6f seconds\n", prof->total_time);
}

#endif
