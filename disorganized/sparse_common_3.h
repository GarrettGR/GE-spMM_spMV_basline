#ifndef SPARSE_COMMON_H
#define SPARSE_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <omp.h>

#ifdef USE_PAPI
#include <papi.h>
#endif

#ifdef __linux__
#include <linux/perf_event.h>
#include <asm/unistd.h>
#endif

typedef struct {
  double l1_hit_rate;
  double l2_hit_rate;
  double l3_hit_rate;
  long tlb_misses;
  double bytes_transferred;
  double memory_bandwidth;
  long cache_misses;
} MemoryMetrics;

typedef struct {
  double total_energy;
  double avg_power;
  double energy_efficiency;
  double dram_energy;
  double cpu_energy;
  struct timeval start_time;
  struct timeval end_time;
} EnergyMetrics;

typedef struct {
  double speedup;
  double parallel_efficiency;
  double load_imbalance;
  int thread_migrations;
  double critical_path_time;
  int num_threads;
} ParallelMetrics;

typedef struct {
  double row_density_avg;
  double row_density_var;
  double compression_ratio;
  double access_regularity;
} SparsityMetrics;

typedef struct {
  double cpu_util;
  double vector_util;
  double mem_controller_util;
  double ipc;
  long long total_instructions;
  long long total_cycles;
} HardwareMetrics;

typedef struct {
  double stability_score;
  double max_rel_error;
  double condition_number;
  double residual_norm;
} NumericalMetrics;

typedef struct {
  int rows_a, cols_a, nnz_a;
  int rows_b, cols_b, nnz_b;
  int rows_c, cols_c, nnz_c;
  double density_a;
  double density_b;
  double density_c;

  double init_time;
  double conversion_time;
  double mult_time;
  double total_time;

  double flops;
  double gflops;
  double memory_usage;

  MemoryMetrics memory;
  EnergyMetrics energy;
  ParallelMetrics parallel;
  SparsityMetrics sparsity;
  HardwareMetrics hardware;
  NumericalMetrics numerical;
} Profile;

void profile_init(Profile *profile);
void profile_start(Profile *profile);
void profile_stop(Profile *profile);
void profile_collect_memory_metrics(Profile *profile);
void profile_collect_energy_metrics(Profile *profile);
void profile_collect_parallel_metrics(Profile *profile);
void profile_collect_hardware_metrics(Profile *profile);
void profile_print(const Profile *profile, FILE *output);

#ifdef __linux__
  static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);
#endif

void profile_init(Profile *profile) {
  memset(profile, 0, sizeof(Profile));

#ifdef USE_PAPI
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI library initialization error!\n");
  }
#endif

  profile->parallel.num_threads = omp_get_max_threads();
}

void profile_start(Profile *profile) {
  gettimeofday(&profile->energy.start_time, NULL);

#ifdef USE_PAPI
  long long start_values[4];
  int events[4] = {PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM, PAPI_TLB_DM};
  PAPI_start_counters(events, 4);
#endif
}

void profile_stop(Profile *profile) {
  gettimeofday(&profile->energy.end_time, NULL);
  profile->total_time = (profile->energy.end_time.tv_sec - profile->energy.start_time.tv_sec) +
                        (profile->energy.end_time.tv_usec - profile->energy.start_time.tv_usec) * 1e-6;

#ifdef USE_PAPI
  long long values[4];
  PAPI_stop_counters(values, 4);
  profile->memory.l1_hit_rate = calculate_hit_rate(values[0]);
  profile->memory.l2_hit_rate = calculate_hit_rate(values[1]);
  profile->memory.l3_hit_rate = calculate_hit_rate(values[2]);
  profile->memory.tlb_misses = values[3];
#endif
}

void profile_collect_memory_metrics(Profile *profile) {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  profile->memory_usage = usage.ru_maxrss * 1024.0;

  profile->memory.memory_bandwidth = profile->memory.bytes_transferred / profile->mult_time / 1e9;
}

void profile_collect_energy_metrics(Profile *profile) {
#ifdef __linux__
  // Note: This requires appropriate permissions and hardware support
  FILE *f = fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r");
  if (f) {
    unsigned long long energy;
    fscanf(f, "%llu", &energy);
    fclose(f);
    profile->energy.cpu_energy = energy * 1e-6;
  }
#endif

  if (profile->energy.total_energy > 0) {
    profile->energy.energy_efficiency = profile->gflops / profile->energy.total_energy;
  }
}

void profile_print(const Profile *profile, FILE *output) {
  fprintf(output, "\nRunning with %d threads\n", profile->parallel.num_threads);

  fprintf(output, "\n==================\n");
  fprintf(output, "Profiling Results:\n");
  fprintf(output, "==================\n");
  fprintf(output, "Multiplication time: %.6f seconds\n", profile->mult_time);
  fprintf(output, "Memory usage: %.2f MB\n", profile->memory_usage / 1048576.0);
  fprintf(output, "Memory Bandwidth: %.2f GB/s\n", profile->memory.memory_bandwidth);
  fprintf(output, "Cache Misses: %ld\n", profile->memory.cache_misses);
  fprintf(output, "Total FLOPS: %.2e\n", profile->flops);
  fprintf(output, "Performance: %.2f GFLOPS\n", profile->gflops);

  fprintf(output, "\n==================\n");
  fprintf(output, "Memory Performance:\n");
  fprintf(output, "==================\n");
  fprintf(output, "L1 Cache Hit Rate: %.2f%%\n", profile->memory.l1_hit_rate);
  fprintf(output, "L2 Cache Hit Rate: %.2f%%\n", profile->memory.l2_hit_rate);
  fprintf(output, "L3 Cache Hit Rate: %.2f%%\n", profile->memory.l3_hit_rate);
  fprintf(output, "TLB Misses: %ld\n", profile->memory.tlb_misses);
  fprintf(output, "Bytes Transferred: %.2f GB\n", profile->memory.bytes_transferred / 1e9);

  fprintf(output, "\n==================\n");
  fprintf(output, "Energy Metrics:\n");
  fprintf(output, "==================\n");
  fprintf(output, "Total Energy Consumed: %.2f Joules\n", profile->energy.total_energy);
  fprintf(output, "Average Power Usage: %.2f Watts\n", profile->energy.avg_power);
  fprintf(output, "Energy Efficiency: %.2f GFLOPS/Watt\n", profile->energy.energy_efficiency);
  fprintf(output, "DRAM Energy: %.2f Joules\n", profile->energy.dram_energy);
  fprintf(output, "CPU Package Energy: %.2f Joules\n", profile->energy.cpu_energy);

  fprintf(output, "\n==================\n");
  fprintf(output, "Matrix Statistics:\n");
  fprintf(output, "==================\n");
  fprintf(output, "Initialization Time: %.6f seconds\n", profile->init_time);
  fprintf(output, "Conversion Time: %.6f seconds\n", profile->conversion_time);
  fprintf(output, "Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)\n",
      profile->rows_a, profile->cols_a, profile->nnz_a, profile->density_a * 100.0);
  fprintf(output, "Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)\n",
      profile->rows_b, profile->cols_b, profile->nnz_b, profile->density_b * 100.0);
  fprintf(output, "Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)\n",
      profile->rows_c, profile->cols_c, profile->nnz_c, profile->density_c * 100.0);

  fprintf(output, "\nTotal Time: %.6f seconds\n", profile->total_time);
}

#endif // SPARSE_COMMON_H
