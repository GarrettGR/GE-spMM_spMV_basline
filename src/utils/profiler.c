#include "utils/profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// #include <papi.h>

struct profile_context {
  profile_flags flags;
  timing_section sections[MAX_SECTIONS];
  int section_count;
  memory_info memory;
};

static double get_current_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static timing_section *find_section(profile_context *ctx, const char *name) {
  for (int i = 0; i < ctx->section_count; i++) {
    if (strcmp(ctx->sections[i].name, name) == 0) {
      return &ctx->sections[i];
    }
  }
  if (ctx->section_count < MAX_SECTIONS) {
    timing_section *section = &ctx->sections[ctx->section_count++];
    strncpy(section->name, name, MAX_SECTION_NAME - 1);
    section->name[MAX_SECTION_NAME - 1] = '\0';
    section->elapsed_time = 0.0;
    section->active = 0;
    return section;
  }
  return NULL;
}

profile_context *profile_init(profile_flags flags) {
  profile_context *ctx = calloc(1, sizeof(profile_context));
  if (!ctx) return NULL;

  ctx->flags = flags;
  ctx->section_count = 0;
  return ctx;
}

void profile_reset(profile_context *ctx) {
  if (!ctx) return;
  ctx->section_count = 0;
  memset(&ctx->memory, 0, sizeof(memory_info));
}

void profile_free(profile_context *ctx) { free(ctx); }

void profile_start_section(profile_context *ctx, const char *section_name) {
  if (!ctx || !(ctx->flags & PROFILE_TIME)) return;

  timing_section *section = find_section(ctx, section_name);
  if (section) {
    section->start_time = get_current_time();
    section->active = 1;
  }
}

void profile_end_section(profile_context *ctx, const char *section_name) {
  if (!ctx || !(ctx->flags & PROFILE_TIME)) return;

  timing_section *section = find_section(ctx, section_name);
  if (section && section->active) {
    section->elapsed_time += get_current_time() - section->start_time;
    section->active = 0;
  }
}

double profile_get_section_time(const profile_context *ctx, const char *section_name) {
  if (!ctx || !(ctx->flags & PROFILE_TIME)) return 0.0;

  for (int i = 0; i < ctx->section_count; i++) {
    if (strcmp(ctx->sections[i].name, section_name) == 0) {
      return ctx->sections[i].elapsed_time;
    }
  }
  return 0.0;
}

void profile_record_alloc(profile_context *ctx, size_t bytes) {
  if (!ctx || !(ctx->flags & PROFILE_MEMORY)) return;
  ctx->memory.current_bytes += bytes;
  ctx->memory.total_allocations++;
  if (ctx->memory.current_bytes > ctx->memory.peak_bytes) {
    ctx->memory.peak_bytes = ctx->memory.current_bytes;
  }
}

void profile_record_free(profile_context *ctx, size_t bytes) {
  if (!ctx || !(ctx->flags & PROFILE_MEMORY)) return;
  ctx->memory.current_bytes -= bytes;
  ctx->memory.total_deallocations++;
}

memory_info profile_get_memory(const profile_context *ctx) {
  static const memory_info empty = {0};
  return ctx ? ctx->memory : empty;
}

void profile_print_results(const profile_context *ctx) {
  if (!ctx) return;

  if (ctx->flags & PROFILE_TIME) {
    printf("\nTiming Results:\n");
    for (int i = 0; i < ctx->section_count; i++) {
      printf("  %-20s: %.6f seconds\n", ctx->sections[i].name, ctx->sections[i].elapsed_time);
    }
  }
  if (ctx->flags & PROFILE_MEMORY) {
    printf("\nMemory Usage:\n");
    printf("  Current: %.2f MB\n", ctx->memory.current_bytes / (1024.0 * 1024.0));
    printf("  Peak:    %.2f MB\n", ctx->memory.peak_bytes / (1024.0 * 1024.0));
    printf("  Allocs:  %zu\n", ctx->memory.total_allocations);
    printf("  Frees:   %zu\n", ctx->memory.total_deallocations);
  }
}
