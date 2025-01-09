#ifndef PROFILER_H
#define PROFILER_H

#include <stddef.h>

#define MAX_SECTIONS 32
#define MAX_SECTION_NAME 64

/**
 * @brief Profiler context
 */
typedef struct profile_context profile_context;

/**
 * @brief Profiler flags / settings
 *
 * PROFILE_NONE: No profiling
 * PROFILE_TIME: Measure time spent in each section
 * PROFILE_MEMORY: Measure memory usage
 * PROFILE_ALL: Measure both time and memory
 */
typedef enum {
  PROFILE_NONE = 0,
  PROFILE_TIME = 1 << 0,
  PROFILE_MEMORY = 1 << 1,
  PROFILE_ALL = PROFILE_TIME | PROFILE_MEMORY
} profile_flags;

/**
 * @brief data structure for a timing section within a profile context
 */
typedef struct {
  char name[MAX_SECTION_NAME];
  double start_time;
  double elapsed_time;
  int active;
} timing_section;

/**
 * @brief data structure for tracking memory usage
*/
typedef struct {
  size_t current_bytes;
  size_t peak_bytes;
  size_t total_allocations;
  size_t total_deallocations;
} memory_info;

/**
 * @brief initialize a profiler context
 * 
 * @param flags Profiler flags
*/
profile_context* profile_init(profile_flags flags);

/**
 * @brief free the memory for a profiler context
 *
 * @param ctx Profiler context
*/
void profile_free(profile_context* ctx);

/**
 * @brief set all the fields of a profiler context to zero
 *
 * @param ctx Profiler context
*/
void profile_reset(profile_context* ctx);

/**
 * @brief demarcate the start of a section
 *
 * @param ctx Profiler context
 * @param section_name Name of the section to start
*/
void profile_start_section(profile_context* ctx, const char* section_name);

/**
 * @brief demarcate the end of a section
 *
 * @param ctx Profiler context
 * @param section_name Name of the section to end
*/
void profile_end_section(profile_context* ctx, const char* section_name);

/**
 * @brief get the time spent in a section
 *
 * @param ctx Profiler context
 * @param section_name Name of the section to get time for
 * @return double Time spent in the section
*/
double profile_get_section_time(const profile_context* ctx, const char* section_name);

/**
 * @brief record the size of a memory allocation
 * 
 * @param ctx Profiler context
 * @param bytes Number of bytes allocated
*/
void profile_record_alloc(profile_context* ctx, size_t bytes);

/**
 * @brief record the size of a memory deallocation
 *
 * @param ctx Profiler context
 * @param bytes Number of bytes deallocated
*/
void profile_record_free(profile_context* ctx, size_t bytes);

/**
 * @brief get the detailed memory usage information from a profiler context
 *
 * @param ctx Profiler context
 * @return memory_info Memory usage information
*/
memory_info profile_get_memory(const profile_context* ctx);

/**
 * @brief print the profiling results to stdout
 *
 * @param ctx Profiler context
*/
void profile_print_results(const profile_context* ctx);

#endif // PROFILER_H
