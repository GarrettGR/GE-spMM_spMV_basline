#ifndef STATUS_H
#define STATUS_H

/**
 * @brief Status codes for sparse matrix operations
 */
typedef enum {
  // success codes (0-99)
  STATUS_SUCCESS = 0,

  // input validation errors (100-199)
  STATUS_INVALID_INPUT = 100,
  STATUS_NULL_POINTER = 101,
  STATUS_INVALID_DIMENSIONS = 102,
  STATUS_INVALID_FORMAT = 103,
  STATUS_INCOMPATIBLE_DIMENSIONS = 104,

  // memory errors (200-299)
  STATUS_ALLOCATION_FAILED = 200,
  STATUS_SIZE_OVERFLOW = 201,
  STATUS_OUT_OF_MEMORY = 202,

  // implementation status (300-399)
  STATUS_NOT_IMPLEMENTED = 300,
  STATUS_UNSUPPORTED_OPERATION = 301,

  // computation errors (400-499)
  STATUS_NUMERICAL_ERROR = 400,
  STATUS_DIVISION_BY_ZERO = 401,
  STATUS_OVERFLOW = 402,
  STATUS_UNDERFLOW = 403,

  // format-specific errors (500-599)
  STATUS_FORMAT_VALIDATION_FAILED = 500,
  STATUS_INVALID_SPARSE_STRUCTURE = 501,
  STATUS_UNSORTED_INDICES = 502,

  // system errors (600-699)
  STATUS_FILE_ERROR = 600,
  STATUS_IO_ERROR = 601,
  STATUS_SYSTEM_ERROR = 602,

  // profiling errors (700-799)
  STATUS_PROFILER_ERROR = 700,
  STATUS_COUNTER_OVERFLOW = 701,

  // internal errors (900-999)
  STATUS_INTERNAL_ERROR = 900,
  STATUS_UNKNOWN_ERROR = 999
} status_t;

/**
 * @brief Get string description of status code
 *
 * @param status Status code
 * @return const char* String description
 */
const char* status_to_string(status_t status);

/**
 * @brief Check if status code indicates success
 *
 * @param status Status code to check
 * @return int 1 if success, 0 if error
 */
static inline int status_is_success(status_t status) {
  return status < 100;
}

#endif // STATUS_H
