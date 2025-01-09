#include "utils/status.h"

const char* status_to_string(status_t status) {
  switch (status) {
    case STATUS_SUCCESS:
      return "Success";

    case STATUS_INVALID_INPUT:
      return "Invalid input";
    case STATUS_NULL_POINTER:
      return "Null pointer";
    case STATUS_INVALID_DIMENSIONS:
      return "Invalid dimensions";
    case STATUS_INVALID_FORMAT:
      return "Invalid format";
    case STATUS_INCOMPATIBLE_DIMENSIONS:
      return "Incompatible dimensions";

    case STATUS_ALLOCATION_FAILED:
      return "Memory allocation failed";
    case STATUS_SIZE_OVERFLOW:
      return "Size overflow";
    case STATUS_OUT_OF_MEMORY:
      return "Out of memory";

    case STATUS_NOT_IMPLEMENTED:
      return "Not implemented";
    case STATUS_UNSUPPORTED_OPERATION:
      return "Unsupported operation";

    case STATUS_NUMERICAL_ERROR:
      return "Numerical error";
    case STATUS_DIVISION_BY_ZERO:
      return "Division by zero";
    case STATUS_OVERFLOW:
      return "Overflow";
    case STATUS_UNDERFLOW:
      return "Underflow";

    case STATUS_FORMAT_VALIDATION_FAILED:
      return "Format validation failed";
    case STATUS_INVALID_SPARSE_STRUCTURE:
      return "Invalid sparse structure";
    case STATUS_UNSORTED_INDICES:
      return "Unsorted indices";

    case STATUS_FILE_ERROR:
      return "File error";
    case STATUS_IO_ERROR:
      return "I/O error";
    case STATUS_SYSTEM_ERROR:
      return "System error";

    case STATUS_PROFILER_ERROR:
      return "Profiler error";
    case STATUS_COUNTER_OVERFLOW:
      return "Counter overflow";

    case STATUS_INTERNAL_ERROR:
      return "Internal error";
    case STATUS_UNKNOWN_ERROR:
      return "Unknown error";

    default:
      return "Undefined error";
  }
}
