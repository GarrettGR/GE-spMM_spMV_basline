#include "parsers/parse_mtx.h"
// #include <ctype.h>
// #include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MTX_BUFFER_SIZE (1024 * 1024) // 1MB buffer for file reading
#define MTX_LINE_MAX 2048
#define MTX_INITIAL_CAPACITY 1048576 // initial allocation size for dynamic arrays (???)

typedef struct {
  uint64_t *rows;
  uint64_t *cols;
  double *values;
  uint64_t size;
  uint64_t capacity;
} entry_buffer_t;

typedef struct {
  char *buffer;
  size_t capacity;
  size_t size;
  size_t pos;
  FILE *fp;
} buffered_reader_t;

static status_t create_buffered_reader(const char *filename, buffered_reader_t *reader) {
  reader->buffer = (char *)malloc(MTX_BUFFER_SIZE);
  if (!reader->buffer) return STATUS_ALLOCATION_FAILED;

  reader->fp = fopen(filename, "r");
  if (!reader->fp) {
    free(reader->buffer);
    return STATUS_FILE_ERROR;
  }

  reader->capacity = MTX_BUFFER_SIZE;
  reader->size = 0;
  reader->pos = 0;

  return STATUS_SUCCESS;
}

static void free_buffered_reader(buffered_reader_t *reader) {
  if (reader->buffer) free(reader->buffer);
  if (reader->fp) fclose(reader->fp);
}

static status_t read_line(buffered_reader_t *reader, char *line, size_t max_len) {
  size_t line_pos = 0;

  while (1) {
    if (reader->pos >= reader->size) {
      reader->size = fread(reader->buffer, 1, reader->capacity, reader->fp);
      reader->pos = 0;
      if (reader->size == 0) {
        if (line_pos == 0) return STATUS_FILE_ERROR;
        break;
      }
    }

    while (reader->pos < reader->size && line_pos < max_len - 1) {
      char c = reader->buffer[reader->pos++];
      if (c == '\n') {
        line[line_pos] = '\0';
        return STATUS_SUCCESS;
      }
      if (c != '\r') line[line_pos++] = c;
    }

    if (line_pos >= max_len - 1) break;
  }

  line[line_pos] = '\0';
  return STATUS_SUCCESS;
}

static status_t skip_comments(buffered_reader_t *reader) {
  char line[MTX_LINE_MAX];
  status_t status;

  do {
    status = read_line(reader, line, MTX_LINE_MAX);
    if (status != STATUS_SUCCESS) return status;
  } while (line[0] == '%');

  long offset = -(strlen(line) + 1);
  fseek(reader->fp, offset, SEEK_CUR);
  reader->pos = reader->size;

  return STATUS_SUCCESS;
}

status_t mtx_read_info(const char *filename, mtx_info_t *info) {
  if (!filename || !info) return STATUS_NULL_POINTER;

  buffered_reader_t reader;
  status_t status = create_buffered_reader(filename, &reader);
  if (status != STATUS_SUCCESS) return status;

  char line[MTX_LINE_MAX];
  status = read_line(&reader, line, MTX_LINE_MAX);
  if (status != STATUS_SUCCESS) {
    free_buffered_reader(&reader);
    return status;
  }

  if (strncmp(line, "%%MatrixMarket", 14) != 0) {
    free_buffered_reader(&reader);
    return STATUS_INVALID_FORMAT;
  }

  char object[64], format[64], field[64], symmetry[64];
  if (sscanf(line, "%%MatrixMarket %63s %63s %63s %63s", object, format, field, symmetry) != 4) {
    free_buffered_reader(&reader);
    return STATUS_INVALID_FORMAT;
  }

  if (strcmp(object, "matrix") != 0 || strcmp(format, "coordinate") != 0) {
    free_buffered_reader(&reader);
    return STATUS_INVALID_FORMAT;
  }

  info->is_pattern = (strcmp(field, "pattern") == 0);
  info->is_complex = (strcmp(field, "complex") == 0);
  info->is_symmetric = (strcmp(symmetry, "symmetric") == 0);

  status = skip_comments(&reader);
  if (status != STATUS_SUCCESS) {
    free_buffered_reader(&reader);
    return status;
  }

  status = read_line(&reader, line, MTX_LINE_MAX);
  if (status != STATUS_SUCCESS || sscanf(line, "%llu %llu %llu", &info->rows, &info->cols, &info->nnz) != 3) {
    free_buffered_reader(&reader);
    return STATUS_INVALID_FORMAT;
  }

  free_buffered_reader(&reader);
  return STATUS_SUCCESS;
}

static inline void swap_entries(uint64_t *col_idx, double *values, uint64_t i, uint64_t j) {
  uint64_t temp_col = col_idx[i];
  col_idx[i] = col_idx[j];
  col_idx[j] = temp_col;

  double temp_val = values[i];
  values[i] = values[j];
  values[j] = temp_val;
}

static void quick_sort_row(uint64_t *col_idx, double *values, uint64_t left, uint64_t right) {
  if (left >= right) return;

  uint64_t pivot_idx = left + (right - left) / 2;
  uint64_t pivot = col_idx[pivot_idx];

  swap_entries(col_idx, values, pivot_idx, right);

  uint64_t store_idx = left;
  for (uint64_t i = left; i < right; i++) {
    if (col_idx[i] < pivot) {
      swap_entries(col_idx, values, store_idx, i);
      store_idx++;
    }
  }

  swap_entries(col_idx, values, store_idx, right);

  if (store_idx > left) quick_sort_row(col_idx, values, left, store_idx - 1);
  if (store_idx < right) quick_sort_row(col_idx, values, store_idx + 1, right);
}

static status_t allocate_csr_matrix(uint64_t rows, uint64_t cols, uint64_t nnz, csr_matrix **matrix, profile_context *prof) {
  status_t status = csr_create(rows, cols, nnz, matrix, prof);
  if (status != STATUS_SUCCESS) return status;

  memset((*matrix)->row_ptr, 0, (rows + 1) * sizeof(uint64_t));
  return STATUS_SUCCESS;
}

static status_t init_entry_buffer(entry_buffer_t *buffer, uint64_t initial_capacity) {
  buffer->rows = (uint64_t *)malloc(initial_capacity * sizeof(uint64_t));
  buffer->cols = (uint64_t *)malloc(initial_capacity * sizeof(uint64_t));
  buffer->values = (double *)malloc(initial_capacity * sizeof(double));

  if (!buffer->rows || !buffer->cols || !buffer->values) {
    free(buffer->rows);
    free(buffer->cols);
    free(buffer->values);
    return STATUS_ALLOCATION_FAILED;
  }

  buffer->size = 0;
  buffer->capacity = initial_capacity;
  return STATUS_SUCCESS;
}

static status_t resize_entry_buffer(entry_buffer_t *buffer) {
  uint64_t new_capacity = buffer->capacity * 2;

  uint64_t *new_rows = (uint64_t *)realloc(buffer->rows, new_capacity * sizeof(uint64_t));
  uint64_t *new_cols = (uint64_t *)realloc(buffer->cols, new_capacity * sizeof(uint64_t));
  double *new_values = (double *)realloc(buffer->values, new_capacity * sizeof(double));

  if (!new_rows || !new_cols || !new_values) {
    free(new_rows);
    free(new_cols);
    free(new_values);
    return STATUS_ALLOCATION_FAILED;
  }

  buffer->rows = new_rows;
  buffer->cols = new_cols;
  buffer->values = new_values;
  buffer->capacity = new_capacity;
  return STATUS_SUCCESS;
}

static void free_entry_buffer(entry_buffer_t *buffer) {
  free(buffer->rows);
  free(buffer->cols);
  free(buffer->values);
}

status_t mtx_to_format(const char *filename, mtx_format_t format, void **matrix) {
  if (!filename || !matrix) return STATUS_NULL_POINTER;
  
mtx_info_t info;
  status_t status = mtx_read_info(filename, &info);
  if (status != STATUS_SUCCESS) return status;

  if (format == MTX_FORMAT_CSR) {
    return mtx_to_csr(filename, (csr_matrix **)matrix);
  } else if (format == MTX_FORMAT_COO) {
    // return mtx_to_coo(filename, (coo_matrix **)matrix);
  }

  return STATUS_INVALID_FORMAT;
}

status_t mtx_to_csr(const char *filename, csr_matrix **matrix) {
  if (!filename || !matrix) return STATUS_NULL_POINTER;

  mtx_info_t info;
  status_t status = mtx_read_info(filename, &info);
  if (status != STATUS_SUCCESS) return status;

  uint64_t actual_nnz = info.is_symmetric ? info.nnz * 2 - count_diagonal_entries(filename, &info) : info.nnz;

  buffered_reader_t reader;
  status = create_buffered_reader(filename, &reader);
  if (status != STATUS_SUCCESS) return status;

  char line[MTX_LINE_MAX];
  do {
    status = read_line(&reader, line, MTX_LINE_MAX);
    if (status != STATUS_SUCCESS) {
      free_buffered_reader(&reader);
      return status;
    }
  } while (line[0] == '%');

  status = read_line(&reader, line, MTX_LINE_MAX);
  if (status != STATUS_SUCCESS) {
    free_buffered_reader(&reader);
    return status;
  }

  entry_buffer_t buffer;
  status = init_entry_buffer(&buffer, MTX_INITIAL_CAPACITY);
  if (status != STATUS_SUCCESS) {
    free_buffered_reader(&reader);
    return status;
  }

  while (read_line(&reader, line, MTX_LINE_MAX) == STATUS_SUCCESS) {
    uint64_t row, col;
    double val;

    if (info.is_pattern) {
      if (sscanf(line, "%llu %llu", &row, &col) != 2) continue;
      val = 1.0;
    } else {
      if (sscanf(line, "%llu %llu %lf", &row, &col, &val) != 3) continue; 
    }

    row--; col--; // zero index the matrix

    if (buffer.size >= buffer.capacity) {
      status = resize_entry_buffer(&buffer);
      if (status != STATUS_SUCCESS) {
        free_entry_buffer(&buffer);
        free_buffered_reader(&reader);
        return status;
      }
    }

    buffer.rows[buffer.size] = row;
    buffer.cols[buffer.size] = col;
    buffer.values[buffer.size] = val;
    buffer.size++;

    if (info.is_symmetric && row != col) {
      if (buffer.size >= buffer.capacity) {
        status = resize_entry_buffer(&buffer);
        if (status != STATUS_SUCCESS) {
          free_entry_buffer(&buffer);
          free_buffered_reader(&reader);
          return status;
        }
      }

      buffer.rows[buffer.size] = col;
      buffer.cols[buffer.size] = row;
      buffer.values[buffer.size] = val;
      buffer.size++;
    }
  }

  free_buffered_reader(&reader);

  status = allocate_csr_matrix(info.rows, info.cols, buffer.size, matrix, NULL);
  if (status != STATUS_SUCCESS) {
    free_entry_buffer(&buffer);
    return status;
  }

  for (uint64_t i = 0; i < buffer.size; i++) {
    (*matrix)->row_ptr[buffer.rows[i] + 1]++;
  }

  for (uint64_t i = 0; i < info.rows; i++) {
    (*matrix)->row_ptr[i + 1] += (*matrix)->row_ptr[i];
  }

  uint64_t *row_counts = (uint64_t *)calloc(info.rows, sizeof(uint64_t));
  if (!row_counts) {
    free_entry_buffer(&buffer);
    csr_free(*matrix, NULL);
    return STATUS_ALLOCATION_FAILED;
  }

  for (uint64_t i = 0; i < buffer.size; i++) {
    uint64_t row = buffer.rows[i];
    uint64_t pos = (*matrix)->row_ptr[row] + row_counts[row]++;
    (*matrix)->col_idx[pos] = buffer.cols[i];
    (*matrix)->values[pos] = buffer.values[i];
  }

#pragma omp parallel for schedule(dynamic) if (info.rows > 1000) // TODO: make this (openmp) optional / dynamic
  for (uint64_t i = 0; i < info.rows; i++) {
    uint64_t start = (*matrix)->row_ptr[i];
    uint64_t end = (*matrix)->row_ptr[i + 1];
    if (end - start > 1) {
      quick_sort_row((*matrix)->col_idx, (*matrix)->values, start, end - 1);
    }
  }

  free(row_counts);
  free_entry_buffer(&buffer);
  return STATUS_SUCCESS;
}

uint64_t count_diagonal_entries(const char* filename, const mtx_info_t* info) {
  if (!info->is_symmetric) return 0;

  buffered_reader_t reader;
  status_t status = create_buffered_reader(filename, &reader);
  if (status != STATUS_SUCCESS) return 0;

  char line[MTX_LINE_MAX];
  do {
    if (read_line(&reader, line, MTX_LINE_MAX) != STATUS_SUCCESS) {
      free_buffered_reader(&reader);
      return 0;
    }
  } while (line[0] == '%');

  if (read_line(&reader, line, MTX_LINE_MAX) != STATUS_SUCCESS) {
    free_buffered_reader(&reader);
    return 0;
  }

  uint64_t diagonal_count = 0;
  uint64_t row, col;
  
  while (read_line(&reader, line, MTX_LINE_MAX) == STATUS_SUCCESS) {
    if (info->is_pattern) {
      if (sscanf(line, "%llu %llu", &row, &col) != 2) continue;
    } else {
        double val;
      if (sscanf(line, "%llu %llu %lf", &row, &col, &val) != 3) continue;
    }
    
    // row--; col--; // zero index the matrix

    if (row == col) diagonal_count++;
  }

  free_buffered_reader(&reader);
  return diagonal_count;
}

status_t mtx_get_memory_estimate(const char* filename, mtx_format_t format, size_t* size) {
  if (!filename || !size) return STATUS_NULL_POINTER;

  mtx_info_t info;
  status_t status = mtx_read_info(filename, &info);
  if (status != STATUS_SUCCESS) return status;

  uint64_t diagonal_entries = 0;
  uint64_t actual_nnz = info.nnz;
  
  if (info.is_symmetric) {
    diagonal_entries = count_diagonal_entries(filename, &info);
    if (diagonal_entries == 0 && status != STATUS_SUCCESS) {
      return status;
    }
    actual_nnz = (info.nnz * 2) - diagonal_entries;
  }

  switch (format) {
    case MTX_FORMAT_CSR:
      *size = sizeof(csr_matrix) +  (actual_nnz * sizeof(double)) + (actual_nnz * sizeof(uint64_t)) + ((info.rows + 1) * sizeof(uint64_t));
      break;
    case MTX_FORMAT_COO: {
      // *size = sizeof(coo_matrix) +  (actual_nnz * sizeof(double)) + (actual_nnz * sizeof(uint64_t)) + (actual_nnz * sizeof(uint64_t));
      // break;
    }
    case MTX_FORMAT_ELL: {
      // uint64_t max_elements_per_row = (actual_nnz + info.rows - 1) / info.rows;
      // *size = sizeof(ell_matrix) + (info.rows * max_elements_per_row * sizeof(double)) + (info.rows * max_elements_per_row * sizeof(uint64_t));
      // break;
    }
    case MTX_FORMAT_DIA: {
      // uint64_t max_diags = info.rows < info.cols ? info.rows : info.cols;
      // *size = sizeof(dia_matrix) +  (max_diags * info.rows * sizeof(double)) +  (max_diags * sizeof(int64_t));
      // break;
    }
    case MTX_FORMAT_HYB: {
      // uint64_t avg_elements_per_row = actual_nnz / info.rows;
      // uint64_t ell_elements = avg_elements_per_row * info.rows;
      // uint64_t coo_elements = actual_nnz - ell_elements;
      
      // *size = sizeof(hyb_matrix) + (ell_elements * sizeof(double)) + (ell_elements * sizeof(uint64_t)) + 
      //         (coo_elements * sizeof(double)) + (coo_elements * sizeof(uint64_t)) + (coo_elements * sizeof(uint64_t));
      break;
    }
    case MTX_FORMAT_BCSR: {
      // const uint64_t BLOCK_SIZE = 4;
      // uint64_t block_rows = (info.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
      // uint64_t block_cols = (info.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
      // uint64_t est_blocks = (actual_nnz + (BLOCK_SIZE * BLOCK_SIZE) - 1) / (BLOCK_SIZE * BLOCK_SIZE);
      
      // *size = sizeof(bcsr_matrix) + (est_blocks * BLOCK_SIZE * BLOCK_SIZE * sizeof(double)) + (est_blocks * sizeof(uint64_t)) + ((block_rows + 1) * sizeof(uint64_t));
      // break;
    }
    default:
      return STATUS_NOT_IMPLEMENTED;
  }

  *size = (*size + 63) & ~63ULL;
  
  return STATUS_SUCCESS;
}