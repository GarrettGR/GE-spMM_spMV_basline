#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <matio.h>
#include <sys/time.h>

typedef struct {
  size_t row;
  size_t col;
  double value;
} NonZeroElement;

typedef struct {
  size_t rows;
  size_t cols;
  size_t nnz;
  NonZeroElement* elements;
} COOMatrix;

typedef struct {
  double load_time;
  double compute_time;
  double memory_usage;
  size_t flop_count;
} Performance;

COOMatrix* create_coo_matrix(size_t rows, size_t cols, size_t nnz) {
  COOMatrix* matrix = (COOMatrix*)malloc(sizeof(COOMatrix));
  if (!matrix) return NULL;

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  matrix->elements = (NonZeroElement*)aligned_alloc(64,
    nnz * sizeof(NonZeroElement));

  if (!matrix->elements) {
    free(matrix);
    return NULL;
  }

  return matrix;
}

int compare_elements(const void* a, const void* b) {
  const NonZeroElement* elem_a = (const NonZeroElement*)a;
  const NonZeroElement* elem_b = (const NonZeroElement*)b;

  if (elem_a->row != elem_b->row)
    return (elem_a->row > elem_b->row) - (elem_a->row < elem_b->row);
  return (elem_a->col > elem_b->col) - (elem_a->col < elem_b->col);
}

COOMatrix* load_matrix_from_mat(const char* filename) { //TODO: Implement this function
  struct timeval start, end;
  gettimeofday(&start, NULL);

  mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
  if (!matfp) {
    fprintf(stderr, "Error opening MAT file: %s\n", filename);
    return NULL;
  }

  matvar_t *matvar = Mat_VarRead(matfp, NULL);
  if (!matvar) {
    fprintf(stderr, "Error reading matrix from MAT file\n");
    Mat_Close(matfp);
    return NULL;
  }

  // Convert from MAT format to COO
  // Implementation depends on input format
  // ...

  Mat_VarFree(matvar);
  Mat_Close(matfp);

  gettimeofday(&end, NULL);
  double load_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
  printf("Matrix loaded in %.3f seconds\n", load_time);

  return NULL;
}

COOMatrix* multiply_coo_matrices(const COOMatrix* A, const COOMatrix* B) {
  if (A->cols != B->rows) {
    fprintf(stderr, "Matrix dimensions incompatible for multiplication\n");
    return NULL;
  }

  size_t max_possible_nnz = A->nnz * B->nnz;
  size_t hash_size = max_possible_nnz < 1000000 ? max_possible_nnz : 1000000;

  typedef struct {
    size_t row;
    size_t col;
    double value;
    size_t next;
  } HashEntry;

  size_t* hash_heads = (size_t*)calloc(hash_size, sizeof(size_t));
  HashEntry* hash_entries = (HashEntry*)calloc(max_possible_nnz, sizeof(HashEntry));
  size_t entry_count = 0;

  #define HASH(r, c) (((r) * 2654435769ull + (c)) % hash_size)

  #pragma omp parallel
  {
    HashEntry* local_entries = (HashEntry*)calloc(max_possible_nnz / omp_get_num_threads(), sizeof(HashEntry));
    size_t local_count = 0;

    #pragma omp for schedule(dynamic, 1000)
    for (size_t i = 0; i < A->nnz; i++) {
      size_t a_row = A->elements[i].row;
      size_t a_col = A->elements[i].col;
      double a_val = A->elements[i].value;

      for (size_t j = 0; j < B->nnz; j++) {
        if (B->elements[j].row == a_col) {
          size_t b_col = B->elements[j].col;
          double result = a_val * B->elements[j].value;

          HashEntry* entry = &local_entries[local_count++];
          entry->row = a_row;
          entry->col = b_col;
          entry->value = result;
        }
      }
    }

    #pragma omp critical
    {
      for (size_t i = 0; i < local_count; i++) {
        HashEntry* local = &local_entries[i];
        size_t hash = HASH(local->row, local->col);
        size_t pos = hash_heads[hash];

        while (pos > 0) {
          if (hash_entries[pos-1].row == local->row &&
            hash_entries[pos-1].col == local->col) {
              hash_entries[pos-1].value += local->value;
              break;
            }
          pos = hash_entries[pos-1].next;
        }

        if (pos == 0) {
          hash_entries[entry_count] = *local;
          hash_entries[entry_count].next = hash_heads[hash];
          hash_heads[hash] = entry_count + 1;
          entry_count++;
        }
      }
    }
    free(local_entries);
  }

  COOMatrix* C = create_coo_matrix(A->rows, B->cols, entry_count);
  if (!C) {
    free(hash_heads);
    free(hash_entries);
    return NULL;
  }

  size_t idx = 0;
  for (size_t i = 0; i < entry_count; i++) {
    if (hash_entries[i].value != 0.0) {
      C->elements[idx].row = hash_entries[i].row;
      C->elements[idx].col = hash_entries[i].col;
      C->elements[idx].value = hash_entries[i].value;
      idx++;
    }
  }
  C->nnz = idx;

  qsort(C->elements, C->nnz, sizeof(NonZeroElement), compare_elements);

  free(hash_heads);
  free(hash_entries);
  return C;
}

Performance benchmark_multiplication(const COOMatrix* A, const COOMatrix* B) {
  Performance perf = {0};
  struct timeval start, end;

  COOMatrix* warmup = multiply_coo_matrices(A, B);
  free_coo_matrix(warmup);

  gettimeofday(&start, NULL);
  COOMatrix* C = multiply_coo_matrices(A, B);
  gettimeofday(&end, NULL);

  perf.compute_time = (end.tv_sec - start.tv_sec) +  (end.tv_usec - start.tv_usec) / 1e6;
  perf.memory_usage = (A->nnz + B->nnz + C->nnz) * sizeof(NonZeroElement);
  perf.flop_count = 2 * C->nnz;

  free_coo_matrix(C);
  return perf;
}

void free_coo_matrix(COOMatrix* matrix) {
  if (matrix) {
    free(matrix->elements);
    free(matrix);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix_A.mat> <matrix_B.mat>\n", argv[0]);
    return 1;
  }

  COOMatrix* A = load_matrix_from_mat(argv[1]);
  COOMatrix* B = load_matrix_from_mat(argv[2]);

  if (!A || !B) {
    fprintf(stderr, "Error loading matrices\n");
    return 1;
  }

  Performance perf = benchmark_multiplication(A, B);

  printf("\nPerformance Results:\n");
  printf("Computation Time: %.3f seconds\n", perf.compute_time);
  printf("Memory Usage: %.2f MB\n", perf.memory_usage / (1024.0 * 1024.0));
  printf("FLOP Count: %zu\n", perf.flop_count);
  printf("FLOPS: %.2e\n", perf.flop_count / perf.compute_time);
  printf("Matrix density: %.2e%%\n", 100.0 * A->nnz / (A->rows * A->cols));

  free_coo_matrix(A);
  free_coo_matrix(B);

  return 0;
}
