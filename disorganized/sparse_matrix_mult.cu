#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
  cudaError_t status = (func);                                                 \
  if (status != cudaSuccess) {                                                 \
    printf("CUDA API failed at line %d with error: %s (%d)\n",                 \
      __LINE__, cudaGetErrorString(status), status);                           \
    return EXIT_FAILURE;                                                       \
  }                                                                            \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
  cusparseStatus_t status = (func);                                            \
  if (status != CUSPARSE_STATUS_SUCCESS) {                                     \
    printf("CUSPARSE API failed at line %d with error: %s (%d)\n",             \
      __LINE__, cusparseGetErrorString(status), status);                       \
    return EXIT_FAILURE;                                                       \
  }                                                                            \
}

int main() {
  int num_rows_A = 1000;
  int num_cols_A = 1000;
  int num_cols_B = 1000;
  float sparsity = 0.01;

  cusparseHandle_t handle = nullptr;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  cusparseMatDescr_t descr_A = nullptr;
  cusparseMatDescr_t descr_B = nullptr;
  cusparseMatDescr_t descr_C = nullptr;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_A));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_B));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_C));
  CHECK_CUSPARSE(cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatType(descr_B, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatType(descr_C, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_B, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_C, CUSPARSE_INDEX_BASE_ZERO));

  int nnz_A = num_rows_A * num_cols_A * sparsity;
  int nnz_B = num_cols_A * num_cols_B * sparsity;

  int *h_csrRowPtr_A = (int*) malloc((num_rows_A + 1) * sizeof(int));
  int *h_csrColInd_A = (int*) malloc(nnz_A * sizeof(int));
  float *h_csrVal_A  = (float*) malloc(nnz_A * sizeof(float));

  int *h_csrRowPtr_B = (int*) malloc((num_cols_A + 1) * sizeof(int));
  int *h_csrColInd_B = (int*) malloc(nnz_B * sizeof(int));
  float *h_csrVal_B  = (float*) malloc(nnz_B * sizeof(float));

  for (int i = 0; i < nnz_A; i++) {
    h_csrColInd_A[i] = rand() % num_cols_A;
    h_csrVal_A[i] = (float) rand() / RAND_MAX;
  }
  h_csrRowPtr_A[0] = 0;
  for (int i = 1; i <= num_rows_A; i++) {
    h_csrRowPtr_A[i] = h_csrRowPtr_A[i-1] + (int)(nnz_A / num_rows_A);
  }
  h_csrRowPtr_A[num_rows_A] = nnz_A;

  for (int i = 0; i < nnz_B; i++) {
    h_csrColInd_B[i] = rand() % num_cols_B;
    h_csrVal_B[i] = (float) rand() / RAND_MAX;
  }
  h_csrRowPtr_B[0] = 0;
  for (int i = 1; i <= num_cols_A; i++) {
    h_csrRowPtr_B[i] = h_csrRowPtr_B[i-1] + (int)(nnz_B / num_cols_A);
  }
  h_csrRowPtr_B[num_cols_A] = nnz_B;

  int *d_csrRowPtr_A = nullptr;
  int *d_csrColInd_A = nullptr;
  float *d_csrVal_A = nullptr;

  int *d_csrRowPtr_B = nullptr;  
  int *d_csrColInd_B = nullptr;
  float *d_csrVal_B = nullptr;

  int *d_csrRowPtr_C = nullptr;
  int *d_csrColInd_C = nullptr;
  float *d_csrVal_C = nullptr;

  CHECK_CUDA(cudaMalloc((void**) &d_csrRowPtr_A, (num_rows_A + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**) &d_csrColInd_A, nnz_A * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**) &d_csrVal_A, nnz_A * sizeof(float)));

  CHECK_CUDA(cudaMalloc((void**) &d_csrRowPtr_B, (num_cols_A + 1) * sizeof(int)));   
  CHECK_CUDA(cudaMalloc((void**) &d_csrColInd_B, nnz_B * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**) &d_csrVal_B, nnz_B * sizeof(float)));
  
  CHECK_CUDA(cudaMemcpy(d_csrRowPtr_A, h_csrRowPtr_A, (num_rows_A + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrColInd_A, h_csrColInd_A, nnz_A * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrVal_A, h_csrVal_A, nnz_A * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_csrRowPtr_B, h_csrRowPtr_B, (num_cols_A + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrColInd_B, h_csrColInd_B, nnz_B * sizeof(int), cudaMemcpyHostToDevice)); 
  CHECK_CUDA(cudaMemcpy(d_csrVal_B, h_csrVal_B, nnz_B * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc((void**) &d_csrRowPtr_C, (num_rows_A + 1) * sizeof(int)));

  int nnz_C = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
    handle, opA, opB,
    &alpha, descr_A, descr_B, &beta, descr_C,
    CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
    &nnz_C, nullptr)
  );

  CHECK_CUDA(cudaMalloc((void**) &d_csrColInd_C, nnz_C * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**) &d_csrVal_C, nnz_C * sizeof(float)));

  CHECK_CUSPARSE(cusparseSpGEMM(
    handle, opA, opB,
    &alpha, descr_A, descr_B, &beta, descr_C,
    CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
    &nnz_C, d_csrVal_C, d_csrRowPtr_C, d_csrColInd_C)
  );

  int *h_csrRowPtr_C = (int*) malloc((num_rows_A + 1) * sizeof(int));
  int *h_csrColInd_C = (int*) malloc(nnz_C * sizeof(int));
  float *h_csrVal_C  = (float*) malloc(nnz_C * sizeof(float));

  CHECK_CUDA(cudaMemcpy(h_csrRowPtr_C, d_csrRowPtr_C, (num_rows_A + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_csrColInd_C, d_csrColInd_C, nnz_C * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_csrVal_C, d_csrVal_C, nnz_C * sizeof(float), cudaMemcpyDeviceToHost));

  printf("Number of nonzero elements in C: %d\n", nnz_C);

  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_A));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_B));  
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_C));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  
  cudaFree(d_csrRowPtr_A);
  cudaFree(d_csrColInd_A);
  cudaFree(d_csrVal_A);
  cudaFree(d_csrRowPtr_B); 
  cudaFree(d_csrColInd_B);
  cudaFree(d_csrVal_B);
  cudaFree(d_csrRowPtr_C);
  cudaFree(d_csrColInd_C);
  cudaFree(d_csrVal_C);

  free(h_csrRowPtr_A);
  free(h_csrColInd_A); 
  free(h_csrVal_A);
  free(h_csrRowPtr_B);
  free(h_csrColInd_B);
  free(h_csrVal_B);
  free(h_csrRowPtr_C);
  free(h_csrColInd_C);
  free(h_csrVal_C);

  return 0;
}
