#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(func) { \
  cudaError_t status = (func); \
  if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status)); \
    exit(1); \
  } \
}

#define CHECK_CUSPARSE(func) { \
  cusparseStatus_t status = (func); \
  if (status != CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr, "CUSPARSE Error: %d\n", status); \
    exit(1); \
  } \
}

void printProfiling(int m, int n, int nnzA, int nnzB, int nnzC, double initTime, double multTime, double totalTime, size_t memUsage) {
  double flops = 2.0 * nnzC;
  double gflops = flops / multTime / 1e9;
  double densityA = 100.0 * nnzA / (m * n);
  double densityB = 100.0 * nnzB / (m * n);
  double densityC = 100.0 * nnzC / (m * n);

  printf("==================\n");
  printf("Profiling Results:\n");
  printf("==================\n");
  printf("Initialization Time: %.4f seconds\n", initTime);
  printf("Multiplication Time: %.4f seconds\n", multTime);
  printf("Total Time: %.4f seconds\n", totalTime);
  printf("Peak Memory Usage: %.2f MB\n", memUsage / (1024.0 * 1024.0));
  printf("Total FLOPS: %.2e\n", flops);
  printf("Performance: %.2f GFLOPS\n", gflops);

  printf("\n");

  printf("==================\n");
  printf("Matrix Statistics:\n");
  printf("==================\n");
  printf("Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)\n", m, n, nnzA, densityA);
  printf("Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)\n", n, n, nnzB, densityB);
  printf("Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)\n", m, n, nnzC, densityC);
}

int main() {
  int m = 10000;
  int n = 10000;
  float sparsity = 0.001;
  int nnzA = sparsity * m * n;
  int nnzB = sparsity * n * n;

  // Initialize matrix data
  int *csrRowPtrA, *csrColIndA;
  float *csrValA;
  int *csrRowPtrB, *csrColIndB;
  float *csrValB;
  int *csrRowPtrC;
  int *csrColIndC;
  float *csrValC;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Allocate host memory
  csrRowPtrA = (int*)malloc((m + 1) * sizeof(int));
  csrColIndA = (int*)malloc(nnzA * sizeof(int));
  csrValA = (float*)malloc(nnzA * sizeof(float));

  csrRowPtrB = (int*)malloc((n + 1) * sizeof(int));
  csrColIndB = (int*)malloc(nnzB * sizeof(int));
  csrValB = (float*)malloc(nnzB * sizeof(float));

  // Generate random sparse matrices
  srand(0);
  for (int i = 0; i <= m; i++) {
    csrRowPtrA[i] = i * nnzA / m;
  }
  for (int i = 0; i < nnzA; i++) {
    csrColIndA[i] = rand() % n;
    csrValA[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i <= n; i++) {
    csrRowPtrB[i] = i * nnzB / n;
  }
  for (int i = 0; i < nnzB; i++) {
    csrColIndB[i] = rand() % n;
    csrValB[i] = (float)rand() / RAND_MAX;
  }

  CHECK_CUDA(cudaEventRecord(start));

  // Allocate device memory
  int *d_csrRowPtrA, *d_csrColIndA;
  float *d_csrValA;
  int *d_csrRowPtrB, *d_csrColIndB;
  float *d_csrValB;
  int *d_csrRowPtrC;
  int *d_csrColIndC;
  float *d_csrValC;

  CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrA, (m + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrColIndA, nnzA * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrValA, nnzA * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrB, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrColIndB, nnzB * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrValB, nnzB * sizeof(float)));

  // Copy data from host to device
  CHECK_CUDA(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrColIndA, csrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrValA, csrValA, nnzA * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrRowPtrB, csrRowPtrB, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrColIndB, csrColIndB, nnzB * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrValB, csrValB, nnzB * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float initTime;
  CHECK_CUDA(cudaEventElapsedTime(&initTime, start, stop));
  initTime /= 1000.0;  // Convert to seconds

  CHECK_CUDA(cudaEventRecord(start));

  // Initialize CuSPARSE
  cusparseHandle_t handle;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Create matrix descriptors
  cusparseMatDescr_t descr_A, descr_B, descr_C;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_A));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_B));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_C));

  // Perform matrix multiplication
  int nnzC;
  CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  CHECK_CUSPARSE(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, n, descr_A, nnzA, d_csrRowPtrA, d_csrColIndA,
        descr_B, nnzB, d_csrRowPtrB, d_csrColIndB,
        descr_C, d_csrRowPtrC, &nnzC));

  CHECK_CUDA(cudaMalloc((void**)&d_csrColIndC, nnzC * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrValC, nnzC * sizeof(float)));

  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUSPARSE(cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, n, &alpha, descr_A, nnzA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        descr_B, nnzB, d_csrValB, d_csrRowPtrB, d_csrColIndB,
        &beta, descr_C, d_csrValC, d_csrRowPtrC, d_csrColIndC));

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float multTime;
  CHECK_CUDA(cudaEventElapsedTime(&multTime, start, stop));
  multTime /= 1000.0;  // Convert to seconds

  float totalTime = initTime + multTime;

  // Copy result from device to host
  csrRowPtrC = (int*)malloc((m + 1) * sizeof(int));
  csrColIndC = (int*)malloc(nnzC * sizeof(int));
  csrValC = (float*)malloc(nnzC * sizeof(float));

  CHECK_CUDA(cudaMemcpy(csrRowPtrC, d_csrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(csrColIndC, d_csrColIndC, nnzC * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(csrValC, d_csrValC, nnzC * sizeof(float), cudaMemcpyDeviceToHost));

  // Print profiling results
  size_t memUsage = ((m + 1) + nnzA + nnzA) * sizeof(int) + nnzA * sizeof(float) +
    ((n + 1) + nnzB + nnzB) * sizeof(int) + nnzB * sizeof(float) +
    ((m + 1) + nnzC + nnzC) * sizeof(int) + nnzC * sizeof(float);

  printProfiling(m, n, nnzA, nnzB, nnzC, initTime, multTime, totalTime, memUsage);

  // Clean up
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_A));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_B));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_C));
  CHECK_CUSPARSE(cusparseDestroy(handle));

  CHECK_CUDA(cudaFree(d_csrRowPtrA));
  CHECK_CUDA(cudaFree(d_csrColIndA));
  CHECK_CUDA(cudaFree(d_csrValA));
  CHECK_CUDA(cudaFree(d_csrRowPtrB));
  CHECK_CUDA(cudaFree(d_csrColIndB));
  CHECK_CUDA(cudaFree(d_csrValB));
  CHECK_CUDA(cudaFree(d_csrRowPtrC));
  CHECK_CUDA(cudaFree(d_csrColIndC));
  CHECK_CUDA(cudaFree(d_csrValC));

  free(csrRowPtrA);
  free(csrColIndA);
  free(csrValA);
  free(csrRowPtrB);
  free(csrColIndB);
  free(csrValB);
  free(csrRowPtrC);
  free(csrColIndC);
  free(csrValC);

  return 0;
}
