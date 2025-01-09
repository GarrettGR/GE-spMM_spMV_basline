#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(func) { \
  cudaError_t status = (func); \
  if (status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status)); \
    exit(1); \
  } \
}

void printProfiling(int m, int n, int nnzA, int nnzB, int nnzC, double initTime, double multTime, double totalTime, size_t memUsage) {
  double flops = 2.0 * nnzC;
  double gflops = flops / multTime / 1e9;
  double densityA = 100.0 * nnzA / (m * n);
  double densityB = 100.0 * nnzB / (n * n);
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

__global__ void csrSpGEMM(int *csrRowPtrA, int *csrColIndA, float *csrValA, int *csrRowPtrB, int *csrColIndB, float *csrValB,
                          int *csrRowPtrC, int *csrColIndC, float *csrValC, int m, int n, int nnzA, int nnzB) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    int rowStart = csrRowPtrA[row];
    int rowEnd = csrRowPtrA[row + 1];
    csrRowPtrC[row] = rowStart;

    for (int i = rowStart; i < rowEnd; i++) {
      int colA = csrColIndA[i];
      float valA = csrValA[i];

      int colStart = csrRowPtrB[colA];
      int colEnd = csrRowPtrB[colA + 1];

      for (int j = colStart; j < colEnd; j++) {
        int colB = csrColIndB[j];
        float valB = csrValB[j];

        float valC = valA * valB;
        int pos = csrRowPtrC[row];
        csrColIndC[pos] = colB;
        csrValC[pos] = valC;
        csrRowPtrC[row]++;
      }
    }
  }
}

int main() {
  int m = 10000;
  int n = 10000;
  float sparsity = 0.001;
  int nnzA = sparsity * m * n;
  int nnzB = sparsity * n * n;

  int *csrRowPtrA, *csrColIndA;
  float *csrValA;
  int *csrRowPtrB, *csrColIndB;
  float *csrValB;
  int *csrRowPtrC, *csrColIndC;
  float *csrValC;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  csrRowPtrA = (int*)malloc((m + 1) * sizeof(int));
  csrColIndA = (int*)malloc(nnzA * sizeof(int));
  csrValA = (float*)malloc(nnzA * sizeof(float));

  csrRowPtrB = (int*)malloc((n + 1) * sizeof(int));
  csrColIndB = (int*)malloc(nnzB * sizeof(int));
  csrValB = (float*)malloc(nnzB * sizeof(float));

  csrRowPtrC = (int*)malloc((m + 1) * sizeof(int));
  csrColIndC = (int*)malloc(nnzA * nnzB * sizeof(int));
  csrValC = (float*)malloc(nnzA * nnzB * sizeof(float));

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

  int *d_csrRowPtrA, *d_csrColIndA;
  float *d_csrValA;
  int *d_csrRowPtrB, *d_csrColIndB;
  float *d_csrValB;
  int *d_csrRowPtrC, *d_csrColIndC;
  float *d_csrValC;

  CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrA, (m + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrColIndA, nnzA * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrValA, nnzA * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrB, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrColIndB, nnzB * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrValB, nnzB * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrC, (m + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrColIndC, nnzA * nnzB * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_csrValC, nnzA * nnzB * sizeof(float)));

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
  initTime /= 1000.0;

  CHECK_CUDA(cudaEventRecord(start));

  int threadsPerBlock = 256;
  int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
  csrSpGEMM<<<blocksPerGrid, threadsPerBlock>>>(d_csrRowPtrA, d_csrColIndA, d_csrValA, d_csrRowPtrB, d_csrColIndB, d_csrValB,
                                                d_csrRowPtrC, d_csrColIndC, d_csrValC, m, n, nnzA, nnzB);

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float multTime;
  CHECK_CUDA(cudaEventElapsedTime(&multTime, start, stop));
  multTime /= 1000.0;

  float totalTime = initTime + multTime;

  CHECK_CUDA(cudaMemcpy(csrRowPtrC, d_csrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(csrColIndC, d_csrColIndC, nnzA * nnzB * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(csrValC, d_csrValC, nnzA * nnzB * sizeof(float), cudaMemcpyDeviceToHost));

  int nnzC = 0;
  for (int i = 0; i < m; i++) {
    nnzC += csrRowPtrC[i + 1] - csrRowPtrC[i];
  }

  size_t memUsage = ((m + 1) + nnzA + nnzA) * sizeof(int) + nnzA * sizeof(float) + ((n + 1) + nnzB + nnzB) * sizeof(int) + nnzB * sizeof(float) +
                    ((m + 1) + nnzA * nnzB + nnzA * nnzB) * sizeof(int) + nnzA * nnzB * sizeof(float);

  printProfiling(m, n, nnzA, nnzB, nnzC, initTime, multTime, totalTime, memUsage);

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
