#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import subprocess
import os
import time
from pathlib import Path

class MatrixGenerator:
  """Generates and manages test matrices in various patterns."""

  def __init__(self, base_path="test_matrices"):
    """Initialize the generator with a base path for storing matrices."""
    self.base_path = Path(base_path)
    self.base_path.mkdir(exist_ok=True)

  def generate_diagonal_pattern(self, size, density=0.1, diagonals=5):
    """Generate a sparse matrix with diagonal pattern."""
    matrix = sp.rand(size, size, density=density/diagonals, format='csr')
    for i in range(1, diagonals):
      offset = int(size/diagonals) * i
      diag = sp.rand(size-offset, size-offset, density=density/diagonals,
        format='csr')
      matrix += sp.block_diag([diag, sp.csr_matrix((offset, offset))])
      return matrix

  def generate_block_pattern(self, size, block_size=32, density=0.1):
    """Generate a sparse matrix with dense blocks."""
    blocks_per_dim = size // block_size
    matrix = sp.csr_matrix((size, size))

    for i in range(blocks_per_dim):
      for j in range(blocks_per_dim):
        if np.random.random() < density:
          block = np.random.randn(block_size, block_size)
          matrix[i*block_size:(i+1)*block_size,
            j*block_size:(j+1)*block_size] = block
          return matrix

    def generate_random_pattern(self, size, density=0.1):
      """Generate a random sparse matrix."""
      return sp.random(size, size, density=density, format='csr')

    def generate_test_suite(self, sizes=[1000, 5000], densities=[0.01, 0.1]):
      """Generate a complete test suite of matrices."""
      patterns = {
        'diagonal': self.generate_diagonal_pattern,
        'block': self.generate_block_pattern,
        'random': self.generate_random_pattern
      }

      generated_files = []
      for pattern_name, generator in patterns.items():
        for size in sizes:
          for density in densities:
            name = f"{pattern_name}_s{size}_d{int(density*100)}"
            matrix = generator(size, density)
            filename = self.save_matrix(matrix, name)
            generated_files.append(filename)
            print(f"Generated {filename}")

            vector = np.random.randn(size)
            vec_filename = self.save_matrix(vector, f"{name}_vector")
            generated_files.append(vec_filename)

      return generated_files

    def save_matrix(self, matrix, name):
      """Save a matrix in MAT format with additional metadata."""
      filename = self.base_path / f"{name}.mat"

      if sp.issparse(matrix):
        matrix = matrix.tocsr()
        metadata = {
          'rows': matrix.shape[0],
          'cols': matrix.shape[1],
          'nnz': matrix.nnz,
          'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
          'type': 'sparse'
        }
      else:
        metadata = {
          'rows': matrix.shape[0],
          'type': 'dense'
        }

      sio.savemat(filename, {
        'matrix': matrix,
        'metadata': metadata
      })

      return filename

class TestRunner:
  """Manages the execution and verification of sparse matrix programs."""

  def __init__(self, bin_path="bin"):
    """Initialize the test runner with path to executables."""
    self.bin_path = Path(bin_path)

  def run_all_tests(self, matrix_files):
      """Run all compiled programs against test matrices."""
      results = {}

      test_cases = self._group_test_files(matrix_files)

      formats = ['csr', 'pcsr', 'bcsr', 'coo', 'ell', 'hybrid']
      for fmt in formats:
        results[fmt] = {}

        spmv_exec = self.bin_path / f"{fmt}_spmv"
        if spmv_exec.exists():
          for case_name, (matrix, vector) in test_cases.items():
            result = self._run_spmv_test(spmv_exec, matrix, vector)
            results[fmt][f"spmv_{case_name}"] = result
        spmm_exec = self.bin_path / f"{fmt}_matmat"
        if spmm_exec.exists():
          for case_name, (matrix, _) in test_cases.items():
            result = self._run_spmm_test(spmm_exec, matrix, matrix)
            results[fmt][f"spmm_{case_name}"] = result

      self._report_results(results)
      return results

  def _group_test_files(self, matrix_files):
    """Group matrix files into test cases with their corresponding vectors."""
    test_cases = {}
    for file in matrix_files:
      if '_vector' not in file.name:
        base_name = file.stem.split('_s')[0]
        vector_file = next(f for f in matrix_files if f.stem.endswith('_vector') and base_name in f.stem)
        test_cases[base_name] = (file, vector_file)
    return test_cases

  def _run_spmv_test(self, executable, matrix_file, vector_file):
    """Run and time a single SpMV test."""
    try:
      start_time = time.time()
      result = subprocess.run([str(executable), str(matrix_file), str(vector_file)], capture_output=True, text=True, timeout=300)
      end_time = time.time()

      return {
        'success': result.returncode == 0,
        'time': end_time - start_time,
        'output': result.stdout,
        'error': result.stderr if result.returncode != 0 else None
      }
    except subprocess.TimeoutExpired:
      return {
        'success': False,
        'error': 'Timeout after 300 seconds'
      }

  def _run_spmm_test(self, executable, matrix_file1, matrix_file2):
    """Run and time a single SpMM test."""
    try:
      start_time = time.time()
      result = subprocess.run([str(executable), str(matrix_file1), str(matrix_file2)], capture_output=True, text=True, timeout=300)
      end_time = time.time()

      return {
        'success': result.returncode == 0,
        'time': end_time - start_time,
        'output': result.stdout,
        'error': result.stderr if result.returncode != 0 else None
      }
    except subprocess.TimeoutExpired:
      return {
        'success': False,
        'error': 'Timeout after 300 seconds'
      }

  def _report_results(self, results):
    """Generate a detailed report of test results."""
    print("\nTest Results Summary")
    print("===================")

    for fmt, tests in results.items():
      print(f"\n{fmt.upper()} Format:")
      for test_name, result in tests.items():
        status = "✓" if result['success'] else "✗"
        time_str = f"{result['time']:.2f}s" if 'time' in result else "N/A"
        print(f"  {status} {test_name}: {time_str}")
        if not result['success']:
          print(f"    Error: {result['error']}")

def main():
  """Main function to generate matrices and run tests."""
  generator = MatrixGenerator()
  print("Generating test matrices...")
  matrix_files = generator.generate_test_suite()

  print("\nRunning tests...")
  runner = TestRunner()
  results = runner.run_all_tests(matrix_files)

if __name__ == "__main__":
  main()
