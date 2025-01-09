# How to use, and the format of, the orchestrator and profilers

## Orchestrator Use

## Detailed Profiler:
### Use:
By setting an environment variable, namely `PROFILE_TARGET`, you can control which part of the profiler is run, it can be set to: `basic`, `memory`, or `energy`
> NOTE: **Memory Performace** profiling requires PAPI (Performance API) to be installed, and **Energy Metrics** requires the system to be an Intel platform and running Linux
### Format:
```
Running with %d threads

==================
Profiling Results:
==================
Multiplication time: %.6f seconds
Memory usage: %.2f MB
Memory Bandwidth: %.2f GB/s
Cache Misses: %ld
Total FLOPS: %.2e
Performance: %.2f GFLOPS

==================
Memory Performance:
==================
L1 Cache Hit Rate: %.2f%%
L2 Cache Hit Rate: %.2f%%
L3 Cache Hit Rate: %.2f%%
TLB Misses: %ld
Bytes Transferred: %.2f GB

==================
Energy Metrics:
==================
Total Energy Consumed: %.2f Joules
Average Power Usage: %.2f Watts
Energy Efficiency: %.2f GFLOPS/Watt
DRAM Energy: %.2f Joules
CPU Package Energy: %.2f Joules

==================
Matrix Statistics:
==================
Initialization Time: %.6f seconds
Conversion Time: %.6f seconds
Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)
Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)
Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)

Total Time: %.6f seconds
```

## Simple Profiler
### Format:

```
Running with %d threads

==================
Profiling Results
==================

Multiplication time: %f seconds
Memory usage: %.2f MB
Memory Bandwidth: %.2f GB/s
Cache Misses: %d
Total FLOPS: %d
Performance: %.2f GFLOPS

==================
Matrix Statistics
==================

Initialization Time: %f seconds
Conversion Time: %f seconds
Input Matrix A: %d x %d, %d non-zeros (%.2f%% dense)
Input Matrix B: %d x %d, %d non-zeros (%.2f%% dense)
Result Matrix C: %d x %d, %d non-zeros (%.2f%% dense)

Total Time: %f seconds
```
