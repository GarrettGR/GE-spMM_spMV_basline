cmake_minimum_required(VERSION 3.21)
project(sparse_matrix_ops LANGUAGES C CXX)

if(CMAKE_C_COMPILER)
  set(ENV{CC} ${CMAKE_C_COMPILER})
endif()
if(CMAKE_CXX_COMPILER)
  set(ENV{CXX} ${CMAKE_CXX_COMPILER})
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(PETSC PETSc REQUIRED)
if(NOT PETSC_FOUND)
  message(FATAL_ERROR "PETSc not found!")
endif()
# message(STATUS "PETSc include dirs: ${PETSC_INCLUDE_DIRS}")
# message(STATUS "PETSc library dirs: ${PETSC_LIBRARY_DIRS}")
# message(STATUS "PETSc libraries: ${PETSC_LIBRARIES}")
message(STATUS "PETSc version: ${PETSC_VERSION}")

find_package(MPI REQUIRED)
# pkg_check_modules(MPI REQUIRED IMPORTED_TARGET mpi)
if(NOT MPI_FOUND)
  message(FATAL_ERROR "MPI not found!")
endif()
# message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")
# message(STATUS "MPI libraries: ${MPI_LIBRARIES}")
message(STATUS "MPI version: ${MPI_VERSION}") #FIXME: this doesn't work ???

function(target_link_petsc target)
  set(PETSC_ROOT "/opt/homebrew/opt/petsc")
  
  target_include_directories(${target} 
    PRIVATE 
    ${PETSC_ROOT}/include
    ${PETSC_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
  )
  target_link_directories(${target} 
    PRIVATE 
    ${PETSC_ROOT}/lib
    ${PETSC_LIBRARY_DIRS}
  )
  target_link_libraries(${target} 
    PRIVATE 
    ${PETSC_LIBRARIES}
    MPI::MPI_C
  )
  target_compile_options(${target} 
    PRIVATE 
    ${PETSC_CFLAGS_OTHER}
  )
endfunction()

if(CMAKE_C_COMPILER)
  set(ENV{CC} ${CMAKE_C_COMPILER})
endif()
if(CMAKE_CXX_COMPILER)
  set(ENV{CXX} ${CMAKE_CXX_COMPILER})
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(FindOpenMP)
include(FindCUDA)
# pkg_check_modules(OPENMP REQUIRED omp) # this can't find openmp on macos (homebrew)
# pkg_check_modules(CUDA REQUIRED cuda) # is this right? 

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(CUDA_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 23)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g -O0)
else()
  add_compile_options(-O3)
endif()

add_library(profiler
  src/utils/profiler.c
)
target_include_directories(profiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(validator 
  src/utils/validator.c
)
target_include_directories(validator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_petsc(validator)

add_library(csr_format
  src/formats/csr.c
)
target_include_directories(csr_format PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(csr_format PRIVATE profiler)

add_library(csr_spmm_cpu
  src/algorithms/spMM/cpu/sequential/csr_mult.c
)
target_link_libraries(csr_spmm_cpu PUBLIC csr_format)

if(OPENMP_FOUND)
  message(STATUS "Building OpenMP implementations")
  add_library(csr_spmm_openmp
    src/algorithms/spMM/cpu/openmp/csr_mult.c
  )
  target_link_libraries(csr_spmm_openmp
    PUBLIC 
      csr_format
    PRIVATE 
      OpenMP::OpenMP
  )
endif()

if(CUDA_FOUND)
  message(STATUS "Building CUDA implementations")
#   add_library(csr_spmm_cuda)
#   cuda_add_sources(csr_spmm_cuda
#     src/algorithms/spMM/gpu/CUDA/csr_spmm.cu
#   )
#   target_link_libraries(csr_spmm_cuda
#     PUBLIC csr_format
#     PRIVATE CUDA::CUDA
#   )
endif()

add_library(csr_spmv_cpu
  src/algorithms/spMV/cpu/sequential/csr_mult.c
)
target_link_libraries(csr_spmv_cpu PUBLIC csr_format)

if(OPENMP_FOUND)
  add_library(csr_spmv_openmp
    src/algorithms/spMV/cpu/openmp/csr_mult.c
  )
  target_link_libraries(csr_spmv_openmp
    PUBLIC 
      csr_format
    PRIVATE 
      OpenMP::OpenMP
  )
endif()

if(CUDA_FOUND)
#   add_library(csr_spmv_cuda)
#   cuda_add_sources(csr_spmv_cuda
#     src/algorithms/spMV/gpu/CUDA/csr_spmv.cu
#   )
#   target_link_libraries(csr_spmv_cuda
#     PUBLIC csr_format
#     PRIVATE CUDA::CUDA
#   )
endif()

# add_executable(spmm_bench benchmarks/spmm_bench.c)
# target_link_libraries(spmm_bench
#   PRIVATE
#     csr_format
#     csr_spmm_cpu
# )

if(OPENMP_FOUND)
  # target_link_libraries(spmm_bench PRIVATE csr_spmm_openmp)
endif()

if(CUDA_FOUND)
#   target_link_libraries(spmm_bench PRIVATE csr_spmm_cuda)
endif()

enable_testing()

# find_package(SuiteSparse REQUIRED)

add_library(test_utils
    tests/test_utils.c
)
target_include_directories(test_utils
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
#   PRIVATE
#     ${PETSC_INCLUDE_DIRS}
)
target_link_libraries(test_utils
#   PUBLIC
#     SuiteSparse::CSPARSE
  PRIVATE
    csr_format
#     ${PETSC_LIBRARIES}
)

# ------ Format Tests -------

# CSR
add_executable(csr_format_tests tests/formats/test_csr.c)
target_link_libraries(csr_format_tests
  PRIVATE
    csr_format
    test_utils
)
# add_test(NAME CSRFormatTests COMMAND csr_format_tests all)
add_test(NAME csr_format_create COMMAND csr_format_tests create)
add_test(NAME csr_format_arrays COMMAND csr_format_tests create_from_arrays)
add_test(NAME csr_format_copy COMMAND csr_format_tests copy)
add_test(NAME csr_format_get COMMAND csr_format_tests get_value)
add_test(NAME csr_format_valid COMMAND csr_format_tests validation)
add_test(NAME csr_format_mem COMMAND csr_format_tests memory_management)
add_test(NAME csr_format_large COMMAND csr_format_tests large_matrix)
add_test(NAME csr_fromat_edge COMMAND csr_format_tests edge_cases)


# ------- Algo Tests --------

# CSR SpMV
add_executable(csr_spmv_tests tests/algorithms/csr_spmv_tests.c)
target_link_libraries(csr_spmv_tests
  PRIVATE
    csr_format
    test_utils
    csr_spmv_cpu
    csr_spmv_openmp
)
# add_test(NAME CSRSpMVTests COMMAND csr_spmv_tests all)
add_test(NAME csr_spmv_null COMMAND csr_spmv_tests null_input)
add_test(NAME csr_spmv_ident COMMAND csr_spmv_tests identity)
add_test(NAME csr_spmv_small COMMAND csr_spmv_tests random_small)
add_test(NAME csr_spmv_large COMMAND csr_spmv_tests random_large)
# add_test(NAME csr_spmv_perf COMMAND csr_spmv_tests performance)

# CSR SpMM
add_executable(csr_spmm_tests tests/algorithms/csr_spmm_tests.c)
target_link_libraries(csr_spmm_tests
  PRIVATE
    csr_format
    test_utils
    csr_spmm_cpu
    csr_spmm_openmp
)
# add_test(NAME CSRSpMMTests COMMAND csr_spmm_tests all)
add_test(NAME csr_spmm_null COMMAND csr_spmm_tests null_input)
add_test(NAME csr_spmm_ident COMMAND csr_spmm_tests identity)
add_test(NAME csr_spmm_small COMMAND csr_spmm_tests random_small)
add_test(NAME csr_spmm_large COMMAND csr_spmm_tests random_large)
add_test(NAME csr_spmm_chain COMMAND csr_spmm_tests chain)
add_test(NAME csr_spmm_bdiag COMMAND csr_spmm_tests block_diagonal)
add_test(NAME csr_spmm_dense COMMAND csr_spmm_tests dense)
# add_test(NAME csr_spmm_perf COMMAND csr_spmm_tests performance)

# ----- Profile Tests -------

# ...
# ...
# ...

# --- Validation Tests ------

add_executable(validate_implementations tests/validate_implementations.c)
target_link_libraries(validate_implementations
  PRIVATE
    validator
    csr_format
    csr_spmm_cpu
    csr_spmm_openmp # TODO: make this optional / dynamically added
    csr_spmv_cpu
    csr_spmv_openmp
    test_utils
)
target_link_petsc(validate_implementations)
# add_test(NAME ValidateImplementations COMMAND validate_implementations all)
add_test(NAME validate_spmm_random COMMAND validate_implementations spmm_random)
add_test(NAME validate_spmm_block COMMAND validate_implementations spmm_block)
add_test(NAME validate_spmm_extreme COMMAND validate_implementations spmm_extreme)
add_test(NAME validate_spmm_chain COMMAND validate_implementations spmm_chain)
add_test(NAME validate_spmv_random COMMAND validate_implementations spmv_random)
add_test(NAME validate_spmv_dense COMMAND validate_implementations spmv_dense)
add_test(NAME validate_spmv_extreme COMMAND validate_implementations spmv_extreme)

set_tests_properties(
  validate_spmm_random
  validate_spmm_block
  validate_spmm_extreme
  validate_spmv_random
  PROPERTIES TIMEOUT 120
)

install(TARGETS
  csr_format
  csr_spmm_cpu
  csr_spmv_cpu
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

if(OPENMP_FOUND)
#   install(TARGETS
#     csr_spmm_openmp
#     csr_spmv_openmp
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
#   )
endif()

if(CUDA_FOUND)
#   install(TARGETS
#     csr_spmm_cuda
#     csr_spmv_cuda
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
#   )
endif()

install(DIRECTORY include/ DESTINATION include)

set(CMAKE_CTEST_ARGUMENTS "--output-on-failure --progress")
