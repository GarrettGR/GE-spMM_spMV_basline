if(DEFINED OPENMP_CONFIG_INCLUDED)
  return()
endif()
set(OPENMP_CONFIG_INCLUDED TRUE)

if(APPLE) # (?) Should fix issues on MacOS / Homebrew GCC compiler
  execute_process(
    COMMAND ${CMAKE_C_COMPILER} -dumpversion
    OUTPUT_VARIABLE GCC_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  find_library(GOMP_LIB
    NAMES libgomp.dylib
    PATHS /opt/homebrew/lib/gcc/${GCC_VERSION} #TODO: Make this more flexible (work for other gcc versions or llvm/clang)
    DOC "Path to OpenMP library"
  )

  if(GOMP_LIB)
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_C_FOUND TRUE)
    set(OpenMP_CXX_FOUND TRUE)
    set(OPENMP_FOUND TRUE)

    message(STATUS "Found OpenMP: ${GOMP_LIB}")

    if(NOT TARGET OpenMP::OpenMP)
      add_library(OpenMP::OpenMP INTERFACE IMPORTED)
      set_target_properties(OpenMP::OpenMP PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-fopenmp"
        INTERFACE_LINK_LIBRARIES "${GOMP_LIB}")
    endif()
  else()
    set(OPENMP_FOUND FALSE)
    message(STATUS "OpenMP not found - OpenMP implementations will be disabled")
  endif()
else()
  find_package(OpenMP QUIET)
  if(OpenMP_C_FOUND)
    set(OPENMP_FOUND TRUE)
    message(STATUS "Found OpenMP: ${OpenMP_C_VERSION}")
    if(NOT TARGET OpenMP::OpenMP)
      add_library(OpenMP::OpenMP INTERFACE IMPORTED)
      set_target_properties(OpenMP::OpenMP PROPERTIES
        INTERFACE_COMPILE_OPTIONS "${OpenMP_C_FLAGS}"
        INTERFACE_LINK_LIBRARIES "${OpenMP_C_FLAGS}")
    endif()
  else()
    set(OPENMP_FOUND FALSE)
    message(STATUS "OpenMP not found - OpenMP implementations will be disabled")
  endif()
endif()