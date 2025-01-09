#!/bin/bash

rm -rf ./build/
mkdir ./build/

export CXXFLAGS="-O3 -march=native -mtune=native -DCMAKE_BUILD_TYPE=Debug"

if [ $(uname -s) == "Linux" ]; then
  export CC="gcc" # -D CMAKE_C_COMPILER=gcc
  export CXX="g++" # -D CMAKE_CXX_COMPILER=g++-14
  cmake -S . --build ./build/ --target all -- -j 16
  cmake -S . --build ./build/ --target test -- -j 16
  # cmake -S . --build ./build/ --target install -- -j 16
else if [ $(uname -s) == "Darwin" ]; then
  export CC="gcc-14" # -D CMAKE_C_COMPILER=gcc-14
  export CXX="g++-14" # -D CMAKE_CXX_COMPILER=g++-14
  cmake -S . --build ./build/ --target all -- -j 8
  cmake -S . --build ./build/ --target test -- -j 8
  # cmake -S . --build ./build/ --target install -- -j 8
fi