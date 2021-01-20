# The HOPS toolbox

[![Build Status](https://travis-ci.org/modsim/hops.svg?branch=master)](https://travis-ci.org/modsim/hops)


The **H**ighly **O**ptimized **P**olytope **S**ampling toolbox is an open-source C++17
library for efficient and scalable MCMC algorithms for sampling convex-constrained spaces possibly
equipped with arbitrary target functions.

For details and benchmarks see the application note https://doi.org/10.1093/bioinformatics/btaa872.
Test data from the application note is downloadable at doi:10.26165/JUELICH-DATA/YXLFKJ.

## Documentation

See https://modsim.github.io/hops/.

## Installation

HOPS uses CMake as build system.  
See the Dockerfile for a demonstration on installing HOPS and its dependencies on Ubuntu 20.4.

### CMake options

* HOPS\_BENCHMARKS (default OFF) - Enables compilation of Benchmarks (Requires Celero). Use -DHOPS\_BENCHMARKS=ON to enable.
* HOPS\_DOCS (default ON) - Enables generation of documentation. Use -DHOPS\_DOCS=OFF to disable. (This creates the Doxygen file from which the docs have to be generated)
* HOPS\_EXAMPLES (default ON) - Enables compilation of Examples. Use -DHOPS\_EXAMPLES=OFF to disable.
* HOPS\_TESTS (default ON) - Enables compilation of unit tests. Use -DHOPS\_TESTS=OFF to disable.

When building HOPS with Tests, an internet connection is required in order to fetch Googletest (https://github.com/google/googletest).

#### Install on Linux:

```
# Create directory for out-of-source build
$ mkdir cmake-build-release
$ cd cmake-build-release
# Run cmake
$ cmake .. -DCMAKE_BUILD_TYPE=Release
# Build HOPS
$ make 
# Run Tests
$ make test
# Alternatively run tests by calling runTests
# cd tests
$ ./runTests
$ cd ..
# Install
$ sudo make install
```

#### Install on Windows 10:

Use an IDE (e.g. CLion) to parse the project and its CMakeLists.txt.



## Examples
See the examples directory for demonstrations on how to use the library.

## Supported Compilers
* g++
* Clang
* Microsoft Visual C++

## Troubleshooting

* If you run into trouble finding CLP on Linux (e.g. Ubuntu 20.04), try extending the cmake prefix path:

    ```-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu```

* In case something went wrong fetching the git lfs content, try:
	```git lfs pull```
