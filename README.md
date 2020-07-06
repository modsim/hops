# The HOPS toolbox

The **H**ighly **O**ptimized **P**olytope **S**ampling toolbox is an open-source C++17
library for efficient and scalable MCMC algorithms for sampling convex-constrained spaces possibly
equipped with arbitrary target functions

## Installation

HOPS uses CMake as build system.  
See the Dockerfile for a demonstration on installing HOPS and its dependencies on Ubuntu 20.4.

### CMake options

* HOPS_BENCHMARKS (default OFF) - Enables compilation of Benchmarks (Requires Celero). Use -DHOPS_BENCHMARKS=ON to enable.
* HOPS_DOCS (default ON) - Enables generation of documentation. Use -DHOPS_DOCS=OFF to disable. (This creates the Doxygen file from which the docs have to be generated)
* HOPS_EXAMPLES (default ON) - Enables compilation of Examples. Use -DHOPS_EXAMPLES=OFF to disable.
* HOPS_TESTS (default ON) - Enables compilation of unit tests. Use -DHOPS_TESTS=OFF to disable.

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

* In case something went wrong fetching the git lfs content, try
	```git lfs pull```
