# The HOPS toolbox

[![Build Status](https://travis-ci.org/modsim/hops.svg?branch=master)](https://travis-ci.org/modsim/hops)

The **H**ighly **O**ptimized **P**olytope **S**ampling toolbox is an open-source C++17
library for efficient and scalable MCMC algorithms for sampling convex-constrained spaces possibly
equipped with arbitrary target functions.

For details and benchmarks see the application note https://doi.org/10.1093/bioinformatics/btaa872.
Test data from the application note is downloadable at https://doi.org/10.26165/JUELICH-DATA/YXLFKJ.


## Documentation

Documentation, interactive demos and further resources can be found at https://modsim.github.io/hops/.


## Cloning from Github

HOPS contains git submodules that point to third-party libraries.
For this reason, HOPS should be fetched recursively:

```
git clone git@github.com:modsim/hops.git --recursive
```

<img src="hops.png" alt="HOPS Logo" width="500"/>


## Installation

HOPS uses CMake as build system.  
See the Dockerfile for a demonstration on installing HOPS and its dependencies on Ubuntu 20.4.


### CMake options

| Option Name               | Default   | Description                                                                                               |
| ------------------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| HOPS\_HDF5\_SUPPORT       |       OFF | Enables HDF5 support with HighFive. Use -DHOPS\_BENCHMARKS=ON to enable.                                  |
| HOPS\_BENCHMARKS          |       OFF | Enables compilation of Benchmarks (Requires Celero). Use -DHOPS\_BENCHMARKS=ON to enable.                 |
| HOPS\_DOCS                |        ON | Enables generation of documentation. Use -DHOPS\_DOCS=OFF to disable. (This creates the Doxygen file fr om which the docs have to be generated) |
| HOPS\_BINARIES            |        ON | Enables compilation of hops executables. Use -DHOPS\_EXAMPLES=OFF to disable.                             |
| HOPS\_TESTS               |        ON | Enables compilation of unit tests. Use -DHOPS\_TESTS=OFF to disable.                                      |
| HOPS\_LIBRARY\_TYPE       |    SHARED | Type of library to build. Options are HEADER\_ONLY, STATIC or SHARED                                      |


#### Install on Linux:

```
$ mkdir cmake-build-release cd cmake-build-release      # create and switch into 
                                                        # directory for out-of-source build
$ cmake .. -DCMAKE_BUILD_TYPE=Release                   # run cmake
$ make                                                  # build hops
$ make test                                             # run Tests
$ sudo make install                                     # install
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
