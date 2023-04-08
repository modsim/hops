# The HOPS library

[![pipeline status](https://jugit.fz-juelich.de/IBG-1/ModSim/hops/badges/main/pipeline.svg)](https://jugit.fz-juelich.de/IBG-1/ModSim/hops/-/commits/main)
[![coverage report](https://jugit.fz-juelich.de/IBG-1/ModSim/hops/badges/main/coverage.svg)](https://jugit.fz-juelich.de/IBG-1/ModSim/hops/-/commits/main)
[![commits](https://jugit.fz-juelich.de/IBG-1/ModSim/hops/-/jobs/artifacts/main/raw/commits.svg?job=create_badges)](https://jugit.fz-juelich.de/IBG-1/ModSim/hops/-/commits/main)


Welcome to the HOPS library, the Highly Optimized Polytope Sampling library - a powerful open-source C++17 library for efficient and scalable MCMC algorithms. HOPS provides a versatile framework for sampling convex-constrained spaces equipped with arbitrary target functions. With HOPS, you can achieve optimized and scalable performance in your MCMC algorithms, making it a valuable tool for various applications.

The library is fully documented, with interactive demos and further resources available to help you get started quickly. Whether you're working on a research project or an industrial application, HOPS provides the performance and flexibility you need to efficiently sample your data.

You can also use HOPS with Python interface available at https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy.
We hope that the HOPS library will prove to be an invaluable tool in your work, and we invite you to explore its many features and benefits.

For details and benchmarks see the application note https://doi.org/10.1093/bioinformatics/btaa872.
Test data from the application note is downloadable at https://doi.org/10.26165/JUELICH-DATA/YXLFKJ.

## Documentation

Documentation, interactive demos and further resources can be found at https://modsim.github.io/hops/.

## Cloning from GitLab

HOPS contains git submodules that point to third-party libraries.
For this reason, HOPS should be fetched recursively:

```
git clone git@jugit.fz-juelich.de:IBG-1/ModSim/hops.git --recursive
```

<img src="hops.png" alt="HOPS Logo" width="500"/>


## Installation

HOPS uses CMake as build system.  
See the Dockerfile for a demonstration on installing HOPS and its dependencies on Ubuntu 20.4.

## Python Interface

Python interface is available at https://github.com/modsim/hopsy.

### CMake options

| Option Name               | Default   | Description                                                                                               |
| ------------------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| HOPS\_HDF5\_SUPPORT       |       OFF | Enables HDF5 support with HighFive. Use -DHOPS\_BENCHMARKS=ON to enable.                                  |
| HOPS\_BENCHMARKS          |       OFF | Enables compilation of Benchmarks (Requires Celero). Use -DHOPS\_BENCHMARKS=ON to enable.                 |
| HOPS\_DOCS                |        ON | Enables generation of documentation. Use -DHOPS\_DOCS=OFF to disable.                                     |
| HOPS\_BINARIES            |        ON | Enables compilation of hops executables. Use -DHOPS\_EXAMPLES=OFF to disable.                             |
| HOPS\_TESTS               |        ON | Enables compilation of unit tests. Use -DHOPS\_TESTS=OFF to disable.                                      |
| HOPS\_LIBRARY\_TYPE       |    SHARED | Type of library to build. Options are STATIC, SHARED or HEADER\_ONLY (deprecated)                         |


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
Note: Much easier to compile with clang than with MSCV.


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

* Compare to the Dockerfile to see if dependencies are missing in your local environment.


# Development

The development of HOPS primarily takes place on (JuGit)[https://jugit.fz-juelich.de/IBG-1/ModSim/hops], where we have access to powerful continuous integration and a Docker registry. The GitHub repository is only a mirror, so please report issues and make pull requests on JuGit.
