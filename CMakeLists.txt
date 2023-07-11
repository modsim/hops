########################################################################################################################
# CMAKE Settings
########################################################################################################################

cmake_minimum_required(VERSION 3.14)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(hops)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if (UNIX)
    message(STATUS "Adjusting CMAKE_PREFIX_PATH on UNIX for CLP")
    list(APPEND CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu)
endif (UNIX)

########################################################################################################################
# MSVC stuff
########################################################################################################################
if (CMAKE_GENERATOR MATCHES "Visual Studio")
    add_definitions(-D_USE_MATH_DEFINES)
    add_compile_options(/Ze)
    add_compile_options(/bigobj)
endif ()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

########################################################################################################################
# Build Parameters
########################################################################################################################

set(MKL_ROOT_DIR "~/intel/compilers_and_libraries/linux/mkl" CACHE PATH "Path to installed mkl directory.")
set(MKL_USE_interface lp64)

########################################################################################################################
# Set Options
########################################################################################################################

option(HOPS_DNEST4_SUPPORT "Enables Nested Sampling support with DNEST4. Use -DHOPS_DNEST4_SUPPORT=ON to enable." ON)
option(HOPS_BENCHMARKS "Enables compilation of Benchmarks (Requires Celero). Use -DHOPS_BENCHMARKS=ON to enable." OFF)
option(HOPS_DOCS "Enables generation of documentation. Use -DHOPS_DOCS=OFF to disable." ON)
option(HOPS_BINARIES "Enables compilation of hops executables. Use -DHOPS_BINARIES=OFF to disable." ON)
option(HOPS_EXAMPLES "Enables compilation of hops examples. Use -DHOPS_EXAMPLES=OFF to disable." ON)
option(HOPS_TESTS "Enables compilation of unit tests. Use -DHOPS_TESTS=OFF to disable." ON)
option(HOPS_NO_INSTALL "Disables installation. Use -DHOPS_NO_INSTALL=ON to disable installation." OFF)
option(HOPS_BUILD_NATIVE "Enable this option for maximum performance. Disable it when linking to third-party software that also uses Eigen3 but is not compiled with -march=native" ON)
set(HOPS_LIBRARY_TYPE SHARED CACHE STRING "Type of library to build. Options are HEADER_ONLY, STATIC or SHARED")
set_property(CACHE HOPS_LIBRARY_TYPE PROPERTY STRINGS HEADER_ONLY STATIC SHARED)

########################################################################################################################
# C++-Compiler Settings
########################################################################################################################

include(CheckCXXCompilerFlag)

set(CMAKE_C_FLAGS "${CMAKE_FLAGS} ${MPI_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif ()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS}")


if(HOPS_BUILD_NATIVE)
	message(STATUS "Building natively using -march=native")
	CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
	if (COMPILER_SUPPORTS_MARCH_NATIVE)
	    message(STATUS "Adding compiler flag march=native")
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
	else (COMPILER_SUPPORTS_MARCH_NATIVE)
	    message(WARNING "compiler does not support flag -march=native")
	endif (COMPILER_SUPPORTS_MARCH_NATIVE)
else(HOPS_BUILD_NATIVE)
	message(STATUS "NOT building natively using -march=native")
endif(HOPS_BUILD_NATIVE)



CHECK_CXX_COMPILER_FLAG("-O3" COMPILER_SUPPORTS_O3)
if (COMPILER_SUPPORTS_O3)
    message(STATUS "Adding compiler flag -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
else ()
    message(STATUS "Adding compiler flag -O2")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif ()


########################################################################################################################
# Pre-Third-Party-Search Definitions
########################################################################################################################

if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "Disabling Runtime checks: Using Release mode for hops and Eigen")
    add_definitions(-DNDEBUG)
else ()
    message(STATUS "Enabling Runtime checks.")
endif (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")

########################################################################################################################
# System-Installed Third-Party Dependencies
########################################################################################################################

find_package(Doxygen)
find_package(Eigen3 REQUIRED)
find_package(MKL)
message(STATUS "FOUND MKL ? ${MKL_FOUND}")

########################################################################################################################
# Post-Third-Party-Search Definitions
########################################################################################################################

if (MKL_LIBRARY_DIR)
    message(STATUS "Found MKL: Adding preprocessor definition EIGEN_USE_MKL_ALL")
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif (MKL_LIBRARY_DIR)

########################################################################################################################
# HOPS
########################################################################################################################

if (HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    message(STATUS "Set to header-only installation")
    set(SCOPE INTERFACE)
    add_library(hops INTERFACE)

    target_compile_definitions(hops INTERFACE HOPS_HEADER_ONLY)

    if (HOPS_DNEST4_SUPPORT)
        target_compile_definitions(hops INTERFACE HOPS_DNEST4_SUPPORT)
    endif (HOPS_DNEST4_SUPPORT)

    target_include_directories(hops INTERFACE ${EIGEN3_INCLUDE_DIR}
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
            )
else (HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    message(STATUS "Set to ${HOPS_LIBRARY_TYPE} library installation")
    set(SCOPE PRIVATE)
    add_library(hops ${HOPS_LIBRARY_TYPE})
    target_compile_definitions(hops ${SCOPE} HOPS_HDF5_SUPPORT)

    target_include_directories(hops PUBLIC ${EIGEN3_INCLUDE_DIR}
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
            )
endif (HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

add_subdirectory(src)


########################################################################################################################
# Linking libc++ and MKL support for faster linear algebra
########################################################################################################################

if (MKL_INCLUDE_DIR)
    target_include_directories(hops ${SCOPE} ${MKL_INCLUDE_DIR})
endif (MKL_INCLUDE_DIR)
if (MKL_LIBRARY_DIR)
    target_link_directories(hops ${SCOPE} ${MKL_LIBRARY_DIR})
endif (MKL_LIBRARY_DIR)
if (MKL_LIBRARIES)
    target_link_libraries(hops ${SCOPE} ${MKL_LIBRARIES})
endif (MKL_LIBRARIES)

########################################################################################################################
# HOPS_DOCS
########################################################################################################################

if (HOPS_DOCS)
    if (NOT_DOXYGEN_FOUND)
        message(WARNING "Skipping Doxygen documentation, because Doxygen is not found.
                 Doxygen is needed to build the API docs.")
    else ()
        message(STATUS "Enabled Doxygen documentation. Use -DHOPS_DOCS=OFF to disable.")
        set(DOCUMENTATION_DIR "docs")

        set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

        set(doxy_main_page "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
        set(doxyfile_in "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
        set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
        configure_file(${doxyfile_in} ${doxyfile} @ONLY)
        add_custom_target(docs
                COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile} && make -C ${CMAKE_CURRENT_BINARY_DIR}/docs/latex
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API docs with Doxygen"
                VERBATIM
                )
    endif ()
else ()
    message(STATUS "Disabled Doxygen generation. Use -DHOPS_DOCS=ON to enable.")
endif ()

########################################################################################################################
# HOPS_BENCHMARKS
########################################################################################################################

if (HOPS_BENCHMARKS)
    message(STATUS "Enabled compilation of benchmarks. Use -DHOPS_BENCHMARKS=OFF to disable.")
    add_subdirectory(benchmarks)
else ()
    message(STATUS "Disabled compilation of benchmarks. Use -DHOPS_BENCHMARKS=ON to enable.")
endif ()

########################################################################################################################
# HOPS_BINARIES
########################################################################################################################

if (HOPS_BINARIES)
    message(STATUS "Enabled compilation of executables. Use -DHOPS_BINARIES=OFF to disable.")
    add_subdirectory(bin)
else ()
    message(STATUS "Disabled compilation of executables. Use -DHOPS_BINARIES=ON to enable.")
endif ()

########################################################################################################################
# HOPS_EXAMPLES
########################################################################################################################

if (HOPS_EXAMPLES)
    message(STATUS "Enabled compilation of examples. Use -DHOPS_EXAMPLES=OFF to disable.")
    add_subdirectory(examples)
else ()
    message(STATUS "Disabled compilation of examples. Use -DHOPS_EXAMPLES=ON to enable.")
endif ()

########################################################################################################################
# HOPS_TESTS
########################################################################################################################

if (HOPS_TESTS)
    message(STATUS "Enabled compilation of tests. Use -DHOPS_TESTS=OFF to disable.")
    enable_testing()
    add_subdirectory(tests)
else ()
    message(STATUS "Disabled compilation of tests. Use -DHOPS_TESTS=ON to enable.")
endif ()



########################################################################################################################
# Resources
########################################################################################################################

if (HOPS_BENCHMARKS OR HOPS_EXAMPLES OR HOPS_TESTS)
    message(STATUS "Copying resources")
    add_subdirectory(resources)
endif ()

########################################################################################################################
# Install
########################################################################################################################

if (NOT HOPS_NO_INSTALL)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ DESTINATION include/
            FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.h")

    #    TODO reinclude when hops-sampler is fixed
    #    if (HOPS_BINARIES)
    #        install(TARGETS hops-sampler DESTINATION bin/)
    #    endif ()

    install(TARGETS hops EXPORT hops-config LIBRARY DESTINATION "lib")
    install(EXPORT hops-config DESTINATION "lib")
endif (NOT HOPS_NO_INSTALL)
