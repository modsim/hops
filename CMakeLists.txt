########################################################################################################################
# CMAKE Settings
########################################################################################################################

cmake_minimum_required(VERSION 3.14)
project(nups)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

########################################################################################################################
# Build Parameters
########################################################################################################################

set(ENV{MKLROOT} "/home/jadebeck/intel/compilers_and_libraries/linux/mkl")
set(MKL_USE_interface lp64)

########################################################################################################################
# Set Options
########################################################################################################################

option(NUPS_BENCHMARKS "Enables compilation of Benchmarks (Requires Celero). Use -DNUPS_BENCHMARKS=OFF to disable." ON)
option(NUPS_DOCS "Enables generation of documentation. Use -DNUPS_DOCS=OFF to disable." ON)
option(NUPS_EXAMPLES "Enables compilation of Examples. Use -DNUPS_EXAMPLES=OFF to disable." ON)
option(NUPS_TESTS "Enables compilation of unit tests. Use -DNUPS_TESTS=OFF to disable." ON)

########################################################################################################################
# C++-Compiler Settings
########################################################################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    message(STATUS "Adding compiler flag march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else ()
    message(WARN "compiler does not support flag march=native")
endif ()

########################################################################################################################
# Definitions
########################################################################################################################

if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "Disabling Runtime checks: Using Release mode for nups and Eigen")
    add_definitions(-DNDEBUG)
else ()
    message(STATUS "Enabling Runtime checks.")
endif (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")

#add_definitions(-DEIGEN_USE_MKL_ALL)

########################################################################################################################
# Third-Party Dependencies
########################################################################################################################

find_package(Doxygen)
find_package(Eigen3 REQUIRED)
find_package(MKL)

message(STATUS "Executed command: \n${MKL_LINK_TOOL_COMMAND}")
message(STATUS "Found MKL_LIBRARIES:\n${MKL_LIBRARIES} ")
message(STATUS "Found MKL_DEFINITIONS:\n${MKL_DEFINITIONS} ")
message(STATUS "Found MKL_LIBRARY_DIR:\n${MKL_LIBRARY_DIR} ")
message(STATUS "Found MKL_INCLUDE_DIR:\n${MKL_INCLUDE_DIR} ")

########################################################################################################################
# NUPS
########################################################################################################################

add_library(nups STATIC "")
target_include_directories(nups PUBLIC ${EIGEN3_INCLUDE_DIR}
        ${MKL_INCLUDE_DIR}
        )
target_link_directories(nups PUBLIC
        ${MKL_LIBRARY_DIR}
        )
target_link_libraries(nups PUBLIC
        ${MKL_LIBRARIES}
        )
add_subdirectory(include)
add_subdirectory(src)

########################################################################################################################
# NUPS_DOCS
########################################################################################################################

if (NUPS_DOCS)
    if (NOT_DOXYGEN_FOUND)
        message(WARNING "Skipping Doxygen documentation, because Doxygen is not found.
                 Doxygen is needed to build the API docs.")
    else ()
        message(STATUS "Enabled Doxygen documentation. Use -DNUPS_DOCS=OFF to disable.")
        set(DOCUMENTATION_DIR "docs")
        set(doxy_main_page "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
        set(doxyfile_in "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in")
        set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile")
        configure_file(${doxyfile_in} ${doxyfile} @ONLY)
        add_custom_target(docs
                COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile} && make -C ${CMAKE_CURRENT_BINARY_DIR}/docs/latex
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API docs with Doxygen"
                VERBATIM
                )
    endif ()
else ()
    message(STATUS "Disabled Doxygen generation. Use -DNUPS_DOCS=ON to enable.")
endif ()

########################################################################################################################
# NUPS_BENCHMARKS
########################################################################################################################

if (NUPS_BENCHMARKS)
    message(STATUS "Enabled compilation of benchmarks. Use -DNUPS_BENCHMARKS=OFF to disable.")
    add_subdirectory(benchmarks)
else ()
    message(STATUS "Disabled compilation of benchmarks. Use -DNUPS_BENCHMARKS=ON to enable.")
endif ()

########################################################################################################################
# NUPS_EXAMPLES
########################################################################################################################

if (NUPS_EXAMPLES)
    message(STATUS "Enabled compilation of examples. Use -DNUPS_EXAMPLES=OFF to disable.")
    add_subdirectory(examples)
else ()
    message(STATUS "Disabled compilation of examples. Use -DNUPS_EXAMPLES=ON to enable.")
endif ()

########################################################################################################################
# NUPS_TESTS
########################################################################################################################

if (NUPS_TESTS)
    message(STATUS "Enabled compilation of tests. Use -DNUPS_TESTS=OFF to disable.")
    enable_testing()
    add_subdirectory(tests)
else ()
    message(STATUS "Disabled compilation of tests. Use -DNUPS_TESTS=ON to enable.")
endif ()


########################################################################################################################
# Resources
########################################################################################################################

if (NUPS_BENCHMARKS OR NUPS_EXAMPLES OR NUPS_TESTS)
    message(STATUS "Copying resources.")
    add_subdirectory(resources)
endif ()

########################################################################################################################
# Install
########################################################################################################################

# TODO install command
