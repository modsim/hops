########################################################################################################################
# CMAKE Settings
########################################################################################################################

cmake_minimum_required(VERSION 3.14)
project(hops)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

########################################################################################################################
# Build Parameters
########################################################################################################################

# TODO replace by standard path and check if env is required
#set(ENV{MKLROOT} "/home/jadebeck/intel/compilers_and_libraries/linux/mkl" CACHE PATH "Path to installed mkl directory.")
set(MKL_ROOT_DIR "/home/jadebeck/intel/compilers_and_libraries/linux/mkl" CACHE PATH "Path to installed mkl directory.")
set(MKL_USE_interface lp64)

########################################################################################################################
# Set Options
########################################################################################################################

option(HOPS_BENCHMARKS "Enables compilation of Benchmarks (Requires Celero). Use -DHOPS_BENCHMARKS=OFF to disable." ON)
option(HOPS_DOCS "Enables generation of documentation. Use -DHOPS_DOCS=OFF to disable." ON)
option(HOPS_EXAMPLES "Enables compilation of Examples. Use -DHOPS_EXAMPLES=OFF to disable." ON)
option(HOPS_TESTS "Enables compilation of unit tests. Use -DHOPS_TESTS=OFF to disable." ON)

########################################################################################################################
# C++-Compiler Settings
########################################################################################################################

# TODO update to target_compile_feature and target_compile_option
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
# Pre-Third-Party-Search Definitions
########################################################################################################################

if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "Disabling Runtime checks: Using Release mode for hops and Eigen")
    add_definitions(-DNDEBUG)
else ()
    message(STATUS "Enabling Runtime checks.")
endif (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")

########################################################################################################################
# Third-Party Dependencies
########################################################################################################################

find_package(Doxygen)
find_package(Eigen3 REQUIRED)
find_package(MKL)
# TODO test mkl is found on ibt015
message(STATUS "FOUND MKL ?  ${MKL_FOUND}")

########################################################################################################################
# Post-Third-Party-Search Definitions
########################################################################################################################

if(MKL_LIBRARY_DIR)
    message(STATUS "Found MKL: Adding preprocessor definition EIGEN_USE_MKL_ALL")
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif(MKL_LIBRARY_DIR)

########################################################################################################################
# HOPS
########################################################################################################################

add_library(hops STATIC "")
target_include_directories(hops PUBLIC ${EIGEN3_INCLUDE_DIR}
        ${MKL_INCLUDE_DIR}
        )
target_link_directories(hops PUBLIC
        ${MKL_LIBRARY_DIR}
        )
target_link_libraries(hops PUBLIC
        ${MKL_LIBRARIES}
        )

add_subdirectory(include)
add_subdirectory(src)

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
    message(STATUS "Copying resources.")
    add_subdirectory(resources)
endif ()

########################################################################################################################
# Install
########################################################################################################################

install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/hops/hops.hpp DESTINATION include/hops)
install(TARGETS hops EXPORT hops-config LIBRARY DESTINATION "lib")
install(EXPORT hops-config DESTINATION "lib" CONFIGURATIONS RELEASE)
