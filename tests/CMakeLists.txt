########################################################################################################################
# GTest download and configuration
########################################################################################################################

set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

# Download and unpack googletest at configure time
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE resul
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include(GoogleTest)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

########################################################################################################################
# Run all tests target
########################################################################################################################

add_executable(runTests runTests.cpp)
target_include_directories(runTests PUBLIC ../include ${CMAKE_CURRENT_BINARY_DIR}/googletest-src/googletest/include)
# exclude coverage measurements, as they don't count lines correctly.
#target_compile_options(runTests  PUBLIC -fprofile-arcs -ftest-coverage)
#target_link_options(runTests  PUBLIC -fprofile-arcs -ftest-coverage)
target_link_libraries(runTests hops gtest_main gtest)

########################################################################################################################
# Test subdirectories
########################################################################################################################

add_subdirectory(FileReader)
add_subdirectory(FileWriter)
add_subdirectory(LinearProgram)
add_subdirectory(MarkovChain)
add_subdirectory(Model)
add_subdirectory(PolytopePreprocessing)
